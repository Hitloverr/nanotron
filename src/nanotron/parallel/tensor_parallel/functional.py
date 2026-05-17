# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
张量并行函数式接口 —— Column/Row 线性层和分片交叉熵的实现。

本模块实现了张量并行中的核心计算函数，包括：
    - column_linear: 张量并行的列切分线性层
    - row_linear: 张量并行的行切分线性层
    - sharded_cross_entropy: 分片词表的交叉熵损失

Column/Row 线性层的并行策略：
    在 Transformer 的 MLP 中，典型的并行组合是 ColumnLinear + RowLinear：
        - ColumnLinear: 将权重沿输出维度切分（列切分），每个 TP rank 持有部分输出
        - RowLinear: 将权重沿输入维度切分（行切分），需要聚合部分结果

    两种通信模式下的数据流：

    ALL_REDUCE 模式（标准张量并行）：
        ColumnLinear: input → identity → F.linear → partial_output
        RowLinear: partial_input → F.linear → AllReduce(SUM) → output

    REDUCE_SCATTER 模式（序列并行）：
        ColumnLinear: input → AllGather → F.linear → output
        RowLinear: input → F.linear → ReduceScatter(SUM) → output

异步通信优化：
    当 async_communication=True 时，通信操作与计算重叠执行：
        - ColumnLinear: 在 AllGather 通信的同时，计算本地分片的矩阵乘法
        - RowLinear: 在 AllGather 梯度的同时，计算本地分片的梯度

    这依赖于 CUDA_DEVICE_MAX_CONNECTIONS=1 的设置，确保通信和计算
    在不同的 CUDA Stream 上执行，实现真正的重叠。

分片交叉熵（Sharded Cross Entropy）：
    当词表大小很大时（如 128K），交叉熵计算的显存占用很高。
    分片交叉熵将词表切分到多个 TP rank 上，每个 rank 只计算
    本地词表部分的 softmax 和损失，通过 AllReduce 聚合结果。

    算法流程：
        1. 跨 rank 求全局 logits 最大值（数值稳定性）
        2. 计算本地词表部分的 exp(logits) 和
        3. 通过 AllReduce 获取全局 exp(logits) 和
        4. 计算损失：loss = log(sum_exp) - logit[target]
        5. 反向传播时，softmax 作为梯度基础
"""

import math
from typing import Optional

import torch
from torch.nn import functional as F

import nanotron.distributed as dist
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_reduce_scatter_sum,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.utils import MemoryBuffer


class _ShardedCrossEntropy(torch.autograd.Function):
    """分片交叉熵损失，支持词表沿 TP 维度切分。

    当词表被切分到多个 TP rank 时，每个 rank 只持有部分 logits。
    该函数在分片 logits 上计算交叉熵损失，通过 AllReduce 聚合
    跨 rank 的信息。

    前向传播算法：
        1. 跨 rank 求 logits 最大值（数值稳定性，避免 exp 溢出）
        2. 减去最大值后，计算本地 exp(logits) 之和
        3. AllReduce SUM 获取全局 exp(logits) 之和
        4. 计算损失：loss = log(global_sum_exp) - logit[target]
        5. 保存 softmax 概率用于反向传播

    反向传播算法：
        梯度 = softmax - one_hot(target)
        由于词表是分片的，只有持有 target 的 rank 需要减去 1，
        其他 rank 的 target 位置被 mask 为 0。
    """

    @staticmethod
    def forward(
        ctx,
        sharded_logits,  # (*, sharded_hidden_size)
        target,  # (*)
        group: dist.ProcessGroup,
    ):
        # 跨 rank 求 logits 最大值，用于数值稳定性
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        # 减去最大值，避免 exp 溢出
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        # 计算当前 rank 持有的词表范围
        sharded_hidden_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_hidden_size
        end_index = start_index + sharded_hidden_size

        # 创建 target mask：1 表示该 target 不在当前 rank 的词表范围内
        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        # 获取 target 位置的 logit 值
        logits_2d = sharded_logits.view(-1, sharded_hidden_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # AllReduce 获取所有 rank 上 target 位置的 logit 值
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # 计算全局 exp(logits) 之和
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        # 计算交叉熵损失：loss = log(sum_exp) - logit[target]
        loss = torch.log(sum_exp_logits) - predicted_logits

        # 归一化得到 softmax 概率，用于反向传播
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # 梯度基础：softmax 概率
        grad_input = softmax
        sharded_hidden_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_hidden_size)

        # 在 target 位置减去 1（softmax - one_hot 的梯度）
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # 乘以输出梯度
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


class _ShardedCrossEntropyWithZLoss(torch.autograd.Function):
    """带 Z-Loss 正则化的分片交叉熵损失。

    Z-Loss 是一种正则化技术，通过惩罚 log(Z)² 来稳定训练，
    其中 Z 是 softmax 的配分函数（所有 exp(logits) 之和）。

    z_loss = z_loss_coef * log²(Z)

    Z-Loss 的作用：
        - 防止 logits 过大，导致 exp 溢出
        - 稳定混合精度训练中的 softmax 计算
        - 典型 z_loss_coef 值：1e-4 到 1e-2
    """

    @staticmethod
    def forward(
        ctx,
        sharded_logits,  # (batch_size, length, sharded_hidden_size)
        target,  # (batch_size, length)
        group: dist.ProcessGroup,
        z_loss_coef: float = 0.0,
    ):
        # 与 _ShardedCrossEntropy 相同的数值稳定性处理
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        sharded_hidden_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_hidden_size
        end_index = start_index + sharded_hidden_size

        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        logits_2d = sharded_logits.view(-1, sharded_hidden_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(sum_exp_logits) - predicted_logits

        # 保存 log(Z) 用于 Z-Loss 计算
        log_z = torch.log(sum_exp_logits)

        # Z-Loss 正则化：z_loss = z_loss_coef * log²(Z)
        if z_loss_coef > 0.0:
            z_loss = z_loss_coef * torch.square(log_z.clamp(min=-20.0, max=20.0))
            loss = loss + z_loss
        else:
            z_loss = torch.zeros_like(loss)

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, log_z)
        ctx.z_loss_coef = z_loss_coef

        return loss.view_as(target), z_loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output, grad_z_loss=None):
        softmax, target_mask, masked_target_1d, log_z = ctx.saved_tensors
        z_loss_coef = ctx.z_loss_coef

        grad_input = softmax.clone()
        sharded_hidden_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_hidden_size)

        # 交叉熵梯度：softmax - one_hot
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Z-Loss 梯度：2 * z_loss_coef * log(Z) * softmax
        # 因为 d/d(logit) [log(Z)] = softmax
        if z_loss_coef > 0.0:
            z_loss_grad_scale = 2.0 * z_loss_coef * log_z
            z_loss_grad = softmax * z_loss_grad_scale.unsqueeze(-1)
            grad_input = grad_input + z_loss_grad

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None


def sharded_cross_entropy(
    sharded_logits,
    target,
    group: dist.ProcessGroup,
    dtype: torch.dtype = None,
    z_loss_coef: float = 0.0,
):
    """分片交叉熵损失的函数式接口。

    根据是否启用 Z-Loss 正则化，选择不同的实现。

    Args:
        sharded_logits: 分片 logits，形状 (*, sharded_vocab_size)。
        target: 目标标签，形状 (*)。
        group: TP 进程组。
        dtype: 可选的数据类型转换。
        z_loss_coef: Z-Loss 正则化系数，0 表示不使用。

    Returns:
        torch.Tensor: 交叉熵损失。
    """
    if dtype is not None:
        sharded_logits = sharded_logits.to(dtype=dtype)
    if z_loss_coef > 0.0:
        return _ShardedCrossEntropyWithZLoss.apply(sharded_logits, target, group, z_loss_coef)
    else:
        return _ShardedCrossEntropy.apply(sharded_logits, target, group)


class _ColumnLinearAsyncCommunication(torch.autograd.Function):
    """异步通信的列切分线性层。

    在 REDUCE_SCATTER 模式下，ColumnLinear 需要 AllGather 输入。
    异步通信将 AllGather 与本地分片的矩阵乘法重叠执行，
    隐藏通信延迟。

    通信与计算重叠策略：
        前向传播时，将输出按 batch 维度分为三段：
        - before_shard: AllGather 数据中当前 rank 之前的部分
        - same_device_shard: 当前 rank 持有的数据（可在 AllGather 同时计算）
        - after_shard: AllGather 数据中当前 rank 之后的部分

        执行顺序：
        1. 启动 AllGather 异步通信
        2. 同时计算 same_device_shard（使用本地数据）
        3. 等待 AllGather 完成
        4. 计算 before_shard 和 after_shard（使用 AllGather 的数据）

    参考：Megatron-LM 的异步通信实现
    """

    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode, tp_recompute_allgather):
        ctx.use_bias = bias is not None
        ctx.tp_mode = tp_mode
        ctx.group = group
        ctx.tp_recompute_allgather = tp_recompute_allgather
        ctx.tensor_shape = tensor.size()

        if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            # ALL_REDUCE 模式：不需要 AllGather，直接计算
            gathered_tensor = tensor
            ctx.save_for_backward(tensor, weight)
            return F.linear(gathered_tensor, weight, bias)
        elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            group_size = group.size()
            current_rank = dist.get_rank(group)
            if group_size == 1:
                gathered_tensor = tensor
                ctx.save_for_backward(tensor, weight)
                return F.linear(gathered_tensor, weight, bias)
            else:
                tensor = tensor.contiguous()

                sharded_batch_size, *intermediate_size, hidden_size = tensor.shape
                if group is None:
                    group = dist.distributed_c10d._get_default_group()
                gathered_batch_size = sharded_batch_size * group.size()

                # 分配 AllGather 输出缓冲区
                if tp_recompute_allgather:
                    # 使用全局内存缓冲区，反向传播时重新 AllGather
                    gathered_tensor = MemoryBuffer().get(
                        "allgather", (gathered_batch_size, *intermediate_size, hidden_size), dtype=tensor.dtype
                    )
                else:
                    gathered_tensor = torch.empty(
                        gathered_batch_size,
                        *intermediate_size,
                        hidden_size,
                        device=tensor.device,
                        dtype=tensor.dtype,
                        requires_grad=False,
                    )

                # 启动异步 AllGather
                handle = dist.all_gather_into_tensor(gathered_tensor, tensor, group=group, async_op=True)

                # 在 AllGather 通信的同时，计算本地分片的矩阵乘法
                output_size = weight.shape[0]
                gathered_output = torch.empty(
                    gathered_batch_size,
                    *intermediate_size,
                    output_size,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    requires_grad=tensor.requires_grad,
                )
                # 将输出分为三段：before_shard | same_device_shard | after_shard
                before_shard, same_device_shard, after_shard = torch.split(
                    gathered_output,
                    split_size_or_sections=[
                        sharded_batch_size * current_rank,
                        sharded_batch_size,
                        sharded_batch_size * (group_size - current_rank - 1),
                    ],
                    dim=0,
                )
                # 计算本地分片（与 AllGather 重叠）
                first_dims = math.prod([sharded_batch_size, *intermediate_size])
                if bias is None:
                    torch.mm(
                        input=tensor.view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=same_device_shard.view(first_dims, output_size),
                    )
                else:
                    torch.addmm(
                        input=bias[None, :],
                        mat1=tensor.view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=same_device_shard.view(first_dims, output_size),
                    )

                # 等待 AllGather 完成
                handle.wait()
                if tp_recompute_allgather:
                    ctx.save_for_backward(tensor, weight)
                else:
                    ctx.save_for_backward(gathered_tensor, weight)

                # 计算 AllGather 数据的其他分片
                if before_shard.numel() > 0:
                    first_dims = math.prod(before_shard.shape[:-1])
                    if bias is None:
                        torch.mm(
                            input=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                            mat2=weight.t(),
                            out=before_shard.view(first_dims, output_size),
                        )
                    else:
                        torch.addmm(
                            input=bias[None, :],
                            mat1=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                            mat2=weight.t(),
                            out=before_shard.view(first_dims, output_size),
                        )
                if after_shard.numel() > 0:
                    first_dims = math.prod(after_shard.shape[:-1])
                    if bias is None:
                        torch.mm(
                            input=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                                first_dims, hidden_size
                            ),
                            mat2=weight.t(),
                            out=after_shard.view(first_dims, output_size),
                        )
                    else:
                        torch.addmm(
                            input=bias[None, :],
                            mat1=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                                first_dims, hidden_size
                            ),
                            mat2=weight.t(),
                            out=after_shard.view(first_dims, output_size),
                        )

                return gathered_output
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")

    @staticmethod
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias
        tp_mode = ctx.tp_mode

        handle1: Optional[dist.Work] = None
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER and ctx.tp_recompute_allgather:
            # 反向传播时重新 AllGather（节省显存，增加通信）
            sharded_batch_size, *rest_size = tensor.shape
            if group is None:
                group = dist.distributed_c10d._get_default_group()

            if group.size() == 1:
                total_tensor = tensor
            else:
                unsharded_batch_size = sharded_batch_size * group.size()

                unsharded_tensor = MemoryBuffer().get(
                    "allgather", (unsharded_batch_size, *rest_size), dtype=tensor.dtype
                )
                handle1 = dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group, async_op=True)
                total_tensor = unsharded_tensor
        else:
            total_tensor = tensor

        # 计算输入梯度
        grad_tensor = grad_output.matmul(weight)

        grad_output = grad_output.contiguous()
        grad_output_first_dims, grad_output_last_dim = grad_output.shape[:-1], grad_output.shape[-1]
        total_tensor_first_dims, total_tensor_last_dim = total_tensor.shape[:-1], total_tensor.shape[-1]
        grad_output = grad_output.view(math.prod(grad_output_first_dims), grad_output_last_dim)
        total_tensor = total_tensor.view(math.prod(total_tensor_first_dims), total_tensor_last_dim)

        # 异步通信输入梯度
        handle2: Optional[dist.Work] = None
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            if group.size() == 1:
                sub_grad_tensor = grad_tensor
            else:
                sub_grad_tensor = torch.empty(
                    ctx.tensor_shape, dtype=grad_tensor.dtype, device=grad_tensor.device, requires_grad=False
                )
                handle2 = dist.reduce_scatter_tensor(sub_grad_tensor, grad_tensor, group=group, async_op=True)
        elif tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            handle2 = dist.all_reduce(grad_tensor, group=group, async_op=True)
        else:
            raise ValueError()

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if handle1 is not None:
            handle1.wait()

        # 计算权重梯度（与通信重叠）
        grad_weight = grad_output.t().matmul(total_tensor)

        if handle2 is not None:
            handle2.wait()

        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            return sub_grad_tensor, grad_weight, grad_bias, None, None, None
        elif tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            return grad_tensor, grad_weight, grad_bias, None, None, None
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")


class _ColumnLinearNoAsyncCommunicationReduceScatterMode(torch.autograd.Function):
    """非异步通信的列切分线性层（REDUCE_SCATTER 模式）。

    与异步版本相比，先完成 AllGather，再执行矩阵乘法。
    实现更简单，但通信延迟无法被计算隐藏。
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group: dist.ProcessGroup,
        tp_recompute_allgather: bool,
    ):

        # 执行 AllGather
        sharded_batch_size, *rest_size = input.shape
        unsharded_batch_size = sharded_batch_size * group.size()
        if group.size() == 1:
            total_input = input.contiguous()
        elif tp_recompute_allgather:
            total_input = MemoryBuffer().get("allgather", (unsharded_batch_size, *rest_size), dtype=input.dtype)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)
        else:
            total_input = torch.empty(unsharded_batch_size, *rest_size, dtype=input.dtype, device=input.device)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)

        ctx.group = group
        ctx.tp_recompute_allgather = tp_recompute_allgather
        ctx.input_size = input.shape
        if tp_recompute_allgather:
            ctx.save_for_backward(input, weight, bias)
        else:
            ctx.save_for_backward(total_input, weight, bias)

        out = F.linear(total_input, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx.group
        tp_recompute_allgather = ctx.tp_recompute_allgather
        input_size = ctx.input_size
        if group.size() == 1 or not tp_recompute_allgather:
            total_input, weight, bias = ctx.saved_tensors
        else:
            input, weight, bias = ctx.saved_tensors
            sharded_batch_size, *rest_size = input.shape
            total_input = sharded_batch_size * group.size()
            unsharded_batch_size = sharded_batch_size * group.size()
            total_input = MemoryBuffer().get("allgather", (unsharded_batch_size, *rest_size), dtype=input.dtype)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)

        grad_output = grad_output.contiguous()
        grad_output_first_dims, grad_output_last_dim = grad_output.shape[:-1], grad_output.shape[-1]
        total_input_first_dims, total_input_last_dim = total_input.shape[:-1], total_input.shape[-1]
        grad_output = grad_output.view(math.prod(grad_output_first_dims), grad_output_last_dim)
        total_input = total_input.view(math.prod(total_input_first_dims), total_input_last_dim)

        grad_weight = grad_output.T @ total_input
        grad_input = grad_output @ weight
        if group.size() == 1:
            sub_grad_input = grad_input
        else:
            grad_input = grad_input.contiguous()
            sub_grad_input = torch.empty(
                input_size, dtype=total_input.dtype, device=total_input.device, requires_grad=False
            )
            dist.reduce_scatter_tensor(sub_grad_input, grad_input, group=group, op=dist.ReduceOp.SUM)
        grad_bias = torch.sum(grad_output, dim=0) if bias is not None else None

        return sub_grad_input, grad_weight, grad_bias, None, None


def column_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
    tp_recompute_allgather: bool = True,
):
    """张量并行的列切分线性层。

    ColumnLinear 将权重沿输出维度切分，每个 TP rank 持有部分输出通道。
    根据通信模式和是否异步，选择不同的实现。

    Args:
        input: 输入张量。
        weight: 权重张量，形状 [output_size/tp, input_size]。
        bias: 偏置张量，形状 [output_size/tp]。
        group: TP 进程组。
        tp_mode: 通信模式（ALL_REDUCE 或 REDUCE_SCATTER）。
        async_communication: 是否使用异步通信。
        tp_recompute_allgather: 是否在反向传播时重新 AllGather。

    Returns:
        torch.Tensor: 线性层输出。
    """
    if async_communication:
        return _ColumnLinearAsyncCommunication.apply(input, weight, bias, group, tp_mode, tp_recompute_allgather)

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        # ALL_REDUCE 模式：输入不需要通信，反向传播时 AllReduce 梯度
        input = differentiable_identity(input, group=group)
        return F.linear(input, weight, bias)
    if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        return _ColumnLinearNoAsyncCommunicationReduceScatterMode.apply(
            input, weight, bias, group, tp_recompute_allgather
        )
    raise ValueError(f"Got unexpected mode: {tp_mode}.")


class _RowLinearAsyncCommunication(torch.autograd.Function):
    """异步通信的行切分线性层（仅支持 REDUCE_SCATTER 模式）。

    RowLinear 的异步通信在反向传播中实现：
        - 前向传播：计算线性变换 + ReduceScatter（同步）
        - 反向传播：AllGather 梯度输出 + 计算梯度（异步重叠）

    反向传播的通信与计算重叠策略：
        1. 启动 AllGather 梯度输出的异步通信
        2. 同时计算本地分片的输入梯度
        3. 等待 AllGather 完成
        4. 计算 AllGather 数据的其他分片的输入梯度
        5. 计算权重梯度
    """

    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode):
        assert (
            tp_mode is TensorParallelLinearMode.REDUCE_SCATTER
        ), f"async communication in RowLinear only supports REDUCE_SCATTER, got {tp_mode}"

        if group is None:
            group = dist.distributed_c10d._get_default_group()

        ctx.use_bias = bias is not None
        ctx.group = group

        out = F.linear(tensor, weight, bias)

        if group.size() > 1:
            out = differentiable_reduce_scatter_sum(out, group=group)

        ctx.save_for_backward(tensor, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias

        handle: Optional[dist.Work] = None

        sharded_batch_size, *rest_size = grad_output.shape

        if group.size() == 1:
            total_grad_output = grad_output
        else:
            unsharded_batch_size = sharded_batch_size * group.size()

            total_grad_output = MemoryBuffer().get(
                "allgather2", (unsharded_batch_size, *rest_size), dtype=tensor.dtype
            )

            grad_output = grad_output.contiguous()

            # 启动异步 AllGather 梯度输出
            handle = dist.all_gather_into_tensor(total_grad_output, grad_output, group=group, async_op=True)

        # 在 AllGather 通信的同时，计算本地分片的输入梯度
        sharded_batch_size, *rest_size_grad_output = grad_output.shape
        rest_size_grad_tensor = rest_size_grad_output[:-1] + [weight.shape[1]]

        if group.size() == 1:
            total_grad_tensor = grad_output.matmul(weight)
        else:
            unsharded_batch_size = sharded_batch_size * group.size()
            total_grad_tensor = torch.empty(
                unsharded_batch_size,
                *rest_size_grad_tensor,
                device=grad_output.device,
                dtype=grad_output.dtype,
                requires_grad=False,
            )
            before_shard_grad_tensor, same_device_shard_grad_tensor, after_shard_grad_tensor = torch.split(
                total_grad_tensor,
                split_size_or_sections=[
                    sharded_batch_size * dist.get_rank(group),
                    sharded_batch_size,
                    sharded_batch_size * (group.size() - dist.get_rank(group) - 1),
                ],
                dim=0,
            )
            # 计算本地分片的输入梯度（与 AllGather 重叠）
            torch.mm(
                input=grad_output.view(-1, grad_output.shape[-1]),
                mat2=weight,
                out=same_device_shard_grad_tensor.view(-1, weight.shape[1]),
            )

            if handle is not None:
                handle.wait()

            before_shard_grad_output, _, after_shard_grad_output = torch.split(
                total_grad_output,
                split_size_or_sections=[
                    sharded_batch_size * dist.get_rank(group),
                    sharded_batch_size,
                    sharded_batch_size * (group.size() - dist.get_rank(group) - 1),
                ],
                dim=0,
            )

            # 计算 AllGather 数据的其他分片的输入梯度
            if before_shard_grad_tensor.numel() > 0:
                torch.mm(
                    input=before_shard_grad_output.view(-1, before_shard_grad_output.shape[-1]),
                    mat2=weight,
                    out=before_shard_grad_tensor.view(-1, weight.shape[1]),
                )
            if after_shard_grad_tensor.numel() > 0:
                torch.mm(
                    input=after_shard_grad_output.view(-1, after_shard_grad_output.shape[-1]),
                    mat2=weight,
                    out=after_shard_grad_tensor.view(-1, weight.shape[1]),
                )

        tensor = tensor.contiguous()
        tensor_first_dims, tensor_last_dim = tensor.shape[:-1], tensor.shape[-1]
        tensor = tensor.view(math.prod(tensor_first_dims), tensor_last_dim)

        total_grad_output_first_dims, total_grad_output_last_dim = (
            total_grad_output.shape[:-1],
            total_grad_output.shape[-1],
        )
        total_grad_output = total_grad_output.view(math.prod(total_grad_output_first_dims), total_grad_output_last_dim)

        grad_weight = total_grad_output.t().matmul(tensor)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None

        return total_grad_tensor, grad_weight, grad_bias, None, None


def row_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
):
    """张量并行的行切分线性层。

    RowLinear 将权重沿输入维度切分，每个 TP rank 持有部分输入通道。
    输出是部分结果的聚合。

    Args:
        input: 输入张量。
        weight: 权重张量，形状 [output_size, input_size/tp]。
        bias: 偏置张量，形状 [output_size]（仅在 rank 0 上非 None）。
        group: TP 进程组。
        tp_mode: 通信模式（ALL_REDUCE 或 REDUCE_SCATTER）。
        async_communication: 是否使用异步通信。

    Returns:
        torch.Tensor: 线性层输出。
    """
    if async_communication:
        return _RowLinearAsyncCommunication.apply(input, weight, bias, group, tp_mode)

    out = F.linear(input, weight, bias)

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        # ALL_REDUCE 模式：AllReduce 聚合部分结果
        out = differentiable_all_reduce_sum(out, group=group)
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        # REDUCE_SCATTER 模式：ReduceScatter 分散聚合结果
        out = differentiable_reduce_scatter_sum(out, group=group)
    else:
        raise ValueError(f"Got unexpected mode: {tp_mode}.")

    return out
