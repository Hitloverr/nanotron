"""
可微分分布式通信原语 —— 张量并行的通信操作封装。

本模块将 PyTorch 的分布式集合通信操作（AllReduce、AllGather、ReduceScatter）
封装为可微分的 autograd.Function，使得这些通信操作可以无缝嵌入到
PyTorch 的自动微分系统中。

核心设计思路：
    在张量并行中，前向传播和反向传播需要使用不同的通信操作：
        - AllReduce 前向 → 梯度直接传递（无需额外通信）
        - AllGather 前向 → ReduceScatter 反向
        - ReduceScatter 前向 → AllGather 反向

    这些对应关系通过 autograd.Function 的 forward/backward 方法实现，
    确保梯度计算的正确性。

通信原语对应关系：
    ┌─────────────────────┬──────────────────────┐
    │ 前向传播操作         │ 反向传播操作          │
    ├─────────────────────┼──────────────────────┤
    │ identity            │ AllReduce(SUM)       │
    │ AllReduce(SUM)      │ identity             │
    │ AllGather           │ ReduceScatter(SUM)   │
    │ ReduceScatter(SUM)  │ AllGather            │
    └─────────────────────┴──────────────────────┘

与张量并行层的关系：
    - TensorParallelColumnLinear (ALL_REDUCE): 前向 identity，反向 AllReduce
    - TensorParallelRowLinear (ALL_REDUCE): 前向 AllReduce，反向 identity
    - TensorParallelColumnLinear (REDUCE_SCATTER): 前向 AllGather，反向 ReduceScatter
    - TensorParallelRowLinear (REDUCE_SCATTER): 前向 ReduceScatter，反向 AllGather
    - TiedLinear (ALL_REDUCE): 前向 identity，反向 AllReduce
    - TiedLinear (REDUCE_SCATTER): 前向 AllGather，反向 ReduceScatter
"""

from typing import Optional

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup


class DifferentiableIdentity(torch.autograd.Function):
    """可微分的恒等操作，反向传播时执行 AllReduce。

    前向传播时不执行任何操作（直接透传张量），
    反向传播时对梯度执行 AllReduce 求和。

    用途：TensorParallelColumnLinear 的 ALL_REDUCE 模式。
    ColumnLinear 的输出是部分结果，不需要前向通信。
    但反向传播时，输入梯度需要 AllReduce 聚合。
    """

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    """可微分的 AllReduce 求和操作。

    前向传播时对张量执行 AllReduce SUM 操作，
    反向传播时梯度直接传递（无需额外通信）。

    用途：TensorParallelRowLinear 的 ALL_REDUCE 模式。
    RowLinear 的输出是部分结果，需要 AllReduce 聚合。
    反向传播时，由于 AllReduce 的梯度就是输入本身，
    不需要额外通信。
    """

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        if group.size() == 1:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DifferentiableAllGather(torch.autograd.Function):
    """可微分的 AllGather 操作。

    前向传播时将各 rank 的张量沿第 0 维拼接（AllGather），
    反向传播时对梯度执行 ReduceScatter SUM 操作。

    用途：
        - TensorParallelColumnLinear 的 REDUCE_SCATTER 模式：
          前向时 AllGather 输入，反向时 ReduceScatter 梯度
        - TiedLinear 的 REDUCE_SCATTER 模式：
          前向时 AllGather 输出，反向时 ReduceScatter 梯度

    注意：当前实现沿第 0 维（batch 维度）进行 gather/scatter，
    这是序列并行的标准做法。
    """

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        sharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        unsharded_batch_size = sharded_batch_size * group.size()

        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # NCCL 要求张量是连续的
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        out = DifferentiableReduceScatterSum.apply(grad_output, group)
        return out, None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    """可微分的 ReduceScatter 求和操作。

    前向传播时对张量执行 ReduceScatter SUM 操作（沿第 0 维分散），
    反向传播时对梯度执行 AllGather 操作。

    用途：
        - TensorParallelRowLinear 的 REDUCE_SCATTER 模式：
          前向时 ReduceScatter 输出，反向时 AllGather 梯度
        - TensorParallelEmbedding 的 REDUCE_SCATTER 模式

    注意：当前实现沿第 0 维（batch 维度）进行 scatter/gather，
    这是序列并行的标准做法。
    """

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        unsharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        assert unsharded_batch_size % group.size() == 0

        # NCCL 要求张量是连续的
        tensor = tensor.contiguous()

        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=False,
        )
        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllGather.apply(grad_output, group), None


# -----------------
# 辅助函数
# -----------------


def differentiable_identity(tensor, group: Optional[ProcessGroup] = None):
    """可微分的恒等操作，反向传播时执行 AllReduce。"""
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    """可微分的 AllReduce 求和操作。"""
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    """可微分的 AllGather 操作。"""
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    """可微分的 ReduceScatter 求和操作。"""
    return DifferentiableReduceScatterSum.apply(tensor, group)
