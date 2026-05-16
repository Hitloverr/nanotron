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
张量并行线性层模块 —— Nanotron 张量并行的核心神经网络组件。

本模块提供了支持张量并行（Tensor Parallelism, TP）的神经网络层实现，
将大型线性层和嵌入层切分到多个 GPU 上，以突破单卡内存限制。

核心组件：
    - TensorParallelColumnLinear: 列切分线性层，权重沿输出维度切分
    - TensorParallelRowLinear: 行切分线性层，权重沿输入维度切分
    - TiedLinear: 绑定权重线性层，多个 TP rank 共享相同权重
    - TensorParallelEmbedding: 张量并行嵌入层，词表沿词元维度切分

张量并行的两种通信模式：
    - ALL_REDUCE: 前向传播后使用 AllReduce 聚合结果，反向传播时无需额外通信
    - REDUCE_SCATTER: 前向传播后使用 ReduceScatter 分散结果，与后续 ColumnLinear
      配合使用时可以避免一次 AllReduce，形成 "序列并行" 优化

典型组合模式：
    ColumnLinear(ALL_REDUCE) → RowLinear(ALL_REDUCE):
        标准张量并行，两次 AllReduce
    ColumnLinear(REDUCE_SCATTER) → RowLinear(ALL_REDUCE):
        序列并行优化，省去一次 AllReduce
"""

import os
from typing import Optional, Tuple

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.distributed import get_global_rank
from nanotron.logging import get_logger
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.sharded_parameters import (
    SplitConfig,
    create_sharded_parameter_from_config,
    mark_all_parameters_in_module_as_sharded,
)
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_reduce_scatter_sum,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.functional import (
    column_linear,
    row_linear,
)
from nanotron.parallel.tied_parameters import create_tied_parameter

logger = get_logger(__name__)


class TensorParallelColumnLinear(nn.Linear):
    """列切分张量并行线性层。

    将标准线性层的权重矩阵沿输出维度（第 0 维，即列方向）切分到多个 GPU 上。
    每个 GPU 持有权重的一个列切片，计算输出的对应部分。

    切分方式：
        完整权重 W: [out_features, in_features]
        每个 rank 持有: W_i: [out_features/TP, in_features]
        输出: y_i = x @ W_i^T + bias_i，形状为 [batch, out_features/TP]

    通信模式：
        - ALL_REDUCE: 前向时先 AllGather 输入（如果需要），输出不需要通信
        - REDUCE_SCATTER: 前向时 AllGather 输入，输出通过 ReduceScatter 分散

    Attributes:
        pg (ProcessGroup): 张量并行进程组。
        world_size (int): 张量并行度（TP 大小）。
        mode (TensorParallelLinearMode): 通信模式（ALL_REDUCE 或 REDUCE_SCATTER）。
        async_communication (bool): 是否使用异步通信。
        tp_recompute_allgather (bool): 是否在反向传播时重新计算 AllGather 而非缓存。
    """

    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
        tp_recompute_allgather: bool = True,
    ):
        """初始化列切分张量并行线性层。

        Args:
            in_features (int): 输入特征维度（未切分）。
            out_features (int): 输出特征维度（未切分），必须能被 TP 大小整除。
            pg (dist.ProcessGroup): 张量并行进程组。
            mode (TensorParallelLinearMode): 通信模式。
            bias (bool, optional): 是否使用偏置。默认为 True。
            device: 设备类型。
            dtype: 数据类型。
            async_communication (bool, optional): 是否使用异步通信。默认为 False。
            contiguous_chunks (Optional[Tuple[int, ...]], optional): 连续块大小配置，
                用于将输出维度划分为多个连续块分别切分。默认为 None。
            tp_recompute_allgather (bool, optional): 是否在反向传播时重新计算 AllGather
                而非缓存前向传播的 AllGather 结果。默认为 True。
                设为 True 可节省显存但增加计算量。

        Raises:
            AssertionError: 当 out_features 不能被 TP 大小整除时。
            AssertionError: 当 TP>1 但 CUDA_DEVICE_MAX_CONNECTIONS 不为 "1" 时。
        """
        self.pg = pg
        self.world_size = pg.size()

        assert out_features % self.world_size == 0

        self.in_features = in_features
        self.out_features = out_features // self.world_size
        self.tp_recompute_allgather = tp_recompute_allgather

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.mode = mode
        self.async_communication = async_communication

        if self.world_size > 1:
            assert (
                os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", None) == "1"
            ), "Env variable CUDA_DEVICE_MAX_CONNECTIONS should be set to 1 when using TP>1"

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == out_features
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to out_features ({out_features})"
        split_config = SplitConfig(split_dim=0, contiguous_chunks=contiguous_chunks)

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=self.pg,
            split_config=split_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行列切分线性层的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 [..., in_features]。

        Returns:
            torch.Tensor: 输出张量，形状取决于通信模式：
                - ALL_REDUCE: [..., out_features/TP]
                - REDUCE_SCATTER: [..., out_features/TP]
        """
        return column_linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
            group=self.pg,
            tp_mode=self.mode,
            async_communication=self.async_communication,
            tp_recompute_allgather=self.tp_recompute_allgather,
        )

    def extra_repr(self) -> str:
        """返回模块的额外表示信息，包含 TP rank 和未切分的输出维度。"""
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_out_features={self.out_features * self.world_size}"


class TensorParallelRowLinear(nn.Linear):
    """行切分张量并行线性层。

    将标准线性层的权重矩阵沿输入维度（第 1 维，即行方向）切分到多个 GPU 上。
    每个 GPU 持有权重的一个行切片，计算部分矩阵乘法后通过通信聚合结果。

    切分方式：
        完整权重 W: [out_features, in_features]
        每个 rank 持有: W_i: [out_features, in_features/TP]
        输出: y = AllReduce/ReduceScatter(sum_i(x_i @ W_i^T)) + bias

    通信模式：
        - ALL_REDUCE: 对部分结果执行 AllReduce 求和
        - REDUCE_SCATTER: 不支持此模式（会抛出异常）

    偏置处理：
        仅 rank 0 持有偏置参数，避免 AllReduce 后重复加偏置。

    Attributes:
        pg (ProcessGroup): 张量并行进程组。
        world_size (int): 张量并行度（TP 大小）。
        mode (TensorParallelLinearMode): 通信模式。
        async_communication (bool): 是否使用异步通信。
    """

    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        """初始化行切分张量并行线性层。

        Args:
            in_features (int): 输入特征维度（未切分），必须能被 TP 大小整除。
            out_features (int): 输出特征维度（未切分）。
            pg (dist.ProcessGroup): 张量并行进程组。
            mode (TensorParallelLinearMode): 通信模式。
            bias (bool, optional): 是否使用偏置。默认为 True。
                注意：仅 rank 0 会实际持有偏置参数。
            device: 设备类型。
            dtype: 数据类型。
            async_communication (bool, optional): 是否使用异步通信。默认为 False。
                仅在 REDUCE_SCATTER 模式下支持。
            contiguous_chunks (Optional[Tuple[int, ...]], optional): 连续块大小配置，
                用于将输入维度划分为多个连续块分别切分。默认为 None。

        Raises:
            AssertionError: 当 in_features 不能被 TP 大小整除时。
            ValueError: 当 ALL_REDUCE 模式下启用 async_communication 时。
        """
        self.pg = pg
        self.world_size = pg.size()

        assert in_features % self.world_size == 0

        self.in_features = in_features // self.world_size
        self.out_features = out_features

        bias = dist.get_rank(self.pg) == 0 and bias

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.mode = mode
        self.async_communication = async_communication
        if self.mode is TensorParallelLinearMode.ALL_REDUCE and self.async_communication:
            raise ValueError("async_communication is not supported for ALL_REDUCE mode")

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == in_features
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to in_features ({in_features})"

        split_config = SplitConfig(split_dim=1, contiguous_chunks=contiguous_chunks)

        self._mark_all_parameters_in_module_as_sharded(split_config)

    def _mark_all_parameters_in_module_as_sharded(self, split_config: SplitConfig):
        """将模块中的所有参数标记为分片参数。

        偏置参数仅在 rank 0 上存在，标记为普通 NanotronParameter。
        权重参数标记为分片参数，记录切分配置。

        Args:
            split_config (SplitConfig): 分片配置，指定切分维度和连续块。
        """
        for name, param in list(self.named_parameters()):
            if name == "bias":
                new_param = NanotronParameter(tensor=param)
            else:
                new_param = create_sharded_parameter_from_config(
                    parameter=param,
                    pg=self.pg,
                    split_config=split_config,
                )
            setattr(self, name, new_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行行切分线性层的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 [..., in_features/TP]。

        Returns:
            torch.Tensor: 输出张量，形状为 [..., out_features]。
        """
        return row_linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
            group=self.pg,
            tp_mode=self.mode,
            async_communication=self.async_communication,
        )

    def extra_repr(self) -> str:
        """返回模块的额外表示信息，包含 TP rank 和未切分的输入维度。"""
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_in_features={self.in_features * self.world_size}"


class TiedLinear(nn.Linear):
    """绑定权重线性层，多个 TP rank 共享相同的权重。

    与 TensorParallelColumnLinear/RowLinear 不同，TiedLinear 不切分权重，
    而是让所有 TP rank 持有完整的权重副本。这适用于需要完整权重的场景，
    例如 LayerNorm 中的仿射变换。

    通信模式：
        - ALL_REDUCE: 前向传播后执行 identity 操作（不通信），
          反向传播时梯度通过 AllReduce 同步
        - REDUCE_SCATTER: 前向传播后执行 AllGather，
          反向传播时梯度通过 ReduceScatter 同步

    Attributes:
        pg (ProcessGroup): 张量并行进程组。
        world_size (int): 张量并行度（TP 大小）。
        mode (TensorParallelLinearMode): 通信模式。
    """

    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
    ):
        """初始化绑定权重线性层。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            pg (dist.ProcessGroup): 张量并行进程组。
            mode (TensorParallelLinearMode): 通信模式。
            bias (bool, optional): 是否使用偏置。默认为 True。
            device: 设备类型。
            dtype: 数据类型。
        """
        self.pg = pg
        self.world_size = pg.size()
        self.mode = mode

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._mark_all_parameters_in_module_as_tied()

    def _mark_all_parameters_in_module_as_tied(self):
        """将模块中的所有参数标记为绑定参数。

        ALL_REDUCE 模式下不需要梯度归约（reduce_op=None），
        因为每个 rank 独立计算完整结果，梯度通过 AllReduce 自动同步。
        REDUCE_SCATTER 模式下需要对梯度求和归约（reduce_op=SUM）。
        """
        for name, param in list(self.named_parameters()):
            new_param = create_tied_parameter(
                parameter=param,
                name=name,
                global_ranks=tuple(sorted((get_global_rank(self.pg, i) for i in range(self.pg.size())))),
                reduce_op=None if self.mode is TensorParallelLinearMode.ALL_REDUCE else dist.ReduceOp.SUM,
                root_module=self,
            )
            setattr(self, name, new_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行绑定权重线性层的前向传播。

        先执行标准线性计算，然后根据通信模式进行后处理：
            - ALL_REDUCE: 执行 identity（占位操作，确保反向传播时梯度正确同步）
            - REDUCE_SCATTER: 执行 AllGather，将各 rank 的部分结果聚合

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。

        Raises:
            ValueError: 当遇到未知的通信模式时。
        """
        y = super().forward(x)
        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            y = differentiable_identity(y, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            y = differentiable_all_gather(y, group=self.pg)
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return y


class TensorParallelEmbedding(nn.Embedding):
    """张量并行嵌入层，将词表沿词元维度切分到多个 GPU。

    每个 GPU 只存储词表的一个子集对应的嵌入向量。前向传播时，
    每个 rank 只查找自己负责的词元 ID 范围内的嵌入，其他位置置零，
    然后通过 AllReduce 或 ReduceScatter 聚合所有 rank 的结果。

    切分方式：
        完整词表: [num_embeddings, embedding_dim]
        Rank i 持有: [num_embeddings/TP, embedding_dim]
        负责的词元 ID 范围: [i * block_size, (i+1) * block_size)

    Attributes:
        pg (ProcessGroup): 张量并行进程组。
        rank (int): 当前 rank 在 TP 进程组中的编号。
        world_size (int): 张量并行度（TP 大小）。
        original_num_embeddings (int): 完整词表大小。
        min_id (int): 当前 rank 负责的最小词元 ID。
        max_id (int): 当前 rank 负责的最大词元 ID（不含）。
        mode (TensorParallelLinearMode): 通信模式。
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        """初始化张量并行嵌入层。

        Args:
            num_embeddings (int): 词表大小（未切分），必须能被 TP 大小整除。
            embedding_dim (int): 嵌入维度。
            pg (dist.ProcessGroup): 张量并行进程组。
            mode (TensorParallelLinearMode): 通信模式。
            padding_idx (Optional[int]): 填充索引。
            max_norm (Optional[float]): 最大范数。
            norm_type (float): 范数类型。默认为 2.0。
            scale_grad_by_freq (bool): 是否按频率缩放梯度。默认为 False。
            sparse (bool): 是否使用稀疏梯度。默认为 False。
            _weight: 自定义权重。
            device: 设备类型。
            dtype: 数据类型。
            contiguous_chunks (Optional[Tuple[int, ...]], optional): 连续块大小配置。
                默认为 None。

        Raises:
            AssertionError: 当 num_embeddings 不能被 TP 大小整除时。
        """
        self.pg = pg
        self.rank = dist.get_rank(self.pg)
        self.world_size = pg.size()

        self.original_num_embeddings = num_embeddings

        assert num_embeddings % self.world_size == 0
        block_size = num_embeddings // self.world_size
        self.min_id = self.rank * block_size
        self.max_id = (self.rank + 1) * block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

        self.mode = mode

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == num_embeddings
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to num_embeddings ({num_embeddings})"

        split_config = SplitConfig(split_dim=0, contiguous_chunks=contiguous_chunks)

        mark_all_parameters_in_module_as_sharded(self, pg=self.pg, split_config=split_config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行张量并行嵌入查找。

        处理流程：
            1. 创建输入掩码：标记不属于当前 rank 的词元 ID
            2. 将输入 ID 映射到本地范围 [0, block_size)
            3. 对超出范围的 ID 用 0 替代（避免越界查找）
            4. 执行嵌入查找
            5. 将不属于当前 rank 的位置置零
            6. 通过 AllReduce 或 ReduceScatter 聚合所有 rank 的结果

        Args:
            input_ids (torch.Tensor): 输入词元 ID 张量，形状为 [batch, seq_len]。

        Returns:
            torch.Tensor: 嵌入向量，形状为 [batch, seq_len, embedding_dim]。

        Raises:
            ValueError: 当遇到未知的通信模式时。
        """
        if self.pg.size() > 1:
            input_mask = torch.logical_or(self.min_id > input_ids, input_ids >= self.max_id)
            masked_input = input_ids.clone() - self.min_id
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids
        out = super().forward(masked_input)

        if self.pg.size() > 1:
            out = out * (~input_mask[..., None])

        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            out = differentiable_all_reduce_sum(out, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            out = differentiable_reduce_scatter_sum(
                out, group=self.pg
            )
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return out

    def extra_repr(self) -> str:
        """返回模块的额外表示信息，包含 TP rank 和未切分的词表大小。"""
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_num_embeddings={self.original_num_embeddings}"
