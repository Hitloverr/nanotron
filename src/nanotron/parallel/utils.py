"""
并行工具模块 —— Nanotron 分布式训练的辅助工具。

本模块提供分布式训练中的通用工具类和函数，包括：
    - MemoryBuffer: 全局内存缓冲区，用于复用中间张量的显存
    - initial_sync: 初始参数同步，确保所有设备上的参数一致

MemoryBuffer 的设计思路：
    在张量并行和序列并行中，AllGather 操作需要临时缓冲区来存储聚合后的张量。
    如果每次 AllGather 都分配新的显存，会导致频繁的显存分配/释放，增加碎片化。
    MemoryBuffer 使用单例模式维护一个全局缓冲区，按 (name, dtype) 索引，
    当需要更大的缓冲区时自动扩容，避免重复分配。

    典型使用场景：
        - ColumnLinear 的 REDUCE_SCATTER 模式：AllGather 输入时需要临时缓冲区
        - RowLinear 的反向传播：AllGather 梯度时需要临时缓冲区

initial_sync 的设计思路：
    在分布式训练开始前，需要确保所有设备上的模型参数完全一致。
    这对于从随机初始化开始的训练尤为重要，因为不同设备可能使用不同的随机种子。
    initial_sync 分两步同步：
        1. 跨 DP 同步：确保同一模型副本的参数一致
        2. 跨绑定权重同步：确保绑定参数在所有相关 rank 上一致
"""

import functools
import operator

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.utils import Singleton


class MemoryBuffer(metaclass=Singleton):
    """全局内存缓冲区，用于复用中间激活值的显存。

    在张量并行和序列并行的通信操作（如 AllGather）中，需要临时缓冲区
    存储聚合后的张量。MemoryBuffer 通过单例模式维护一个全局缓冲池，
    按 (名称, 数据类型) 索引，当需要更大缓冲区时自动扩容。

    工作机制：
        - 首次请求时分配指定形状的缓冲区
        - 后续请求如果需要更大的缓冲区，自动扩容
        - 后续请求如果缓冲区足够大，直接复用（返回前 required_numel 个元素的视图）

    注意事项：
        - 由于是单例模式，整个进程只有一个 MemoryBuffer 实例
        - 缓冲区只会扩大不会缩小，避免频繁的显存分配
        - 返回的是视图（view），不是拷贝，因此使用前需要确保数据已写入
    """

    def __init__(self):
        self.buffer = {}

    def get(self, name: str, shape: tuple[int], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """获取指定形状和数据类型的缓冲区。

        如果缓冲区不存在或不够大，则分配新的缓冲区。

        Args:
            name (str): 缓冲区名称，用于区分不同的用途。
            shape (tuple[int]): 所需的张量形状。
            dtype (torch.dtype): 数据类型，默认为 bfloat16。

        Returns:
            torch.Tensor: 形状为 shape 的张量视图。
        """
        required_numel = functools.reduce(operator.mul, shape, 1)
        if (name, dtype) not in self.buffer or self.buffer[name, dtype].numel() < required_numel:
            self.buffer[name, dtype] = torch.empty(
                required_numel, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )
        return self.buffer[name, dtype][:required_numel].view(shape)


def initial_sync(model: nn.Module, parallel_context: ParallelContext):
    """在训练开始前同步所有设备上的模型参数。

    确保所有设备上的参数完全一致，这对于从随机初始化开始的训练尤为重要。
    同步分两步进行：

    步骤 1 - 跨 DP 同步：
        对所有参数在 DP 进程组内执行 AllReduce AVG 操作。
        这确保了同一模型副本在不同 DP rank 上的参数一致。
        使用 AVG 而非 SUM 是因为参数在初始化时各 rank 应该相同，
        AVG 操作相当于取任意一个 rank 的值（因为它们应该相同）。

    步骤 2 - 跨绑定权重同步：
        对所有绑定参数在其专属进程组内执行 AllReduce AVG 操作。
        这确保了绑定参数（如 Embedding 和 LM Head 共享的权重）
        在所有相关 rank 上一致。

    Args:
        model (nn.Module): 待同步的模型。
        parallel_context (ParallelContext): 并行上下文，提供进程组信息。

    Note:
        - 必须在训练循环开始前调用
        - 参数按名称排序后同步，确保所有 rank 上的处理顺序一致
        - 使用 AVG 操作而非 SUM，因为初始化时各 rank 的参数应该相同
    """
    # 步骤 1：跨 DP 同步所有参数
    sorted_name_params = sorted(model.named_parameters(), key=lambda x: x[0])
    for name, param in sorted_name_params:
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=parallel_context.dp_pg)

    # 步骤 2：跨绑定权重同步
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model.parameters(), root_module=model).items(), key=lambda x: x[0]
    ):
        group = parallel_context.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
