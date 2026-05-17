"""
分片参数管理模块 —— Nanotron 分布式训练中的参数切分与元数据管理。

本模块负责将模型参数按指定维度切分到多个设备上，并为每个切分后的参数
生成分片元数据（ShardedInfo），以支持序列化/反序列化和跨设备通信。

核心概念：
    - 分片参数（Sharded Parameter）：将一个完整参数沿某个维度切分到多个 GPU 上，
      每个 GPU 只持有部分数据。典型场景：张量并行（TP）中的权重矩阵切分、
      专家并行（EP）中的专家切分、ZeRO 优化器中的优化器状态切分。

    - SplitConfig：描述参数的切分配置，包括切分维度和连续块大小。
      - split_dim: 沿哪个维度切分（0=行/输出维度, 1=列/输入维度）
      - contiguous_chunks: 将切分维度划分为多个连续块，每个块独立切分。
        这对于某些参数（如 QKV 投影的合并权重）很有用，因为它们在逻辑上
        由多个不同语义的部分组成，需要分别切分。

    - SlicesPair：描述本地切片与全局切片的映射关系，用于序列化时重建完整参数。

分片参数的创建流程：
    1. 定义 SplitConfig，指定切分维度和连续块
    2. 调用 create_sharded_parameter_from_config() 生成分片元数据
    3. 元数据附加到 NanotronParameter 上，供后续序列化/通信使用

与张量并行（TP）的关系：
    - TensorParallelColumnLinear: 权重沿第 0 维（输出维度）切分
    - TensorParallelRowLinear: 权重沿第 1 维（输入维度）切分
    - TensorParallelEmbedding: 词表沿第 0 维（词元维度）切分

与专家并行（EP）的关系：
    - MoE 模型中的专家权重沿专家维度切分，每个设备持有部分专家

潜在优化点：
    - 当前 contiguous_chunks 要求每个块大小能被进程组大小整除，
      可以考虑支持不均匀切分以适应更多场景
    - 可以增加对多维同时切分的支持
"""

import dataclasses
from typing import List, Optional, Tuple

import numpy as np
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel.parameters import NanotronParameter, SlicesPair


@dataclasses.dataclass
class SplitConfig:
    """参数切分配置，描述如何将参数沿指定维度切分到多个设备。

    Attributes:
        split_dim (int): 切分维度索引。
            - 0: 沿第 0 维（行/输出维度）切分，用于 ColumnLinear 和 Embedding
            - 1: 沿第 1 维（列/输入维度）切分，用于 RowLinear
        contiguous_chunks (Optional[Tuple[int, ...]]): 切分维度上的连续块大小。
            如果为 None，则默认整个切分维度是一个连续块。
            如果指定，则将切分维度划分为多个连续块，每个块独立切分。

    使用场景示例：
        1. 标准线性层权重 [out, in]，沿第 0 维切分：
           SplitConfig(split_dim=0)  # contiguous_chunks=None

        2. QKV 合并投影权重 [3*hidden, hidden]，沿第 0 维切分，
           需要确保 Q/K/V 各自独立切分：
           SplitConfig(split_dim=0, contiguous_chunks=(hidden, hidden, hidden))

        3. FFN 中 gate_up 合并投影权重 [2*intermediate, hidden]：
           SplitConfig(split_dim=0, contiguous_chunks=(intermediate, intermediate))
    """

    split_dim: int
    contiguous_chunks: Optional[Tuple[int, ...]] = None


def create_sharded_parameter(
    parameter: nn.Parameter,
    global_ranks: Tuple[int, ...],
    local_global_slices_pairs: Tuple[SlicesPair, ...],
    unsharded_shape: Tuple[int, ...],
) -> NanotronParameter:
    """将普通参数转换为分片参数，附加分片元数据。

    这是创建分片参数的低级接口，需要手动提供所有分片信息。
    通常应使用 create_sharded_parameter_from_config() 代替。

    Args:
        parameter (nn.Parameter): 待分片的参数。如果不是 NanotronParameter，
            会自动转换。
        global_ranks (Tuple[int, ...]): 持有该参数分片的所有全局 rank 编号。
        local_global_slices_pairs (Tuple[SlicesPair, ...]): 本地切片与全局切片
            的映射对，描述本地数据在完整参数中的位置。
        unsharded_shape (Tuple[int, ...]): 完整（未分片）参数的形状。

    Returns:
        NanotronParameter: 带有分片元数据的参数。
    """
    if not isinstance(parameter, NanotronParameter):
        parameter = NanotronParameter(tensor=parameter)
    parameter.mark_as_sharded(
        global_ranks=global_ranks,
        local_global_slices_pairs=local_global_slices_pairs,
        unsharded_shape=unsharded_shape,
    )
    return parameter


def create_sharded_parameter_from_config(
    parameter: nn.Parameter,
    pg: dist.ProcessGroup,
    split_config: SplitConfig,
) -> NanotronParameter:
    """根据切分配置自动计算分片元数据并创建分片参数。

    该方法根据当前 rank 在进程组中的位置，自动计算本地切片在完整参数中
    的位置，生成 SlicesPair 映射关系。

    算法流程：
        1. 获取当前 rank 在进程组中的编号和进程组大小
        2. 根据 contiguous_chunks 是否为 None，选择不同的切分策略：
           a. 无连续块（默认）：整个切分维度均匀切分
           b. 有连续块：每个块独立均匀切分
        3. 计算本地切片在全局参数中的偏移量
        4. 构建 SlicesPair 映射和 unsharded_shape
        5. 调用 create_sharded_parameter 创建分片参数

    Args:
        parameter (nn.Parameter): 待分片的参数。其形状已经是切分后的本地形状。
        pg (dist.ProcessGroup): 进程组，通常是 TP 或 EP 进程组。
        split_config (SplitConfig): 切分配置。

    Returns:
        NanotronParameter: 带有分片元数据的参数。

    Raises:
        AssertionError: 当 split_dim 超出参数维度范围时。
        AssertionError: 当连续块大小不能被进程组大小整除时。

    示例（TP=2，权重形状 [2048, 4096]，沿第 0 维切分）：
        - Rank 0: parameter.shape = [1024, 4096]
          global_ranks = (0, 1)
          local_global_slices_pairs = (SlicesPair(local=(slice(None),slice(None)),
                                                   global=(slice(0,1024),slice(None))),)
          unsharded_shape = (2048, 4096)
        - Rank 1: parameter.shape = [1024, 4096]
          global_ranks = (0, 1)
          local_global_slices_pairs = (SlicesPair(local=(slice(None),slice(None)),
                                                   global=(slice(1024,2048),slice(None))),)
          unsharded_shape = (2048, 4096)
    """
    current_rank = dist.get_rank(pg)
    param_num_dims = len(parameter.shape)
    global_ranks = dist.get_global_ranks(pg)
    split_dim = split_config.split_dim
    assert split_dim < param_num_dims
    contiguous_chunks = split_config.contiguous_chunks

    if contiguous_chunks is None:
        # 简单情况：整个切分维度是一个连续块，均匀切分
        # 每个 rank 持有切分维度上的 shard_length 个元素
        shard_length = parameter.shape[split_dim]
        # 计算本地数据在完整参数中的起始位置
        # 例如：rank 0 → [0, shard_length), rank 1 → [shard_length, 2*shard_length)
        global_slice = slice(current_rank * shard_length, (current_rank + 1) * shard_length)
        # 本地切片：所有维度都是 slice(None)，因为本地数据是连续的
        local_slices = tuple(slice(None) for _ in range(param_num_dims))
        # 全局切片：只有切分维度有偏移，其他维度不变
        global_slices = tuple(global_slice if dim_id == split_dim else slice(None) for dim_id in range(param_num_dims))
        local_global_slices_pairs = (SlicesPair(local_slices=local_slices, global_slices=global_slices),)
        # 完整参数形状：切分维度的大小 = 本地大小 × 进程组大小
        unsharded_shape = tuple(
            pg.size() * param_dim_size if dim_id == split_dim else param_dim_size
            for dim_id, param_dim_size in enumerate(parameter.shape)
        )
    else:
        # 复杂情况：切分维度由多个连续块组成，每个块独立切分
        # 这用于 QKV 合并投影等场景，确保 Q/K/V 各自独立切分
        local_global_slices_pairs: List[SlicesPair] = []
        # 计算每个连续块在全局参数中的累积偏移量
        chunks_global_offset = np.cumsum((0,) + contiguous_chunks)
        # 计算每个连续块在本地参数中的累积偏移量（每个块被均匀切分）
        chunks_local_offset = chunks_global_offset // pg.size()
        for chunk, chunk_global_start, chunk_local_start, chunk_local_end in zip(
            contiguous_chunks,
            chunks_global_offset[:-1],
            chunks_local_offset[:-1],
            chunks_local_offset[1:],
            strict=True,
        ):
            # 每个连续块的大小必须能被进程组大小整除，确保均匀切分
            assert chunk % pg.size() == 0, f"chunk size {chunk} must be divisible by process group size {pg.size()}"
            # 当前 rank 在该块中持有的元素数量
            shard_length = chunk // pg.size()
            # 本地切片：该块在本地参数中的范围
            local_slice = slice(chunk_local_start, chunk_local_end)
            # 全局切片：该块在完整参数中的范围（加上当前 rank 的偏移）
            global_slice = slice(
                current_rank * shard_length + chunk_global_start,
                (current_rank + 1) * shard_length + chunk_global_start,
            )
            local_slices = tuple(
                local_slice if dim_id == split_dim else slice(None) for dim_id in range(param_num_dims)
            )
            global_slices = tuple(
                global_slice if dim_id == split_dim else slice(None) for dim_id in range(param_num_dims)
            )
            local_global_slices_pairs.append(SlicesPair(local_slices=local_slices, global_slices=global_slices))
        local_global_slices_pairs: Tuple[SlicesPair, ...] = tuple(local_global_slices_pairs)
        # 完整参数形状：切分维度的大小 = 所有连续块大小之和
        unsharded_shape = tuple(
            chunks_global_offset[-1] if dim_id == split_dim else param_dim_size
            for dim_id, param_dim_size in enumerate(parameter.shape)
        )

    return create_sharded_parameter(
        parameter=parameter,
        global_ranks=global_ranks,
        local_global_slices_pairs=local_global_slices_pairs,
        unsharded_shape=unsharded_shape,
    )

"""
# 创建分片参数的流程
SplitConfig(split_dim=0, contiguous_chunks=None)
    │
    ▼ create_sharded_parameter_from_config()
    │
    ▼ mark_all_parameters_in_module_as_sharded()
    │
    ▼ NanotronParameter.mark_as_sharded()

**示例**：`TensorParallelColumnLinear` 的权重分片

原始权重: [4096, 4096]  (out_features × in_features)
TP=2 时：
  Rank 0: weight[0:2048, :]   (local_slice)
  Rank 1: weight[2048:4096, :] (local_slice)

ShardedInfo:
  global_ranks = (0, 1)
  local_global_slices_pairs = (
    SlicesPair(
      local_slices = (slice(0, 2048), slice(None)),
      global_slices = (slice(0, 2048), slice(None))   # Rank 0
    ),
  )
  unsharded_shape = (4096, 4096)
"""
def mark_all_parameters_in_module_as_sharded(module: nn.Module, pg: dist.ProcessGroup, split_config: SplitConfig):
    """将模块中所有参数标记为分片参数。

    遍历模块及其所有子模块，将每个参数都转换为带有分片元数据的
    NanotronParameter。假设所有参数都可以沿同一维度均匀切分。

    该方法通常在构建张量并行层时调用，例如：
        - TensorParallelColumnLinear: 所有参数沿第 0 维切分
        - TensorParallelRowLinear: 权重沿第 1 维切分，偏置不切分
        - TensorParallelEmbedding: 词表沿第 0 维切分

    Args:
        module (nn.Module): 待标记的模块。
        pg (dist.ProcessGroup): 进程组，通常是 TP 进程组。
        split_config (SplitConfig): 切分配置，指定切分维度和连续块。

    Note:
        该方法会原地修改模块中的参数，将原始 nn.Parameter 替换为
        NanotronParameter。使用 list() 避免在迭代时修改字典。
    """

    for module_name, submodule in module.named_modules():
        for param_name, param in list(submodule.named_parameters(recurse=False)):
            new_param = create_sharded_parameter_from_config(parameter=param, pg=pg, split_config=split_config)
            setattr(submodule, param_name, new_param)
