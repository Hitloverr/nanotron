"""
分布式通信封装模块 —— Nanotron 的底层通信基础设施。

本模块对 PyTorch 的分布式通信操作进行封装和增强，提供：
    - 版本兼容性：支持 PyTorch 1.12+ 的 API 差异
    - 批量通信操作：reduce_scatter_coalesced、all_gather_coalesced、all_reduce_coalesced
    - 进程组管理：new_group、get_global_rank、get_rank
    - 初始化：initialize_torch_distributed

与并行策略的关系：
    - DP（数据并行）：使用 all_reduce 同步梯度，reduce_scatter/all_gather 用于 ZeRO
    - PP（流水线并行）：使用 isend/irecv 点对点通信传输激活值和梯度
    - TP（张量并行）：使用 all_reduce 或 reduce_scatter/all_gather 聚合部分结果
    - EP（专家并行）：使用 all_to_all 路由 token 到对应专家
    - CP（上下文并行）：使用环形点对点通信传递 K/V

批量通信操作的设计思路：
    当需要同时通信多个张量时，逐个通信效率较低。批量操作将多个张量
    拼接（flatten）为一个连续缓冲区，执行一次通信操作，然后拆分回
    原始张量。这减少了通信启动次数，提高带宽利用率。

    flatten/unflatten 技巧：
        - torch._utils._flatten_dense_tensors: 将多个张量拼接为一个连续缓冲区
        - torch._utils._unflatten_dense_tensors: 将缓冲区拆分回原始张量
        - 使用 copy_() 将结果写回原始张量，保持引用不变

注意事项：
    - 通信超时默认 20 分钟，调试通信死锁时可减小此值
    - 单 rank 进程组的通信操作会跳过或报错（无意义）
    - get_rank 在当前进程不在进程组中时抛出异常（而非返回 -1）
"""

import datetime
import os
from functools import cache, lru_cache
from typing import List, Optional, Tuple

import torch
from packaging import version
from torch import distributed as dist
from torch.distributed import *  # noqa
from torch.distributed.distributed_c10d import ProcessGroup

from nanotron.utils import find_free_port

torch_version_above_1_13 = version.parse(torch.__version__) >= version.parse("1.13.0")
Work = dist.Work if torch_version_above_1_13 else dist._Work

# Note: When debugging communication hangs, try decreasing this timeout.
default_pg_timeout = datetime.timedelta(minutes=20)


def new_group(  # pylint: disable=function-redefined
    ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None
) -> ProcessGroup:
    """创建新的进程组。

    封装 torch.distributed.new_group，增加空 ranks 检查。

    Args:
        ranks: 进程组包含的全局 rank 列表。
        timeout: 通信超时时间，默认 20 分钟。
        backend: 通信后端（如 "nccl"），默认继承父组。
        pg_options: 进程组选项。

    Returns:
        ProcessGroup: 新创建的进程组。

    Raises:
        ValueError: 当 ranks 为空时。
    """
    if len(ranks) == 0:
        raise ValueError("Cannot create a group with not ranks inside it")

    return dist.new_group(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)


def reduce_scatter_tensor(  # pylint: disable=function-redefined
    output: torch.Tensor,
    input: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    """ReduceScatter 单张量操作。

    对输入张量执行归约操作（如求和），然后将结果沿第 0 维分散到各 rank。
    每个 rank 获得结果的一部分。

    在张量并行中的用途：
        - REDUCE_SCATTER 模式下，RowLinear 的输出通过 ReduceScatter 分散
        - 序列并行中，将聚合后的序列分散回各 rank

    Args:
        output: 输出张量，形状为 input 沿第 0 维切分后的大小。
        input: 输入张量。
        op: 归约操作，默认 SUM。
        group: 进程组。
        async_op: 是否异步执行。

    Returns:
        Optional[Work]: 异步操作句柄（如果 async_op=True）。
    """
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `reduce_scatter_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.reduce_scatter_tensor(output=output, input=input, group=group, op=op, async_op=async_op)
    else:
        # Support pytorch 1.12
        return dist._reduce_scatter_base(output=output, input=input, group=group, op=op, async_op=async_op)


def all_gather_into_tensor(  # pylint: disable=function-redefined
    output_tensor, input_tensor, group: Optional[ProcessGroup] = None, async_op: bool = False
) -> Optional[Work]:
    """AllGather 单张量操作。

    将各 rank 的输入张量沿第 0 维拼接，结果存入输出张量。

    在张量并行中的用途：
        - REDUCE_SCATTER 模式下，ColumnLinear 的输入通过 AllGather 聚合
        - 序列并行中，将分散的序列聚合为完整序列

    Args:
        output_tensor: 输出张量，形状为 input_tensor 沿第 0 维扩展 group.size() 倍。
        input_tensor: 输入张量。
        group: 进程组。
        async_op: 是否异步执行。

    Returns:
        Optional[Work]: 异步操作句柄（如果 async_op=True）。
    """
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `all_gather_into_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.all_gather_into_tensor(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )
    else:
        # Support Pytorch 1.12
        return dist.distributed_c10d._all_gather_base(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )


def reduce_scatter_coalesced(
    output_tensor_list: List[torch.Tensor],
    input_tensor_lists: List[List[torch.Tensor]],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """批量 ReduceScatter 操作。

    对多个张量同时执行 ReduceScatter，将它们拼接为一个连续缓冲区
    后执行一次通信，然后拆分回原始张量。比逐个通信更高效。

    在 ZeRO 优化器中的用途：
        对多个参数的梯度同时执行 ReduceScatter，减少通信次数。

    Args:
        output_tensor_list: 输出张量列表，每个张量对应一个参数的分片。
        input_tensor_lists: 输入张量列表的列表，[param_idx][group_rank]。
        op: 归约操作，默认 SUM。
        group: 进程组。
        async_op: 是否异步执行。

    Returns:
        Optional[torch._C.Future]: 异步操作的未来对象（如果 async_op=True）。
    """
    assert len(output_tensor_list) > 0
    assert len(input_tensor_lists) == len(output_tensor_list)
    device = output_tensor_list[0].device
    dtype = output_tensor_list[0].dtype
    group_size = len(input_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `reduce_scatter_coalesced` with a single rank, as it copies data over"

    for output_tensor in output_tensor_list:
        assert device == output_tensor.device
        assert dtype == output_tensor.dtype

    for input_tensor_list in input_tensor_lists:
        assert len(input_tensor_list) == group_size, f"Expected {len(input_tensor_list)} == {group_size}"
        for input_tensor in input_tensor_list:
            assert device == input_tensor.device
            assert dtype == input_tensor.dtype

    # 将多个输出张量拼接为一个连续缓冲区
    output_tensor_buffer = torch._utils._flatten_dense_tensors(output_tensor_list)
    # 将每个 rank 的多个输入张量拼接为一个连续缓冲区
    input_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(
            [input_tensor_list[group_rank] for input_tensor_list in input_tensor_lists]
        )
        for group_rank in range(group_size)
    ]

    work = dist.reduce_scatter(output_tensor_buffer, input_tensor_buffer_list, op=op, group=group, async_op=async_op)

    def update_output():
        # 将缓冲区拆分回原始张量，并复制到原始位置
        for original_buffer, reduced_buffer in zip(
            output_tensor_list, torch._utils._unflatten_dense_tensors(output_tensor_buffer, output_tensor_list)
        ):
            original_buffer.copy_(reduced_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


def all_reduce_coalesced(  # pylint: disable=function-redefined
    tensors: List[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """批量 AllReduce 操作。

    对多个张量同时执行 AllReduce，减少通信启动次数。

    在绑定权重梯度同步中的用途：
        对同一绑定组的多个参数梯度执行一次批量 AllReduce。

    Args:
        tensors: 待归约的张量列表。
        op: 归约操作，默认 SUM。
        group: 进程组。
        async_op: 是否异步执行。

    Returns:
        Optional[torch._C.Future]: 异步操作的未来对象。
    """
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    if group.size() == 1:
        return

    return dist.all_reduce_coalesced(tensors, op=op, group=group, async_op=async_op)


def all_gather_coalesced(  # pylint: disable=function-redefined
    output_tensor_lists: List[List[torch.Tensor]],
    input_tensor_list: List[torch.Tensor],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """批量 AllGather 操作。

    对多个张量同时执行 AllGather，将它们拼接为一个连续缓冲区
    后执行一次通信，然后拆分回原始张量。

    torch 原生的 all_gather_coalesced 在 NCCL 上不可用，
    因此我们手动实现，使用 flatten/unflatten 技巧。

    Args:
        output_tensor_lists: 输出张量列表的列表，[param_idx][group_rank]。
        input_tensor_list: 输入张量列表。
        group: 进程组。
        async_op: 是否异步执行。

    Returns:
        Optional[torch._C.Future]: 异步操作的未来对象（如果 async_op=True）。
    """
    assert len(output_tensor_lists) > 0
    assert len(input_tensor_list) == len(output_tensor_lists)
    device = input_tensor_list[0].device
    dtype = input_tensor_list[0].dtype
    group_size = len(output_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `all_gather_coalesced` with a single rank, as it copies data over"

    for input_tensor in input_tensor_list:
        assert device == input_tensor.device
        assert dtype == input_tensor.dtype

    for output_tensor_list in output_tensor_lists:
        assert len(output_tensor_list) == group_size
        for output_tensor in output_tensor_list:
            assert device == output_tensor.device
            assert dtype == output_tensor.dtype

    # 将 [param_idx][group_rank] 转置为 [group_rank][param_idx]
    output_tensor_lists = [
        [output_tensor_list[group_rank] for output_tensor_list in output_tensor_lists]
        for group_rank in range(group_size)
    ]

    # 将多个输入/输出张量拼接为连续缓冲区
    input_tensor_buffer = torch._utils._flatten_dense_tensors(input_tensor_list)
    output_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(output_tensor_list) for output_tensor_list in output_tensor_lists
    ]

    work = dist.all_gather(output_tensor_buffer_list, input_tensor_buffer, group=group, async_op=async_op)

    def update_output():
        # 将缓冲区拆分回原始张量，并复制到原始位置
        for original_buffer_list, gathered_buffer_tensor in zip(output_tensor_lists, output_tensor_buffer_list):
            for original_buffer, gathered_buffer in zip(
                original_buffer_list,
                torch._utils._unflatten_dense_tensors(gathered_buffer_tensor, original_buffer_list),
            ):
                original_buffer.copy_(gathered_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


# This cache has a speedup of 4 tflops on a 7b model
@cache
def get_global_rank(group: ProcessGroup, group_rank: int) -> int:  # pylint: disable=function-redefined
    """将进程组内的 rank 编号转换为全局 rank 编号。

    带缓存，避免重复查询。在 7B 模型上有约 4 TFLOPS 的加速。

    Args:
        group: 进程组。
        group_rank: 进程组内的 rank 编号。

    Returns:
        int: 全局 rank 编号。
    """
    if torch_version_above_1_13:
        return dist.get_global_rank(group, group_rank=group_rank)
    else:
        # Support pytorch 1.12
        return dist.distributed_c10d._get_global_rank(group=group, rank=group_rank)


def get_global_ranks(group: ProcessGroup) -> Tuple[int]:
    """获取进程组中所有 rank 的全局编号（排序后）。

    Args:
        group: 进程组。

    Returns:
        Tuple[int]: 排序后的全局 rank 编号元组。
    """
    return tuple(sorted((get_global_rank(group, i) for i in range(group.size()))))


# We cache for dp, pp, tp process groups, world group, and tied process group for tied params
@lru_cache
def get_rank(group: Optional[ProcessGroup] = None) -> int:  # pylint: disable=function-redefined
    """获取当前进程在指定进程组中的 rank 编号。

    与 torch.distributed.get_rank 的区别：
        当当前进程不在进程组中时，抛出 RuntimeError 而非返回 -1。
        这有助于及早发现配置错误。

    Args:
        group: 进程组，None 表示默认进程组。

    Returns:
        int: 当前进程的 rank 编号。

    Raises:
        RuntimeError: 当当前进程不在进程组中时。
    """
    result = dist.get_rank(group)
    if result == -1:
        raise RuntimeError("Can not call `get_rank` on a group in which current process is not a part of")
    return result


def initialize_torch_distributed():
    """初始化 PyTorch 分布式环境。

    从环境变量读取分布式配置，初始化进程组。
    支持的环境变量：
        - RANK: 全局 rank 编号
        - WORLD_SIZE: 总进程数
        - LOCAL_RANK: 节点内 rank 编号
        - MASTER_PORT: 主节点端口（可选，自动检测空闲端口）

    初始化流程：
        1. 读取环境变量获取 rank 和 world_size
        2. 设置 CUDA 设备
        3. 选择通信后端（NCCL）
        4. 初始化进程组

    Returns:
        bool: 初始化成功返回 True。
    """
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        # Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))
        backend = "nccl"
    else:
        # TODO @thomasw21: Maybe figure out a way to do distributed `cpu` training at some point
        raise NotImplementedError(f"CUDA was not found: torch.cuda.is_available(): {torch.cuda.is_available()}")
        backend = "gloo"

    # Call the init process.

    port = os.getenv("MASTER_PORT")
    if port is None:
        port = find_free_port()
    else:
        port = int(port)

    init_method = f"env://localhost:{port}"
    dist.init_process_group(
        init_method=init_method, backend=backend, world_size=world_size, rank=rank, timeout=dist.default_pg_timeout
    )
    return True
