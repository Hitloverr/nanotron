"""
点对点通信模块 —— Nanotron 流水线并行的通信基础设施。

本模块实现了流水线并行中相邻 stage 之间的点对点（P2P）通信，
用于传输激活值和梯度。是流水线并行的核心通信层。

核心组件：
    - P2PTensorMetaData: 张量元数据，描述张量的形状、步幅、数据类型等信息
    - P2P: 点对点通信类，提供同步/异步的 send/recv 操作
    - BatchTensorSendRecvState: 批量通信状态管理器，支持通信重叠优化

通信协议设计：
    P2P 通信采用三阶段协议传输一个张量：
        1. 第一阶段元数据（7 个 int64）：张量维度数、步幅数、是否连续、
           存储大小、存储偏移、数据类型 ID、是否需要梯度
        2. 第二阶段元数据（变长 int64）：张量的形状和步幅值
        3. 实际数据：张量的原始数据

    三阶段协议的必要性：
        - 接收方需要先知道张量的形状和数据类型，才能分配接收缓冲区
        - 分两步发送元数据是因为第二阶段的大小取决于第一阶段的信息
        - 这种设计避免了预分配固定大小的缓冲区

与流水线并行的关系：
    - PipelineBlock 使用 P2P 在相邻 PP rank 之间传输激活值和梯度
    - 1F1B 调度中的通信与计算重叠依赖 BatchTensorSendRecvState
    - P2P 通信是流水线气泡的主要来源之一

性能考量：
    - 使用 batch_isend_irecv 批量提交通信操作，减少通信启动开销
    - 非连续张量通过 view_as_contiguous 转换为连续存储后发送
    - 复数类型张量需要特殊处理（view_as_real/view_as_complex）
"""

import dataclasses
from typing import List, Sequence, Tuple

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.utils import get_untyped_storage, tensor_from_untyped_storage

logger = logging.get_logger(__name__)

# 第一阶段元数据大小：7 个 int64 值
# [维度数, 步幅数, 是否连续, 存储大小, 存储偏移, 数据类型ID, 是否需要梯度]
FIRST_METADATA_SIZE = 7
# 第二阶段元数据缓冲区大小：预分配 1024 个 int64，足够容纳大多数张量的形状和步幅
SECOND_METADATA_SIZE = 1024

# 数据类型到 ID 的映射，用于元数据的序列化
ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

ID_TO_REQUIRES_GRAD = [True, False]
REQUIRES_GRAD_TO_ID = {value: id_ for id_, value in enumerate(ID_TO_REQUIRES_GRAD)}
ID_TO_IS_CONTIGUOUS = [True, False]
IS_CONTIGUOUS_TO_ID = {value: id_ for id_, value in enumerate(ID_TO_IS_CONTIGUOUS)}


@dataclasses.dataclass
class P2PTensorMetaData:
    """P2P 通信中张量的元数据。

    在发送张量数据之前，先发送元数据让接收方知道如何分配缓冲区。
    元数据包含张量的完整描述信息，足以在接收方重建张量的视图。

    Attributes:
        shape (Sequence[int]): 张量形状。
        stride (Sequence[int]): 张量步幅。
        is_contiguous (bool): 张量是否在内存中连续存储。
        untyped_storage_size (int): 底层未类型化存储的大小（字节数）。
        storage_offset (int): 张量在存储中的偏移量。
        dtype (torch.dtype): 张量数据类型。
        requires_grad (bool): 张量是否需要梯度。
    """

    shape: Sequence[int]
    stride: Sequence[int]
    is_contiguous: bool
    untyped_storage_size: int
    storage_offset: int
    dtype: torch.dtype
    requires_grad: bool

    def create_empty_storage(self, device: torch.device) -> torch.Tensor:
        """根据元数据创建空缓冲区，用于接收张量数据。

        分配与原始张量相同大小的存储空间，并根据元数据设置视图。

        Args:
            device (torch.device): 目标设备。

        Returns:
            torch.Tensor: 空缓冲区，形状和步幅与原始张量一致。
        """
        buffer = torch.empty(
            size=(self.untyped_storage_size,),
            requires_grad=False,
            dtype=torch.int8,
            device=device,
            memory_format=torch.contiguous_format,
        ).view(dtype=self.dtype)
        buffer.requires_grad = self.requires_grad

        if self.is_contiguous:
            buffer = buffer.as_strided(
                size=tuple(self.shape), stride=tuple(self.stride), storage_offset=self.storage_offset
            )

        # 复数类型需要先视为实数类型，因为 NCCL 不直接支持复数传输
        buffer = torch.view_as_real(buffer) if self.dtype.is_complex else buffer

        return buffer

    def reshape(self, buffer):
        """将接收到的缓冲区重塑为与原始张量相同的视图。

        Args:
            buffer: 接收到的原始缓冲区。

        Returns:
            torch.Tensor: 重塑后的张量。
        """
        buffer = torch.view_as_complex(buffer) if self.dtype.is_complex else buffer

        if not self.is_contiguous:
            buffer = buffer.as_strided(
                size=tuple(self.shape), stride=tuple(self.stride), storage_offset=self.storage_offset
            )

        return buffer

    @staticmethod
    def to_first_metadata(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """将张量的第一阶段元数据编码为 int64 张量。

        第一阶段元数据包含 7 个值：
        [维度数, 步幅数, 是否连续ID, 存储大小, 存储偏移, 数据类型ID, 是否需要梯度ID]

        Args:
            tensor (torch.Tensor): 源张量。
            device (torch.device): 目标设备。

        Returns:
            torch.Tensor: 形状为 [7] 的 int64 张量。
        """
        return torch.tensor(
            [
                len(tensor.shape),
                len(tensor.stride()),
                IS_CONTIGUOUS_TO_ID[tensor.is_contiguous()],
                get_untyped_storage(tensor).size(),
                tensor.storage_offset(),
                DTYPE_TO_ID[tensor.dtype],
                REQUIRES_GRAD_TO_ID[tensor.requires_grad],
            ],
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def to_second_metadata(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """将张量的第二阶段元数据（形状+步幅）编码为 int64 张量。

        Args:
            tensor (torch.Tensor): 源张量。
            device (torch.device): 目标设备。

        Returns:
            torch.Tensor: 形状为 [len(shape)+len(stride)] 的 int64 张量。
        """
        return torch.tensor(tensor.shape + tensor.stride(), dtype=torch.long, device=device)

    @classmethod
    def from_metadata(cls, first_metadata: List[int], second_metadata: List[int]):
        """从两阶段元数据解码重建 P2PTensorMetaData 对象。

        Args:
            first_metadata (List[int]): 第一阶段元数据列表。
            second_metadata (List[int]): 第二阶段元数据列表（形状+步幅拼接）。

        Returns:
            P2PTensorMetaData: 重建的元数据对象。
        """
        shape_and_stride = second_metadata
        (
            num_shape,
            num_stride,
            is_contiguous,
            untyped_storage_size,
            storage_offset,
            dtype_id,
            requires_grad_id,
        ) = first_metadata
        return cls(
            shape=shape_and_stride[: len(shape_and_stride) // 2],
            stride=shape_and_stride[len(shape_and_stride) // 2 :],
            is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
            untyped_storage_size=untyped_storage_size,
            storage_offset=storage_offset,
            dtype=ID_TO_DTYPE[dtype_id],
            requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
        )


def view_as_contiguous(tensor: torch.Tensor):
    """将张量转换为连续存储视图，用于 P2P 传输。

    非连续张量（如转置后的张量）不能直接通过 NCCL 传输，
    需要先转换为连续存储。该函数通过底层存储重建连续视图。

    Args:
        tensor (torch.Tensor): 输入张量。

    Returns:
        torch.Tensor: 连续存储的张量视图。

    Raises:
        AssertionError: 当存储大小超过张量逻辑大小时（通常不应发生）。
    """
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    untyped_storage = get_untyped_storage(tensor)
    untyped_storage_size = untyped_storage.size()
    untyped_element_size = untyped_storage.element_size()
    assert (
        tensor_numel * tensor_element_size >= untyped_storage_size * untyped_element_size
    ), "Expect storage_size to be smaller than tensor size. It might not be true, when you use slicing for example though. We probably don't want to support it in our P2P system"
    buffer = tensor_from_untyped_storage(untyped_storage=untyped_storage, dtype=tensor.dtype)
    return buffer


class P2P:
    """点对点通信类，提供流水线并行中的张量传输功能。

    P2P 封装了 torch.distributed 的 send/recv 操作，支持：
        - 同步发送/接收（send_tensors/recv_tensors）
        - 异步发送/接收（isend_tensors/irecv_tensors）
        - 元数据+数据的三阶段通信协议

    通信流程（发送一个张量）：
        1. 发送第一阶段元数据（张量维度、类型等基本信息）
        2. 发送第二阶段元数据（张量形状和步幅）
        3. 发送实际张量数据

    Attributes:
        pg (dist.ProcessGroup): 流水线并行的进程组。
        device (torch.device): 通信设备。
        first_metadata (torch.Tensor): 第一阶段元数据缓冲区。
        second_metadata (torch.Tensor): 第二阶段元数据缓冲区。
    """

    def __init__(self, pg: dist.ProcessGroup, device: torch.device):
        self.pg = pg
        self.device = device
        self.first_metadata = torch.empty(FIRST_METADATA_SIZE, dtype=torch.long, device=self.device)
        self.second_metadata = torch.empty(SECOND_METADATA_SIZE, dtype=torch.long, device=self.device)

    def _send_first_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        """创建发送第一阶段元数据的异步操作。"""
        first_metadata = P2PTensorMetaData.to_first_metadata(tensor=tensor, device=self.device)
        return dist.P2POp(
            op=dist.isend,
            tensor=first_metadata,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_first_metadata_p2p_op(self, from_rank: int, tag: int = 0) -> Tuple[torch.Tensor, dist.P2POp]:
        """创建接收第一阶段元数据的异步操作。"""
        first_metadata_buffer = torch.empty((FIRST_METADATA_SIZE,), dtype=torch.long, device=self.device)
        return first_metadata_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=first_metadata_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_second_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        """创建发送第二阶段元数据的异步操作。"""
        second_metadata = P2PTensorMetaData.to_second_metadata(tensor=tensor, device=self.device)
        return dist.P2POp(
            op=dist.isend,
            tensor=second_metadata,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_second_metadata_p2p_op(
        self, shape_length: int, stride_length: int, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        """创建接收第二阶段元数据的异步操作。"""
        second_metadata_buffer = torch.empty((shape_length + stride_length,), dtype=torch.long, device=self.device)
        return second_metadata_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=second_metadata_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_data_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        """创建发送张量数据的异步操作。"""
        return dist.P2POp(
            op=dist.isend,
            tensor=tensor,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_data_p2p_op(
        self, tensor_metadata: P2PTensorMetaData, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        """创建接收张量数据的异步操作。"""
        tensor_buffer = tensor_metadata.create_empty_storage(self.device)
        return tensor_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=tensor_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_meta(self, tensor: torch.Tensor, to_rank: int, tag: int):
        """同步发送张量的两阶段元数据。

        Args:
            tensor (torch.Tensor): 待发送的张量。
            to_rank (int): 目标 rank（进程组内的编号）。
            tag (int): 通信标签，用于区分不同的通信流。
        """
        cpu_tensor = torch.tensor(
            [
                len(tensor.shape),
                len(tensor.stride()),
                IS_CONTIGUOUS_TO_ID[tensor.is_contiguous()],
                get_untyped_storage(tensor).size(),
                tensor.storage_offset(),
                DTYPE_TO_ID[tensor.dtype],
                REQUIRES_GRAD_TO_ID[tensor.requires_grad],
            ],
            dtype=torch.long,
        )
        self.first_metadata.copy_(cpu_tensor)
        dist.send(
            self.first_metadata,
            dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

        second_metadata = tensor.shape + tensor.stride()
        assert len(tensor.shape) == self.first_metadata[0]
        assert len(tensor.stride()) == self.first_metadata[1]

        # 动态扩容第二阶段元数据缓冲区
        if len(second_metadata) > len(self.second_metadata):
            self.second_metadata = torch.empty(len(second_metadata), dtype=torch.long, device=self.device)

        self.second_metadata[: len(second_metadata)].copy_(torch.tensor(second_metadata, dtype=torch.long))

        dist.send(
            self.second_metadata[: len(second_metadata)],
            dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_meta(self, from_rank: int, tag: int) -> P2PTensorMetaData:
        """同步接收张量的两阶段元数据。

        Args:
            from_rank (int): 源 rank（进程组内的编号）。
            tag (int): 通信标签。

        Returns:
            P2PTensorMetaData: 接收到的元数据。
        """
        dist.recv(
            self.first_metadata,
            src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )
        (
            num_shape,
            num_stride,
            is_contiguous,
            untyped_storage_size,
            storage_offset,
            dtype_id,
            requires_grad_id,
        ) = self.first_metadata

        second_metadata_num_elements = num_shape + num_stride

        # 动态扩容第二阶段元数据缓冲区
        if second_metadata_num_elements > len(self.second_metadata):
            self.second_metadata = torch.empty(second_metadata_num_elements, dtype=torch.long, device=self.device)

        dist.recv(
            self.second_metadata[:second_metadata_num_elements],
            src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

        shape = self.second_metadata[:num_shape]
        stride = self.second_metadata[num_shape:second_metadata_num_elements]

        return P2PTensorMetaData(
            dtype=ID_TO_DTYPE[dtype_id],
            requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
            shape=shape,
            stride=stride,
            is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
            untyped_storage_size=untyped_storage_size,
            storage_offset=storage_offset,
        )

    def isend_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0) -> List[dist.Work]:
        """异步发送多个张量到指定 rank。

        对每个张量执行三阶段发送：元数据1 → 元数据2 → 数据。

        Args:
            tensors (List[torch.Tensor]): 待发送的张量列表。
            to_rank (int): 目标 rank。
            tag (int): 通信标签。

        Returns:
            List[dist.Work]: 异步操作句柄列表。

        Raises:
            ValueError: 当尝试发送张量到自身时。
        """
        futures = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Current rank {current_rank} sending to rank {to_rank}. Nb_tensors: {len(tensors)}")
        for tensor in tensors:
            if to_rank != current_rank:
                self._send_meta(tensor, to_rank=to_rank, tag=tag)
                if tensor.is_contiguous():
                    buffer = tensor
                else:
                    # 非连续张量需要转换为连续存储后发送
                    buffer = view_as_contiguous(tensor)

                # 复数类型需要转换为实数视图
                buffer = torch.view_as_real(buffer) if buffer.is_complex() else buffer

                futures.append(
                    dist.isend(
                        buffer,
                        dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
                        group=self.pg,
                        tag=tag,
                    )
                )
            else:
                raise ValueError("Tried sending tensor to itself")
        return futures

    def irecv_tensors(
        self, num_tensors: int, from_rank: int, tag: int = 0
    ) -> Tuple[List[torch.Tensor], List[dist.Work]]:
        """异步接收多个张量从指定 rank。

        对每个张量执行三阶段接收：元数据1 → 元数据2 → 数据。

        Args:
            num_tensors (int): 待接收的张量数量。
            from_rank (int): 源 rank。
            tag (int): 通信标签。

        Returns:
            Tuple[List[torch.Tensor], List[dist.Work]]: 接收缓冲区列表和异步操作句柄列表。

        Raises:
            ValueError: 当尝试从自身接收张量时。
        """
        futures = []
        buffers = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Current rank {current_rank} receiving from rank {from_rank}. Nb_tensors: {num_tensors}")
        for _ in range(num_tensors):
            if from_rank != current_rank:
                meta = self._recv_meta(from_rank=from_rank, tag=tag)

                buffer = meta.create_empty_storage(device=self.device)

                futures.append(
                    dist.irecv(
                        buffer,
                        src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
                        group=self.pg,
                        tag=tag,
                    )
                )

                buffer = meta.reshape(buffer=buffer)

                buffers.append(buffer)
            else:
                raise ValueError("Tried receiving tensor from itself")
        return buffers, futures

    def send_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0):
        """同步发送多个张量，等待所有发送完成。"""
        futures = self.isend_tensors(tensors=tensors, to_rank=to_rank, tag=tag)
        for future in futures:
            future.wait()

    def recv_tensors(self, num_tensors: int, from_rank: int, tag: int = 0) -> List[torch.Tensor]:
        """同步接收多个张量，等待所有接收完成。"""
        buffers, futures = self.irecv_tensors(num_tensors=num_tensors, from_rank=from_rank, tag=tag)
        for future in futures:
            future.wait()
        return buffers


class BatchTensorSendRecvState:
    """批量张量收发状态管理器，支持通信与计算的重叠。

    在流水线并行的 1F1B 调度中，激活值的发送和接收可以与计算重叠。
    BatchTensorSendRecvState 将多个 send/recv 操作批量提交，
    通过 batch_isend_irecv 实现通信重叠优化。

    通信优化策略：
        - 将多个小张量的通信合并为一次 batch_isend_irecv 调用
        - 三阶段通信（元数据1 → 元数据2 → 数据）分批执行
        - 接收操作需要等待前一阶段的元数据才能确定下一阶段的缓冲区大小

    使用流程：
        1. 调用 add_send() 注册发送操作
        2. 调用 add_recv() 注册接收操作
        3. 调用 flush() 执行所有通信并返回接收到的张量

    Attributes:
        p2p (P2P): P2P 通信对象。
        first_metadata_p2p_ops (List[dist.P2POp]): 第一阶段元数据的 P2P 操作列表。
        second_metadata_p2p_ops (List[dist.P2POp]): 第二阶段元数据的 P2P 操作列表。
        data_p2p_ops (List[dist.P2POp]): 数据的 P2P 操作列表。
        recv_first_metadata_buffers (List[torch.Tensor]): 接收第一阶段元数据的缓冲区列表。
        recv_from_ranks (List[int]): 接收操作的源 rank 列表。
    """

    p2p: P2P
    first_metadata_p2p_ops: List[dist.P2POp]
    second_metadata_p2p_ops: List[dist.P2POp]
    data_p2p_ops: List[dist.P2POp]
    recv_first_metadata_buffers: List[torch.Tensor]
    recv_from_ranks: List[int]

    def __init__(self, p2p: P2P):
        self.p2p = p2p
        self._reset()

    def _reset(self):
        """重置所有通信状态。"""
        self.first_metadata_p2p_ops: List[dist.P2POp] = []
        self.second_metadata_p2p_ops: List[dist.P2POp] = []
        self.data_p2p_ops: List[dist.P2POp] = []
        self.recv_first_metadata_buffers: List[torch.Tensor] = []
        self.recv_from_ranks: List[int] = []

    def __str__(self):
        return f"BatchTensorSendRecvState(first_metadata_p2p_ops={len(self.first_metadata_p2p_ops)}, second_metadata_p2p_ops={len(self.second_metadata_p2p_ops)}, data_p2p_ops={len(self.data_p2p_ops)}, recv_first_metadata_buffers={len(self.recv_first_metadata_buffers)}, recv_from_ranks={self.recv_from_ranks})"

    def add_send(self, tensor: torch.Tensor, to_rank: int, tag: int = 0):
        """注册一个发送操作。

        将发送操作的三阶段 P2P 操作分别添加到对应的列表中，
        等待 flush() 时批量执行。

        Args:
            tensor (torch.Tensor): 待发送的张量。
            to_rank (int): 目标 rank。
            tag (int): 通信标签。
        """
        self.first_metadata_p2p_ops.append(
            self.p2p._send_first_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
        )
        self.second_metadata_p2p_ops.append(
            self.p2p._send_second_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
        )
        self.data_p2p_ops.append(
            self.p2p._send_data_p2p_op(tensor=view_as_contiguous(tensor), to_rank=to_rank, tag=tag)
        )

    def add_recv(self, from_rank: int, tag: int = 0) -> int:
        """注册一个接收操作。

        只添加第一阶段元数据的接收操作，因为第二阶段和数据接收
        需要等待第一阶段元数据到达后才能确定缓冲区大小。

        Args:
            from_rank (int): 源 rank。
            tag (int): 通信标签。

        Returns:
            int: 接收缓冲区在 recv_first_metadata_buffers 中的索引。
        """
        buffer, recv_op = self.p2p._recv_first_metadata_p2p_op(from_rank=from_rank, tag=tag)
        self.first_metadata_p2p_ops.append(recv_op)
        self.recv_first_metadata_buffers.append(buffer)
        self.recv_from_ranks.append(from_rank)
        return len(self.recv_first_metadata_buffers) - 1

    def _send_recv_first_metadata(self) -> List[List[int]]:
        """批量执行第一阶段元数据的发送和接收。"""
        reqs = dist.batch_isend_irecv(self.first_metadata_p2p_ops)
        for req in reqs:
            req.wait()
        # 尽早将 GPU 数据拷贝到 CPU，避免延迟同步影响性能
        first_metadatas = [tensor.tolist() for tensor in self.recv_first_metadata_buffers]
        return first_metadatas

    def _send_recv_second_metadata(self, first_metadata: List[List[int]]) -> List[List[int]]:
        """批量执行第二阶段元数据的发送和接收。

        根据第一阶段元数据中的维度信息，确定第二阶段缓冲区的大小。

        Args:
            first_metadata: 第一阶段元数据列表。

        Returns:
            List[List[int]]: 第二阶段元数据列表。
        """
        recv_second_metadata_buffers, recv_second_metadata_ops = zip(
            *(
                self.p2p._recv_second_metadata_p2p_op(
                    shape_length=num_shape, stride_length=num_stride, from_rank=from_rank
                )
                for (num_shape, num_stride, *_), from_rank in zip(first_metadata, self.recv_from_ranks)
            )
        )
        recv_second_metadata_ops = list(recv_second_metadata_ops)
        reqs = dist.batch_isend_irecv(self.second_metadata_p2p_ops + recv_second_metadata_ops)
        for req in reqs:
            req.wait()

        second_metadatas = [tensor.tolist() for tensor in recv_second_metadata_buffers]
        return second_metadatas

    def _send_recv_data(self, tensor_metadatas: List[P2PTensorMetaData]) -> List[torch.Tensor]:
        """批量执行张量数据的发送和接收。

        Args:
            tensor_metadatas: 接收张量的元数据列表。

        Returns:
            List[torch.Tensor]: 接收到的张量列表。
        """
        recv_data_buffers, recv_data_ops = zip(
            *(
                self.p2p._recv_data_p2p_op(tensor_metadata=tensor_metadata, from_rank=from_rank)
                for tensor_metadata, from_rank in zip(tensor_metadatas, self.recv_from_ranks)
            )
        )
        recv_data_ops = list(recv_data_ops)
        futures = dist.batch_isend_irecv(self.data_p2p_ops + recv_data_ops)
        for future in futures:
            future.wait()

        # 根据元数据设置接收张量的形状和步幅
        return [
            recv_data_buffer.as_strided(size=tuple(tensor_metadata.shape), stride=tuple(tensor_metadata.stride))
            for recv_data_buffer, tensor_metadata in zip(recv_data_buffers, tensor_metadatas)
        ]

    def flush(self) -> List[torch.Tensor]:
        """执行所有注册的通信操作，返回接收到的张量。

        通信按三阶段顺序执行：
            1. 发送/接收第一阶段元数据
            2. 发送/接收第二阶段元数据
            3. 发送/接收张量数据

        Returns:
            List[torch.Tensor]: 接收到的张量列表。如果没有接收操作，返回空列表。
        """
        assert len(self.recv_first_metadata_buffers) == len(
            self.recv_from_ranks
        ), f"len(self.recv_first_metadata_buffers)={len(self.recv_first_metadata_buffers)}, len(self.recv_from_ranks)={len(self.recv_from_ranks)} but should be equal."

        # 没有通信操作，直接返回
        if len(self.first_metadata_p2p_ops) == 0:
            return []

        # 只有发送操作，没有接收操作
        if len(self.recv_first_metadata_buffers) == 0:
            reqs = dist.batch_isend_irecv(
                self.first_metadata_p2p_ops + self.second_metadata_p2p_ops + self.data_p2p_ops
            )
            for req in reqs:
                req.wait()
            self._reset()
            return []

        # 三阶段通信
        logger.debug(f"First metadata: {[p2pop.op for p2pop in self.first_metadata_p2p_ops]}")
        first_metadatas = self._send_recv_first_metadata()
        second_metadatas = self._send_recv_second_metadata(first_metadatas)

        tensor_metadatas = [
            P2PTensorMetaData.from_metadata(first_metadata, second_metadata)
            for first_metadata, second_metadata in zip(first_metadatas, second_metadatas)
        ]

        recv_tensors = self._send_recv_data(tensor_metadatas)
        self._reset()

        return recv_tensors
