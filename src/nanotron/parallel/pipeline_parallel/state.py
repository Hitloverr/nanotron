"""
流水线并行状态管理模块 —— 管理流水线训练中的激活值和梯度缓冲。

本模块定义了流水线并行训练中的状态管理器，负责缓冲和管理
微批次之间的激活值发送/接收和梯度发送/接收操作。

核心概念：
    - SendActivation/RecvActivation: 延迟执行的激活值发送/接收操作
    - SendGrad/RecvGrad: 延迟执行的梯度发送/接收操作
    - PipelineBatchState: 流水线状态管理器的抽象基类
    - PipelineTrainBatchState: 训练时的流水线状态管理器（支持 1F1B 调度）
    - PipelineEvalBatchState: 推理时的流水线状态管理器（仅前向传播）

与 1F1B 调度的关系：
    在 1F1B 调度中，激活值的发送和接收不是立即执行的，而是注册到
    状态管理器的缓冲区中，由调度引擎在合适的时机触发执行。
    这种延迟执行机制允许通信与计算的重叠，提高 GPU 利用率。

通信顺序：
    run_communication() 按以下顺序执行一次通信：
        1. 发送一个激活值（到下一个 PP rank）
        2. 接收一个激活值（从上一个 PP rank）
        3. 发送一个梯度（到上一个 PP rank）
        4. 接收一个梯度（从下一个 PP rank）

    这个顺序确保了通信不会死锁：先发送后接收，避免环形等待。
"""

import collections
import dataclasses
from abc import ABC, abstractmethod
from typing import List

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.p2p import P2P

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class SendActivation:
    """延迟执行的激活值发送操作。

    将激活值发送到指定的 PP rank。调用时执行实际的 P2P 通信。

    Attributes:
        activation (torch.Tensor): 待发送的激活值张量。
        to_rank (int): 目标 PP rank。
        p2p (P2P): P2P 通信对象。
    """

    activation: torch.Tensor
    to_rank: int
    p2p: P2P

    def __call__(self):
        self.p2p.send_tensors([self.activation], to_rank=self.to_rank)


@dataclasses.dataclass
class RecvActivation:
    """延迟执行的激活值接收操作。

    从指定的 PP rank 接收激活值。调用时执行实际的 P2P 通信。

    Attributes:
        from_rank (int): 源 PP rank。
        p2p (P2P): P2P 通信对象。
    """

    from_rank: int
    p2p: P2P

    def __call__(self) -> torch.Tensor:
        return self.p2p.recv_tensors(num_tensors=1, from_rank=self.from_rank)[0]


@dataclasses.dataclass
class SendGrad:
    """延迟执行的梯度发送操作。

    将梯度发送到指定的 PP rank（通常是前一个 stage）。

    Attributes:
        grad (torch.Tensor): 待发送的梯度张量。
        to_rank (int): 目标 PP rank。
        p2p (P2P): P2P 通信对象。
    """

    grad: torch.Tensor
    to_rank: int
    p2p: P2P

    def __call__(self):
        self.p2p.send_tensors([self.grad], to_rank=self.to_rank)


@dataclasses.dataclass
class RecvGrad:
    """延迟执行的梯度接收操作。

    从指定的 PP rank 接收梯度（通常是后一个 stage）。

    Attributes:
        from_rank (int): 源 PP rank。
        p2p (P2P): P2P 通信对象。
    """

    from_rank: int
    p2p: P2P

    def __call__(self) -> torch.Tensor:
        return self.p2p.recv_tensors(num_tensors=1, from_rank=self.from_rank)[0]


class PipelineBatchState(ABC):
    """流水线批次状态的抽象基类。

    定义了流水线训练/推理中需要实现的状态管理接口。
    子类需要实现激活值和梯度的注册、通信和弹出等操作。

    Attributes:
        activations_buffer: 激活值缓冲区，存储接收到的激活值。
    """

    activations_buffer = collections.deque()

    @abstractmethod
    def register_activation_requiring_backward(self, activation: torch.Tensor):
        """注册需要反向传播的激活值。"""
        ...

    @abstractmethod
    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        """注册激活值发送操作。"""
        ...

    @abstractmethod
    def register_recv_activation(self, from_rank: int, p2p: P2P):
        """注册激活值接收操作。"""
        ...

    @abstractmethod
    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        """注册梯度发送操作。"""
        ...

    @abstractmethod
    def register_recv_grad(self, from_rank: int, p2p: P2P):
        """注册梯度接收操作。"""
        ...

    @abstractmethod
    def run_communication(self, send_only_activation: bool = False):
        """执行一次通信操作。"""
        ...

    @abstractmethod
    def new_micro_batch_forward(self):
        """通知状态管理器开始一个新的微批次前向传播。"""
        ...

    @abstractmethod
    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        """弹出最早注册的需要反向传播的激活值。"""
        ...


@dataclasses.dataclass
class PipelineTrainBatchState(PipelineBatchState):
    """训练时的流水线批次状态管理器。

    管理 1F1B 调度中的激活值和梯度缓冲，支持延迟通信。
    所有通信操作都注册到缓冲区中，由调度引擎在合适的时机触发。

    缓冲区结构：
        - microbatches_activations_to_send: 待发送的激活值队列
        - microbatches_activations_to_recv: 待接收的激活值队列
        - microbatches_grads_to_send: 待发送的梯度队列
        - microbatches_grads_to_recv: 待接收的梯度队列
        - activations_buffer: 已接收的激活值缓冲区
        - grads_buffer: 已接收的梯度缓冲区
        - microbatches_activations_requiring_backward: 需要反向传播的激活值队列
          （外层按微批次索引，内层按激活值索引）

    通信执行顺序（run_communication）：
        1. 发送一个激活值 → 2. 接收一个激活值 → 3. 发送一个梯度 → 4. 接收一个梯度
        这个顺序确保通信不会死锁。

    Attributes:
        nb_backwards (int): 已完成的反向传播次数。
        nb_forwards (int): 已完成的前向传播次数。
    """

    microbatches_activations_to_send = collections.deque()
    microbatches_activations_to_recv = collections.deque()
    microbatches_grads_to_send = collections.deque()
    microbatches_grads_to_recv = collections.deque()
    grads_buffer = collections.deque()

    # 外层按微批次索引，内层按激活值索引
    microbatches_activations_requiring_backward = collections.deque()

    nb_backwards = 0
    nb_forwards = 0

    def register_activation_requiring_backward(self, activation: torch.Tensor):
        """将激活值注册到当前微批次的反向传播列表中。

        Args:
            activation (torch.Tensor): 需要反向传播的激活值。
        """
        self.microbatches_activations_requiring_backward[-1].append(activation)

    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        """注册激活值发送操作到缓冲区。

        Args:
            activation (torch.Tensor): 待发送的激活值。
            to_rank (int): 目标 PP rank。
            p2p (P2P): P2P 通信对象。
        """
        self.microbatches_activations_to_send.append(SendActivation(activation=activation, to_rank=to_rank, p2p=p2p))

    def register_recv_activation(self, from_rank: int, p2p: P2P):
        """注册激活值接收操作到缓冲区。

        Args:
            from_rank (int): 源 PP rank。
            p2p (P2P): P2P 通信对象。
        """
        self.microbatches_activations_to_recv.append(RecvActivation(from_rank=from_rank, p2p=p2p))

    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        """注册梯度发送操作到缓冲区。

        Args:
            grad (torch.Tensor): 待发送的梯度。
            to_rank (int): 目标 PP rank。
            p2p (P2P): P2P 通信对象。
        """
        self.microbatches_grads_to_send.append(SendGrad(grad=grad, to_rank=to_rank, p2p=p2p))

    def register_recv_grad(self, from_rank: int, p2p: P2P):
        """注册梯度接收操作到缓冲区。

        Args:
            from_rank (int): 源 PP rank。
            p2p (P2P): P2P 通信对象。
        """
        self.microbatches_grads_to_recv.append(RecvGrad(from_rank=from_rank, p2p=p2p))

    def run_communication(self, send_only_activation: bool = False):
        """执行一次通信操作，按固定顺序处理发送/接收。

        通信顺序：
            1. 发送一个激活值（如果缓冲区非空）
            2. 接收一个激活值（如果缓冲区非空），存入 activations_buffer
            3. 发送一个梯度（如果缓冲区非空）
            4. 接收一个梯度（如果缓冲区非空），存入 grads_buffer

        特殊处理：
            - send_only_activation=True 时，只执行步骤 1（用于不需要梯度的激活值）
            - 接收到的激活值如果不需要梯度，则跳过后续通信
            - 接收梯度前，可能需要先发送更多激活值以确保通信不阻塞

        Args:
            send_only_activation (bool): 是否只发送激活值而不执行后续通信。
                用于 SendTensorWithoutGradientToPipelineBuffer 的反向传播。
        """
        log_rank(
            f"activation_to_send: {len(self.microbatches_activations_to_send)} | "
            f"activation_to_recv: {len(self.microbatches_activations_to_recv)} | "
            f"grads_to_send: {len(self.microbatches_grads_to_send)} | "
            f"grads_to_recv: {len(self.microbatches_grads_to_recv)} | "
            f"activation_buffer: {len(self.activations_buffer)} | "
            f"grads_buffer: {len(self.grads_buffer)}",
            logger=logger,
            level=logging.DEBUG,
        )
        # 步骤 1：发送一个激活值
        activation_send_requires_grad = False
        if len(self.microbatches_activations_to_send) > 0:
            send_activation = self.microbatches_activations_to_send.popleft()
            activation_send_requires_grad = send_activation.activation.requires_grad
            send_activation()
            if send_only_activation:
                return

        # 步骤 2：接收一个激活值
        if len(self.microbatches_activations_to_recv) > 0:
            recv_activation = self.microbatches_activations_to_recv.popleft()
            recv_activation_tensor = recv_activation()
            self.activations_buffer.append(recv_activation_tensor)
            # 如果接收到的激活值不需要梯度，跳过后续通信
            if recv_activation_tensor.requires_grad is False:
                return

        # 步骤 3：发送一个梯度
        if len(self.microbatches_grads_to_send) > 0:
            send_grad = self.microbatches_grads_to_send.popleft()
            send_grad()

        # 步骤 4：接收一个梯度
        if len(self.microbatches_grads_to_recv) > 0:
            # 在接收梯度前，可能需要先发送更多不需要梯度的激活值
            # 以确保通信通道不被阻塞
            while len(self.microbatches_activations_to_send) > 0 and not activation_send_requires_grad:
                send_activation = self.microbatches_activations_to_send.popleft()
                activation_send_requires_grad = send_activation.activation.requires_grad
                send_activation()
            recv_grad = self.microbatches_grads_to_recv.popleft()
            self.grads_buffer.append(recv_grad())

    def new_micro_batch_forward(self):
        """通知状态管理器开始一个新的微批次前向传播。

        在微批次列表中添加一个新的空列表，用于存储该微批次的激活值。
        """
        self.microbatches_activations_requiring_backward.append(collections.deque())

    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        """弹出最早注册的需要反向传播的激活值列表。

        Returns:
            List[torch.Tensor]: 最早注册的微批次的所有需要反向传播的激活值。
        """
        return self.microbatches_activations_requiring_backward.popleft()

    def check_buffers_empty(self):
        """验证所有缓冲区都已清空。

        在一个训练批次结束后调用，确保没有遗漏的通信操作。

        Raises:
            AssertionError: 当任何缓冲区非空时。
        """
        assert (
            len(self.microbatches_activations_requiring_backward) == 0
        ), f"There are still activations that require backward: {len(self.microbatches_activations_requiring_backward)}"
        assert (
            len(self.microbatches_activations_to_send) == 0
        ), f"There are activations left for me to send still: {len(self.microbatches_activations_to_send)}"
        assert (
            len(self.microbatches_activations_to_recv) == 0
        ), f"There are activations left for me to recv still: {len(self.microbatches_activations_to_recv)}"
        assert (
            len(self.microbatches_grads_to_send) == 0
        ), f"There are gradients left for me to send still: {len(self.microbatches_grads_to_send)}"
        assert (
            len(self.microbatches_grads_to_recv) == 0
        ), f"There are gradients left for me to recv still: {len(self.microbatches_grads_to_recv)}"


@dataclasses.dataclass
class PipelineEvalBatchState(PipelineBatchState):
    """推理时的流水线批次状态管理器。

    推理模式下只进行前向传播，不需要梯度通信。
    激活值的发送和接收在注册时立即触发（如果有配对的发送/接收操作）。

    通信策略：
        当同时有待发送和待接收的激活值时，立即执行一次通信。
        通信顺序根据 rank 的高低决定：
            - 最低 rank（同时向高 rank 发送和从高 rank 接收）：先发送后接收
            - 其他 rank：先接收后发送
        这种策略避免了环形通信中的死锁。

    Attributes:
        microbatches_activations_to_send: 待发送的激活值队列
        microbatches_activations_to_recv: 待接收的激活值队列
        activations_buffer: 已接收的激活值缓冲区
    """

    microbatches_activations_to_send = collections.deque()
    microbatches_activations_to_recv = collections.deque()
    activations_buffer = collections.deque()

    def register_activation_requiring_backward(self, activation: torch.Tensor):
        """推理模式下不需要反向传播，空实现。"""
        pass

    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        """注册激活值发送操作，如果同时有接收操作则立即触发通信。"""
        self.microbatches_activations_to_send.append(SendActivation(activation=activation, to_rank=to_rank, p2p=p2p))

        # 如果同时有发送和接收操作，立即触发通信
        if len(self.microbatches_activations_to_recv) > 0 and len(self.microbatches_activations_to_recv) > 0:
            self.run_communication()

    def register_recv_activation(self, from_rank: int, p2p: P2P):
        """注册激活值接收操作，如果同时有发送操作则立即触发通信。"""
        self.microbatches_activations_to_recv.append(RecvActivation(from_rank=from_rank, p2p=p2p))

        # 如果同时有发送和接收操作，立即触发通信
        if len(self.microbatches_activations_to_recv) > 0 and len(self.microbatches_activations_to_recv) > 0:
            self.run_communication()

    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        """推理模式下不支持梯度发送。"""
        raise NotImplementedError("You can't register a send grad in pipeline eval mode")

    def register_recv_grad(self, from_rank: int, p2p: P2P):
        """推理模式下不支持梯度接收。"""
        raise NotImplementedError("You can't register a recv grad in pipeline eval mode")

    def new_micro_batch_forward(self):
        """推理模式下不需要微批次管理，空实现。"""
        pass

    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        """推理模式下不需要反向传播，空实现。"""
        pass

    def run_communication(self, send_only_activation: bool = False):
        """执行一次推理模式的通信操作。

        通信顺序策略（避免死锁）：
            - 判断当前 rank 是否是通信环中的最低 rank
            - 最低 rank：先发送后接收
            - 其他 rank：先接收后发送

        这确保了在环形通信中，至少有一个 rank 先发送，
        打破潜在的环形等待。
        """
        send_activation = None
        for _ in range(min(1, len(self.microbatches_activations_to_send))):
            send_activation = self.microbatches_activations_to_send.popleft()

        recv_activation = None
        for _ in range(min(1, len(self.microbatches_activations_to_recv))):
            recv_activation = self.microbatches_activations_to_recv.popleft()

        if send_activation is None:
            if recv_activation is None:
                raise ValueError("Why the hell do we communicate when there's nothing to communicate?")
            self.activations_buffer.append(recv_activation())
        else:
            if recv_activation is None:
                send_activation()
            else:
                # 根据通信方向决定顺序，避免死锁
                p2p = send_activation.p2p
                assert p2p == recv_activation.p2p
                # 判断是否是最低 rank（同时向高 rank 发送和从高 rank 接收）
                is_lowest = send_activation.to_rank > dist.get_rank(
                    p2p.pg
                ) and recv_activation.from_rank > dist.get_rank(p2p.pg)
                if is_lowest:
                    # 最低 rank 先发送后接收
                    send_activation()
                    self.activations_buffer.append(recv_activation())
                else:
                    # 其他 rank 先接收后发送
                    self.activations_buffer.append(recv_activation())
                    send_activation()

    def check_buffers_empty(self):
        """验证所有缓冲区都已清空。"""
        assert (
            len(self.microbatches_activations_to_send) == 0
        ), f"There are activations left for me to send still: {len(self.microbatches_activations_to_send)}"
        assert (
            len(self.microbatches_activations_to_recv) == 0
        ), f"There are activations left for me to recv still: {len(self.microbatches_activations_to_recv)}"
        assert (
            len(self.activations_buffer) == 0
        ), f"There are activations left in the buffer: {len(self.activations_buffer)}"
