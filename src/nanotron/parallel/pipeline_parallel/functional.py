"""
流水线并行函数式接口 —— 可微分的张量发送/接收操作。

本模块实现了流水线并行中可微分的张量发送和接收操作，通过自定义
autograd.Function 将 P2P 通信嵌入到 PyTorch 的计算图中，
使得梯度可以自动跨 PP rank 反向传播。

核心设计思路：
    在流水线并行中，激活值需要从前一个 stage 发送到后一个 stage，
    梯度需要从后一个 stage 发回到前一个 stage。为了使这一过程
    对 PyTorch 的 autograd 系统透明，我们使用自定义的
    torch.autograd.Function 来实现：

    前向传播：
        - SendTensorToPipelineBuffer: 发送激活值，返回一个标量占位符
          （触发反向传播的钩子）
        - RecvTensorFromPipelineBuffer: 接收激活值，直接透传

    反向传播：
        - SendTensorToPipelineBuffer.backward: 接收来自后一个 stage 的梯度
        - RecvTensorFromPipelineBuffer.backward: 发送梯度到前一个 stage

    这种设计确保了梯度可以自动沿着流水线的反方向流动，
    而不需要手动管理梯度的传输。

与 PipelineBlock 的关系：
    PipelineBlock.forward() 中调用 send_to_pipeline_state_buffer() 和
    recv_from_pipeline_state_buffer() 来实现跨 rank 的数据传输，
    这两个函数内部使用了本模块定义的 autograd.Function。

注意事项：
    - 不使用流水线引擎时（pipeline_state is None），不支持梯度传播
    - SendTensorToPipelineBuffer 使用 CPU 标量作为占位符，
      这是为了触发 PyTorch 的反向传播机制
"""

import torch
from nanotron import logging
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState

logger = logging.get_logger(__name__)


class SendTensorToPipelineBuffer(torch.autograd.Function):
    """可微分的张量发送操作。

    前向传播时将激活值注册到流水线状态缓冲区中（延迟发送），
    反向传播时从流水线状态缓冲区接收梯度。

    前向传播：
        1. 将激活值注册到 pipeline_state 的发送缓冲区
        2. 返回一个 CPU 标量 1.0（requires_grad=True）作为占位符
           这个占位符用于触发反向传播链

    反向传播：
        1. 从 pipeline_state 注册一个梯度接收操作
        2. 如果梯度缓冲区为空，执行一次通信
        3. 从梯度缓冲区弹出接收到的梯度作为当前张量的梯度

    这种设计的关键洞察：
        - 前向发送的激活值在反向传播时需要接收对应的梯度
        - 梯度的来源是发送目标 rank（to_rank），因此反向时从 to_rank 接收
    """

    @staticmethod
    def forward(
        ctx,
        activation: torch.Tensor,
        to_rank: int,
        p2p: P2P,
        pipeline_state: PipelineBatchState,
    ):
        assert activation.requires_grad
        ctx.p2p = p2p
        ctx.to_rank = to_rank
        ctx.pipeline_state = pipeline_state

        # 将激活值注册到发送缓冲区（延迟发送，由调度引擎触发）
        pipeline_state.register_send_activation(activation, to_rank=to_rank, p2p=p2p)

        # 返回 CPU 标量作为占位符，触发反向传播
        # 这是关键技巧：PyTorch autograd 要求前向输出参与计算图，
        # 使用 CPU 标量避免不必要的 GPU 显存占用
        return torch.tensor(1, dtype=torch.float, device="cpu", requires_grad=True)

    @staticmethod
    def backward(ctx, grad_tensor):
        p2p = ctx.p2p
        to_rank = ctx.to_rank
        pipeline_state = ctx.pipeline_state

        # 注册从目标 rank 接收梯度的操作
        # 注意：梯度从 to_rank 接收，因为前向时发送到了 to_rank
        pipeline_state.register_recv_grad(from_rank=to_rank, p2p=p2p)
        if len(pipeline_state.grads_buffer) == 0:
            # 梯度缓冲区为空，需要执行一次通信来接收梯度
            pipeline_state.run_communication()

        grad_tensor = pipeline_state.grads_buffer.popleft()

        return grad_tensor, None, None, None


class SendTensorWithoutGradientToPipelineBuffer(torch.autograd.Function):
    """发送不需要梯度的张量到流水线缓冲区。

    用于发送不需要反向传播的激活值（如推理模式或 detach 的张量）。
    前向传播时发送激活值，反向传播时只发送激活值（不接收梯度）。

    前向传播：
        1. 将激活值注册到发送缓冲区
        2. 返回 CPU 标量占位符

    反向传播：
        1. 执行一次仅发送激活值的通信（send_only_activation=True）
        2. 不接收任何梯度
    """

    @staticmethod
    def forward(
        ctx,
        dummy_input: torch.Tensor,
        activation: torch.Tensor,
        to_rank: int,
        p2p: P2P,
        pipeline_state: PipelineBatchState,
    ):
        assert dummy_input.requires_grad
        assert activation.requires_grad is False
        ctx.p2p = p2p
        ctx.to_rank = to_rank
        ctx.pipeline_state = pipeline_state

        # 将激活值注册到发送缓冲区
        pipeline_state.register_send_activation(activation, to_rank=to_rank, p2p=p2p)

        return torch.tensor(1, dtype=torch.float, device="cpu", requires_grad=True)

    @staticmethod
    def backward(ctx, grad_tensor):
        pipeline_state = ctx.pipeline_state

        # 反向传播时只发送激活值，不接收梯度
        pipeline_state.run_communication(send_only_activation=True)

        return None, None, None, None, None


def send_to_pipeline_state_buffer(tensor: torch.Tensor, to_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
    """将张量发送到流水线状态缓冲区，支持梯度反向传播。

    根据张量是否需要梯度，选择不同的发送策略：
        - 需要梯度：使用 SendTensorToPipelineBuffer（反向时接收梯度）
        - 不需要梯度：使用 SendTensorWithoutGradientToPipelineBuffer（反向时只发送）

    两种情况都会将结果注册到 pipeline_state 的反向传播激活值列表中。

    Args:
        tensor (torch.Tensor): 待发送的张量。
        to_rank (int): 目标 PP rank。
        p2p (P2P): P2P 通信对象。
        pipeline_state (PipelineBatchState): 流水线状态管理器。
    """
    if tensor.requires_grad:
        result = SendTensorToPipelineBuffer.apply(tensor, to_rank, p2p, pipeline_state)
    else:
        # 使用 dummy_input 技巧：创建一个需要梯度的 CPU 标量，
        # 让 PyTorch autograd 系统在反向传播时调用 backward 方法
        dummy_input = torch.empty(1, dtype=torch.float, requires_grad=True, device="cpu")
        result = SendTensorWithoutGradientToPipelineBuffer.apply(dummy_input, tensor, to_rank, p2p, pipeline_state)

    # 将结果注册为需要反向传播的激活值
    pipeline_state.register_activation_requiring_backward(result)


class RecvTensorFromPipelineBuffer(torch.autograd.Function):
    """可微分的张量接收操作。

    前向传播时直接透传接收到的激活值，
    反向传播时将梯度注册到发送缓冲区。

    前向传播：
        直接返回接收到的激活值（激活值已在调用前通过
        pipeline_state.register_recv_activation 注册并接收）

    反向传播：
        将梯度注册到发送缓冲区，发送回前一个 stage（from_rank）
    """

    @staticmethod
    def forward(ctx, activation: torch.Tensor, from_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
        ctx.pipeline_state = pipeline_state
        ctx.p2p = p2p
        ctx.from_rank = from_rank

        return activation

    @staticmethod
    def backward(ctx, grad_tensor):
        pipeline_state = ctx.pipeline_state
        from_rank = ctx.from_rank
        p2p = ctx.p2p

        # 将梯度注册到发送缓冲区，发送回前一个 stage
        pipeline_state.register_send_grad(grad_tensor, to_rank=from_rank, p2p=p2p)

        return None, None, None, None


def recv_from_pipeline_state_buffer(from_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
    """从流水线状态缓冲区接收张量，支持梯度反向传播。

    接收流程：
        1. 注册接收操作到 pipeline_state
        2. 如果激活值缓冲区为空，执行一次通信
        3. 从缓冲区弹出接收到的激活值
        4. 使用 RecvTensorFromPipelineBuffer 包装，确保反向传播时梯度正确发送

    Args:
        from_rank (int): 源 PP rank。
        p2p (P2P): P2P 通信对象。
        pipeline_state (PipelineBatchState): 流水线状态管理器。

    Returns:
        torch.Tensor: 接收到的激活值，已包装为可微分操作。
    """
    pipeline_state.register_recv_activation(from_rank=from_rank, p2p=p2p)
    if len(pipeline_state.activations_buffer) == 0:
        # 激活值缓冲区为空，需要执行一次通信来接收激活值
        pipeline_state.run_communication()
    activation = pipeline_state.activations_buffer.popleft()
    return RecvTensorFromPipelineBuffer.apply(activation, from_rank, p2p, pipeline_state)
