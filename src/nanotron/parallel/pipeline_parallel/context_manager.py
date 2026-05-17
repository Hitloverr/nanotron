"""
流水线并行上下文管理器 —— 管理流水线状态的临时绑定。

本模块提供了将流水线状态（PipelineBatchState）临时绑定到模型中
所有 PipelineBlock 的上下文管理器。

工作原理：
    在训练循环中，每个批次需要创建一个新的 PipelineBatchState，
    并将其绑定到模型中所有的 PipelineBlock 上，使得 PipelineBlock
    在前向/反向传播时可以注册和执行通信操作。

    attach_pipeline_state_to_model 使用 Python 上下文管理器协议，
    确保在训练批次结束后自动解绑流水线状态，避免状态泄漏。

使用方式：
    with attach_pipeline_state_to_model(model, pipeline_state):
        # 在此上下文中，所有 PipelineBlock 都可以访问 pipeline_state
        output = model(input)
        output.backward()
    # 退出上下文后，pipeline_state 自动解绑

与 PipelineBlock 的关系：
    PipelineBlock 在前向传播中需要 pipeline_state 来注册
    激活值的发送/接收操作。pipeline_state 管理了通信缓冲区，
    由流水线调度引擎在合适的时机触发通信。
"""

from contextlib import contextmanager

from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState
from torch import nn as torch_nn


@contextmanager
def attach_pipeline_state_to_model(model: torch_nn.Module, pipeline_state: PipelineBatchState):
    """将流水线状态临时绑定到模型中所有 PipelineBlock。

    遍历模型中的所有 PipelineBlock，将 pipeline_state 设置到每个 block 上。
    退出上下文时恢复原始状态（None）。

    约束条件：
        - 每个 PipelineBlock 在绑定前必须没有已绑定的 pipeline_state
        - 这确保了流水线状态不会被意外覆盖

    Args:
        model (torch_nn.Module): 模型对象。
        pipeline_state (PipelineBatchState): 流水线批次状态。

    Yields:
        None: 上下文管理器不返回值。
    """
    old_pipeline_states = []

    for name, module in model.named_modules():
        if not isinstance(module, PipelineBlock):
            continue

        old_pipeline_state = module.pipeline_state
        assert old_pipeline_state is None, "We never replace an old pipeline engine, we just set one when there's none"

        old_pipeline_states.append((old_pipeline_state, module))

        module.set_pipeline_state(pipeline_state)

    try:
        yield
    finally:
        # 恢复所有 PipelineBlock 的原始状态
        for old_pipeline_state, module in old_pipeline_states:
            module.set_pipeline_state(old_pipeline_state)
