"""
流水线并行引擎模块 —— Nanotron 流水线调度策略的实现。

本模块实现了两种流水线调度策略：
    - AllForwardAllBackward (AFAB): 先执行所有微批次的前向传播，再执行所有反向传播。
        实现简单，但内存占用高（需缓存所有微批次的激活值）。
    - OneForwardOneBackward (1F1B): 交替执行一个前向和一个反向传播。
        内存效率更高，是工业界常用的流水线调度策略。参考论文: https://arxiv.org/abs/2104.04473

调度策略的选择影响：
    - 内存占用：AFAB 需要缓存所有微批次激活，1F1B 只需缓存部分
    - 通信模式：1F1B 的前向和反向通信交错进行
    - 气泡率：两种策略的气泡率相同，但 1F1B 的峰值内存更低
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Union

import torch
from torch import nn as torch_nn
from torch.nn.parallel import DistributedDataParallel

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.data_parallel.utils import ddp_trigger_sync_in_bwd
from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
from nanotron.parallel.pipeline_parallel.state import PipelineTrainBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import ContextManagers

# from nanotron.logging.timers import nanotron_timer

logger = logging.get_logger(__name__)


class PipelineEngine(ABC):
    """流水线并行引擎的抽象基类，定义了前向/反向传播和批次迭代的接口。

    PipelineEngine 负责协调流水线并行中的微批次调度，包括：
        - 微批次的前向传播执行
        - 微批次的反向传播执行
        - 激活值的缓冲和梯度通信
        - DDP 梯度同步的时机控制

    子类必须实现 train_batch_iter 方法，定义具体的调度策略。

    Attributes:
        nb_microbatches (Optional[int]): 当前批次的微批次数量，在 train_batch_iter 中设置。
    """

    def __init__(self):
        self.nb_microbatches: Optional[int] = None
        pass

    def forward(
        self,
        context: ContextManagers,
        state: PipelineTrainBatchState,
        micro_batch: Dict[str, Union[torch.Tensor, TensorPointer]],
        model: torch_nn.Module,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """执行单个微批次的前向传播。

        Args:
            context (ContextManagers): 上下文管理器，通常包含 DDP no_sync 上下文。
            state (PipelineTrainBatchState): 流水线训练状态，跟踪前向/反向计数和激活缓冲。
            micro_batch (Dict[str, Union[torch.Tensor, TensorPointer]]): 微批次输入数据。
            model (torch_nn.Module): 待训练的模型。

        Returns:
            Dict[str, Union[torch.Tensor, TensorPointer]]: 前向传播输出。
                包含 "loss" 键，loss 已除以微批次数量进行归一化。

        Note:
            - loss 会被自动除以 nb_microbatches 进行归一化，确保梯度累积后
              得到正确的平均梯度
            - 非 TensorPointer 的 loss 会被注册到 state 中，等待后续反向传播
        """
        state.nb_forwards += 1
        log_rank(
            f"Forward micro batch id: {state.nb_forwards}",
            logger=logger,
            level=logging.DEBUG,
        )

        state.new_micro_batch_forward()
        with context:
            output = model(**micro_batch)

        if not isinstance(output, dict):
            output = {"loss": output}

        if not isinstance(output["loss"], TensorPointer):
            output["loss"] = output["loss"] / self.nb_microbatches

        if not isinstance(output["loss"], TensorPointer):
            assert output["loss"].requires_grad
            state.register_activation_requiring_backward(output["loss"])
        return output

    @staticmethod
    def _get_fwd_context(model: torch_nn.Module):
        """获取前向传播的上下文管理器。

        如果模型是 DDP 包装的，使用 no_sync() 避免在前向传播时触发梯度同步。

        Args:
            model (torch_nn.Module): 模型对象。

        Returns:
            ContextManagers: 包含必要上下文的上下文管理器。
        """
        is_ddp = isinstance(model, DistributedDataParallel)
        context = ContextManagers([model.no_sync()] if is_ddp else [])
        return context

    def backward(
        self, context: ContextManagers, state: PipelineTrainBatchState, grad_accumulator: Optional[GradientAccumulator]
    ):
        """执行单个微批次的反向传播。

        从 state 中弹出最近注册的激活值，对其求和后执行反向传播。

        Args:
            context (ContextManagers): 上下文管理器，控制 DDP 和梯度累积器的同步行为。
            state (PipelineTrainBatchState): 流水线训练状态。
            grad_accumulator (Optional[GradientAccumulator]): 梯度累积器，
                None 时直接调用 torch.autograd.backward。

        Note:
            如果没有待反向传播的激活值（len(activations) == 0），则跳过反向传播。
        """
        state.nb_backwards += 1
        log_rank(
            f"Backward micro batch id: {state.nb_forwards}",
            logger=logger,
            level=logging.DEBUG,
        )
        activations = state.pop_last_activations_requiring_backward()
        if len(activations) == 0:
            return

        with context:
            if grad_accumulator is None:
                sum(activations).backward()
            else:
                grad_accumulator.backward(sum(activations))

    def _get_bwd_context(
        self,
        model: torch_nn.Module,
        nb_backwards: int,
        grad_accumulator: Optional[GradientAccumulator],
    ):
        """获取反向传播的上下文管理器。

        控制梯度同步的时机：
            - 非最后一个微批次：使用 no_sync() 延迟梯度同步
            - 最后一个微批次：触发 DDP 梯度同步

        Args:
            model (torch_nn.Module): 模型对象。
            nb_backwards (int): 当前已完成的反向传播次数。
            grad_accumulator (Optional[GradientAccumulator]): 梯度累积器。

        Returns:
            ContextManagers: 包含必要上下文的上下文管理器。

        Raises:
            AssertionError: 当 nb_microbatches 未设置时。
        """
        assert (
            self.nb_microbatches is not None
        ), "You must call `train_batch_iter` first and set `self.nb_microbatches`"
        is_ddp = isinstance(model, DistributedDataParallel)
        context_list = []
        if is_ddp:
            if grad_accumulator is not None and nb_backwards < self.nb_microbatches - 1:
                context_list.append(grad_accumulator.no_sync())
            if nb_backwards == self.nb_microbatches - 1:
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        context = ContextManagers(context_list)
        return context

    @torch.profiler.record_function("train_batch_iter")
    @abstractmethod
    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """训练批次的迭代器，定义具体的微批次调度策略。

        子类必须实现此方法，定义前向和反向传播的执行顺序。

        Args:
            model (torch_nn.Module): 待训练的模型。
            pg (ProcessGroup): 流水线并行的进程组。
            batch (Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]): 输入批次数据。
            nb_microbatches (int): 微批次数量。
            grad_accumulator (Optional[GradientAccumulator]): 梯度累积器。

        Returns:
            Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]: 各微批次的输出结果。
        """
        ...

    @torch.inference_mode()
    def validate_batch_iter(
        self,
        model: torch_nn.Module,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """验证批次的迭代器，执行所有微批次的前向传播（不进行反向传播）。

        Args:
            model (torch_nn.Module): 待验证的模型。
            batch (Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]): 输入批次数据。
            nb_microbatches (int): 微批次数量。

        Returns:
            Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]: 各微批次的输出结果。
                loss 值已 detach，不保留计算图。
        """
        state = PipelineTrainBatchState()
        self.nb_microbatches = nb_microbatches

        outputs = []

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    send_activation()

                if not isinstance(output, dict):
                    output = {"loss": output}

                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

        return outputs

    def __str__(self):
        return self.__class__.__name__

    def __format__(self, format_spec):
        return str(self)


class AllForwardAllBackwardPipelineEngine(PipelineEngine):
    """全前向全反向（AFAB）流水线调度引擎。

    AFAB 策略先执行所有微批次的前向传播，然后执行所有微批次的反向传播。

    优点：
        - 实现简单
        - 调试方便

    缺点：
        - 峰值内存高：需要缓存所有微批次的中间激活值
        - 不适合大模型或大批次训练

    调度示意（PP=4, 4个微批次）：
        Rank 0: F0 F1 F2 F3 B0 B1 B2 B3
        Rank 1: F0 F1 F2 F3 B0 B1 B2 B3
        Rank 2: F0 F1 F2 F3 B0 B1 B2 B3
        Rank 3: F0 F1 F2 F3 B0 B1 B2 B3
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "afab"

    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """AFAB 调度：先执行所有前向，再执行所有反向。

        Args:
            model (torch_nn.Module): 待训练的模型。
            pg (ProcessGroup): 流水线并行的进程组。
            batch (Iterable[Dict]): 输入批次数据。
            nb_microbatches (int): 微批次数量。
            grad_accumulator (Optional[GradientAccumulator]): 梯度累积器。

        Returns:
            Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]: 各微批次的输出结果。
        """
        state = PipelineTrainBatchState()
        self.nb_microbatches = nb_microbatches

        outputs = []

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # === 阶段一：所有前向传播 ===
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    send_activation()

                if not isinstance(output, dict):
                    output = {"loss": output}

                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

            # === 阶段二：所有反向传播 ===
            for _ in range(len(state.microbatches_activations_requiring_backward)):
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

                for _ in range(len(state.microbatches_grads_to_send)):
                    send_grads = state.microbatches_grads_to_send.popleft()
                    send_grads()

            state.check_buffers_empty()

            return outputs


class OneForwardOneBackwardPipelineEngine(PipelineEngine):
    """一前一后（1F1B）流水线调度引擎。

    1F1B 策略在预热阶段后，交替执行一个前向和一个反向传播，
    有效降低了峰值内存占用。

    优点：
        - 峰值内存低：只需缓存 (PP_SIZE - current_rank) 个微批次的激活值
        - 工业界常用的流水线调度策略

    缺点：
        - 实现较复杂
        - 需要微批次数量 >= PP_SIZE - 1

    调度示意（PP=4, 8个微批次）：
        Rank 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
        Rank 1:    F0 F1 F2    B0 F3 B1 F4 B2 F5 B3    B4 B5 B6 B7
        Rank 2:       F0 F1       B0 F2 B1 F3 B2       B3 B4 B5 B6 B7
        Rank 3:          F0          B0 F1 B1          B2 B3 B4 B5 B6 B7

    参考: https://arxiv.org/abs/2104.04473
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "1f1b"

    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """1F1B 调度：预热后交替执行前向和反向传播。

        调度分为三个阶段：
            1. 预热阶段：执行 (PP_SIZE - current_rank - 1) 个前向传播
            2. 稳态阶段：交替执行一个前向和一个反向传播
            3. 冷却阶段：执行剩余的反向传播

        Args:
            model (torch_nn.Module): 待训练的模型。
            pg (ProcessGroup): 流水线并行的进程组。
            batch (Iterable[Dict]): 输入批次数据。
            nb_microbatches (int): 微批次数量。
            grad_accumulator (Optional[GradientAccumulator]): 梯度累积器。

        Returns:
            Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]: 各微批次的输出结果。

        Raises:
            AssertionError: 当微批次数量 < PP_SIZE - 1 时。
        """
        self.nb_microbatches = nb_microbatches
        assert (
            self.nb_microbatches >= pg.size() - 1
        ), f"Number of microbatches ({self.nb_microbatches}) must be at least PP_SIZE-1={pg.size() - 1} when using the OneForwardOneBackwardPipelineEngine"

        state = PipelineTrainBatchState()

        outputs = []
        batch = iter(batch)

        current_pp_rank = dist.get_rank(pg)

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # === 阶段一：预热阶段 ===
            # 执行 (PP_SIZE - current_rank - 1) 个前向传播，填充流水线
            for _ in range(pg.size() - current_pp_rank - 1):
                micro_batch = next(batch)
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)

                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    send_activation()

                if not isinstance(output, dict):
                    output = {"loss": output}

                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    send_activation()

                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

            # === 阶段二：稳态阶段 ===
            # 交替执行一个前向和一个反向传播
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)

                if not isinstance(output, dict):
                    output = {"loss": output}

                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

            # === 阶段三：冷却阶段 ===
            # 执行剩余的反向传播
            assert len(state.microbatches_activations_requiring_backward) == pg.size() - current_pp_rank - 1
            assert (
                len(state.microbatches_activations_to_send) == 0
            ), f"There are activations left for me to send still: {len(state.microbatches_activations_to_send)}"
            assert (
                len(state.microbatches_activations_to_recv) == 0
            ), f"There are activations left for me to recv still: {len(state.microbatches_activations_to_recv)}"

            for _ in range(len(state.microbatches_grads_to_send)):
                send_grads = state.microbatches_grads_to_send.popleft()
                send_grads()
            for _ in range(len(state.microbatches_activations_requiring_backward)):
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

                for _ in range(len(state.microbatches_grads_to_send)):
                    send_grads = state.microbatches_grads_to_send.popleft()
                    send_grads()

            state.check_buffers_empty()

        return outputs
