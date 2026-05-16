"""
流水线并行块模块 —— Nanotron 流水线并行的核心调度单元。

PipelineBlock 是流水线并行中最细粒度的调度单元。每个 PipelineBlock 定义了
模型中一段计算逻辑及其所在的 PP rank。所有 PipelineBlock 的定义存在于每个
rank 上，但只在指定的 rank 上实例化和执行计算。

核心机制：
    - TensorPointer: 当数据不在当前 rank 时，使用 TensorPointer 作为占位符，
      表示张量存在于另一个 rank 上
    - P2P 通信: 非 compute rank 将实际张量发送到 compute rank，
      compute rank 从前序 rank 接收张量
    - Pipeline State: 流水线状态管理器，缓冲激活和梯度以支持 1F1B 调度

与文献中的概念区别：
    文献中的 "pipeline stage" 通常指一个粒度块，而 Nanotron 的 PipelineBlock
    更细粒度——一个 pipeline stage 由多个连续的 PipelineBlock 组成。
"""

from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel.pipeline_parallel.functional import (
    recv_from_pipeline_state_buffer,
    send_to_pipeline_state_buffer,
)
from nanotron.parallel.pipeline_parallel.p2p import P2P, BatchTensorSendRecvState
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState, PipelineTrainBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


class PipelineBlock(nn.Module):
    """流水线并行的最细粒度调度块。

    PipelineBlock 封装了模型中的一段计算逻辑，并指定该逻辑在哪个 PP rank 上执行。
    在同一 PP 进程组中，所有 rank 都持有 PipelineBlock 的定义，但只有指定的 rank
    会实例化并执行实际计算，其他 rank 通过 TensorPointer 机制传递数据。

    工作流程：
        1. build_and_set_rank(): 指定该块在哪个 PP rank 上执行
        2. forward(): 根据当前 rank 是否为 compute rank 执行不同逻辑：
           - 非 compute rank: 将输入张量发送到 compute rank，返回 TensorPointer
           - compute rank: 接收来自前序 rank 的张量，执行计算，返回实际结果

    限制：
        - PipelineBlock 包装的模块必须返回 Dict[str, torch.Tensor]

    Attributes:
        p2p (P2P): 点对点通信对象，用于跨 rank 传输张量。
        pipeline_state (Optional[PipelineBatchState]): 流水线状态管理器，
            None 表示不使用特定流水线引擎（即普通前向/反向传播）。
        module_builder (Callable): 模块构建函数，延迟实例化以节省内存。
        module_kwargs (Dict[str, Any]): 传递给 module_builder 的参数。
        module_input_keys (Set[str]): 模块期望的输入键名集合。
        module_output_keys (Set[str]): 模块输出的键名集合。
        rank (int): 该块被分配的 PP rank（由 build_and_set_rank 设置）。
        pp_block (nn.Module): 实例化后的子模块（仅在 compute rank 上存在）。
    """

    def __init__(
        self,
        p2p: P2P,
        module_builder: Callable[..., Callable[..., Union[torch.Tensor, Dict[str, torch.Tensor]]]],
        module_kwargs: Dict[str, Any],
        module_input_keys: Set[str],
        module_output_keys: Set[str],
    ):
        """初始化 PipelineBlock。

        Args:
            p2p (P2P): 点对点通信对象，用于流水线阶段间的张量传输。
            module_builder (Callable): 模块构建函数，调用后返回一个可调用对象
                （通常是 nn.Module），其 forward 方法返回 Dict[str, torch.Tensor]。
            module_kwargs (Dict[str, Any]): 传递给 module_builder 的关键字参数。
            module_input_keys (Set[str]): 模块 forward 方法期望的输入键名集合，
                用于验证输入的完整性。
            module_output_keys (Set[str]): 模块 forward 方法输出的键名集合，
                用于构建 TensorPointer 返回值。
        """
        super().__init__()
        self.p2p = p2p
        self.pipeline_state: Optional[PipelineBatchState] = None

        self.module_builder = module_builder
        self.module_kwargs = module_kwargs
        self.module_input_keys = set(module_input_keys)
        self.module_output_keys = set(module_output_keys)

    def build_and_set_rank(self, pp_rank: int):
        """指定该 PipelineBlock 在哪个 PP rank 上执行计算，并在该 rank 上实例化模块。

        该方法在模型构建阶段调用，为每个 PipelineBlock 分配 PP rank。
        只有当前进程的 rank 等于 pp_rank 时，才会调用 module_builder 实例化子模块，
        其他 rank 上 self.pp_block 不存在，forward 时仅做数据转发。

        Args:
            pp_rank (int): 分配的 PP rank 编号，必须小于 PP 进程组大小。

        Raises:
            AssertionError: 当 pp_rank >= PP 进程组大小时。
        """
        assert pp_rank < self.p2p.pg.size()
        self.rank = pp_rank
        if pp_rank == dist.get_rank(self.p2p.pg):
            self.pp_block = self.module_builder(**self.module_kwargs)

    def extra_repr(self) -> str:
        """返回模块的额外表示信息，用于 print(model) 时显示 PP rank。"""
        return f"pp_rank={self.rank}" if hasattr(self, "rank") else ""

    def set_pipeline_state(self, pipeline_state: Optional[PipelineBatchState]):
        """设置流水线状态管理器。

        Args:
            pipeline_state (Optional[PipelineBatchState]): 流水线状态管理器。
                设置为 None 表示不使用特定流水线引擎。
        """
        self.pipeline_state = pipeline_state

    def forward(self, **kwargs):
        """执行前向传播，根据当前 rank 是否为 compute rank 走不同路径。

        核心逻辑分为两个分支：

        1. 非 compute rank（当前 rank ≠ self.rank）：
           - 遍历输入，将实际张量发送到 compute rank
           - TensorPointer 类型的输入直接跳过（已在其他地方处理）
           - 如果有 pipeline_state，使用缓冲区发送（支持 1F1B 调度）
           - 否则使用直接 P2P 发送（仅支持推理，不支持梯度传播）
           - 返回所有输出键对应的 TensorPointer

        2. compute rank（当前 rank == self.rank）：
           - 从前序 rank 接收 TensorPointer 指向的张量
           - 实际张量直接使用
           - 调用 pp_block 执行计算
           - 返回实际计算结果

        Args:
            **kwargs: 模块输入，键名必须与 module_input_keys 完全匹配。
                值可以是 torch.Tensor（实际数据）、TensorPointer（跨 rank 引用）
                或其他非张量对象（直接传递，不跨进程通信）。

        Returns:
            Dict[str, Union[TensorPointer, torch.Tensor, Any]]:
                - 非 compute rank: 返回 TensorPointer 字典
                - compute rank: 返回实际计算结果字典

        Raises:
            AssertionError: 当输入键名与 module_input_keys 不匹配时。
            ValueError: 当未使用流水线引擎但张量需要梯度时（无法跨 rank 传播梯度）。
        """
        assert self.module_input_keys == set(
            kwargs.keys()
        ), f"Expected {self.module_input_keys}, got {set(kwargs.keys())}"

        sorted_kwargs = sorted(kwargs.items(), key=get_sort_key(dist.get_rank(self.p2p.pg)))

        if dist.get_rank(self.p2p.pg) != self.rank:
            # === 非 compute rank 分支：发送数据并返回 TensorPointer ===
            batch_send_recv = BatchTensorSendRecvState(self.p2p)
            for name, tensor in sorted_kwargs:
                if isinstance(tensor, TensorPointer):
                    continue
                else:
                    assert isinstance(tensor, torch.Tensor)
                    if self.pipeline_state is not None:
                        # 使用流水线状态缓冲区发送，支持 1F1B 调度的延迟通信
                        send_to_pipeline_state_buffer(
                            tensor,
                            to_rank=self.rank,
                            p2p=self.p2p,
                            pipeline_state=self.pipeline_state,
                        )
                        continue

                    if tensor.requires_grad is True:
                        raise ValueError(
                            f"Pipeline engine is None and tensor requires grad. Tried sending a tensor to {self.rank}. Usually that means that your model is pipeline sharded and you haven't chosen a specific pipeline engine."
                        )

                    batch_send_recv.add_send(tensor=tensor, to_rank=self.rank)

            batch_send_recv.flush()
            return {k: TensorPointer(group_rank=self.rank) for k in self.module_output_keys}

        # === compute rank 分支：接收数据并执行计算 ===
        new_kwargs: Dict[str, torch.Tensor] = {}
        name_to_recv_id = {}
        batch_send_recv = BatchTensorSendRecvState(self.p2p)
        for name, tensor in sorted_kwargs:
            if isinstance(tensor, TensorPointer):
                # 在 1F1B 交错调度中，如果是第二个模型块，
                # 需要先发送之前的激活再接收当前激活
                if isinstance(self.pipeline_state, PipelineTrainBatchState):
                    for _ in range(len(self.pipeline_state.microbatches_activations_to_send)):
                        send_activation = self.pipeline_state.microbatches_activations_to_send.popleft()
                        send_activation()

                if self.pipeline_state is not None:
                    new_kwargs[name] = recv_from_pipeline_state_buffer(
                        from_rank=tensor.group_rank,
                        p2p=self.p2p,
                        pipeline_state=self.pipeline_state,
                    )
                    continue

                recv_id = batch_send_recv.add_recv(from_rank=tensor.group_rank)
                name_to_recv_id[name] = recv_id
            else:
                new_kwargs[name] = tensor

        recv_tensors = batch_send_recv.flush()
        assert len(recv_tensors) == len(name_to_recv_id)
        for name, recv_id in name_to_recv_id.items():
            assert name not in new_kwargs
            new_tensor = recv_tensors[recv_id]
            if new_tensor.requires_grad is True:
                raise ValueError(
                    f"Pipeline engine is None and tensor requires grad. Tried receiving a tensor to {self.rank}. Usually that means that your model is pipeline sharded and you haven't chosen a specific pipeline engine."
                )
            new_kwargs[name] = new_tensor

        output = self.pp_block(**new_kwargs)

        if isinstance(output, torch.Tensor):
            assert len(self.module_output_keys) == 1
            output = {next(iter(self.module_output_keys)): output}

        assert isinstance(output, dict), "Modules within a Pipeline Block have to return a Dict[str, torch.Tensor]"
        assert self.module_output_keys == set(
            output.keys()
        ), f"Expected {self.module_output_keys}, got {set(output.keys())}"

        return output


def get_min_max_rank(module: torch.nn.Module) -> Tuple[int, int]:
    """查找模块中所有 PipelineBlock 的最小和最大 PP rank。

    用于确定模型在流水线并行中的输入和输出 rank。

    Args:
        module (torch.nn.Module): 待搜索的模块。

    Returns:
        Tuple[int, int]: (最小 PP rank, 最大 PP rank)。
    """
    ranks = [module.rank for module in module.modules() if isinstance(module, PipelineBlock)]
    return min(ranks), max(ranks)


def get_sort_key(current_rank: int):
    """生成用于排序输入键的排序函数。

    排序策略：优先处理来自较早 rank 的张量，以便尽早释放前序 rank 的资源。
    来自当前 rank 的张量（本地数据）优先级最低。

    Args:
        current_rank (int): 当前进程的 rank。

    Returns:
        Callable: 排序键函数，接受 (name, tensor) 元组，返回 (rank, name) 排序键。
    """
    def sort_key(elt: Tuple[str, Union[torch.Tensor, TensorPointer]]):
        name, tensor = elt
        rank: int
        if isinstance(tensor, TensorPointer):
            rank = tensor.group_rank
        else:
            rank = current_rank
        return rank, name

    return sort_key
