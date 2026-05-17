"""
模型基类模块 —— Nanotron 模型系统的核心抽象和构建工具。

本模块定义了 Nanotron 中所有模型的基类和模型构建基础设施：
    - NanotronModel: 所有 Nanotron 模型的抽象基类，定义了模型必须实现的接口
    - DTypeInvariantTensor: 禁止修改数据类型的张量子类，用于保护关键张量
    - build_model: 模型构建函数，负责 PipelineBlock 的 rank 分配和负载均衡
    - init_on_device_and_dtype: 上下文管理器，控制参数初始化的设备和数据类型

设计原则：
    - 模型通过 PipelineBlock 组织，支持流水线并行
    - PipelineBlock 按 PP rank 分配时采用负载均衡策略
    - 参数初始化时统一控制设备和数据类型，避免不必要的显存分配
"""

import threading
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.logging import LoggingCollectorMixin
if TYPE_CHECKING:
    from nanotron.config import NanotronConfigs
    from nanotron.parallel.parameters import NanotronParameter

logger = logging.get_logger(__name__)


class NanotronModel(nn.Module, LoggingCollectorMixin, metaclass=ABCMeta):
    """Nanotron 模型的抽象基类，定义了所有模型必须遵循的接口和约定。
    
    build_model(model_config, parallel_context, ...)
    │
    ├── 1. 实例化模型类（如 LlamaForTraining）
    ├── 2. 调用 model.init_model_randomly() 初始化参数
    ├── 3. 绑定参数（tie_parameters）
    │   └── 例如：embedding 和 lm_head 权重共享
    ├── 4. 为绑定权重创建进程组
    ├── 5. 健全性检查（sanity_check）
    └── 6. 包装 DDP（如果 DP > 1）

    所有 Nanotron 中的模型（如 Llama、Qwen2 等）都必须继承此类。
    该基类提供了分布式训练所需的基础设施，包括：
        - 并行上下文管理
        - 绑定参数的正确命名
        - 流水线并行的 rank 信息
        - 模型初始化和检查的钩子方法

    子类必须实现：
        - init_model_randomly(): 随机初始化模型参数

    子类可选实现：
        - tie_custom_params(): 标记自定义绑定参数
        - get_embeddings_lm_head_tied_names(): 返回嵌入层和语言模型头的绑定参数名
        - 各种 sanity_check 钩子方法

    假设：
        - PipelineBlock 的定义顺序与前向传播的执行顺序一致

    Attributes:
        parallel_context (ParallelContext): 并行上下文，包含所有进程组信息。
        config (NanotronConfigs): 模型配置。
        module_id_to_prefix (dict[int, str]): 模块对象 ID 到名称前缀的映射，
            用于构建参数的完整名称。
        input_pp_rank (int): 输入 PP rank（模型第一个 PipelineBlock 所在的 rank）。
        output_pp_rank (int): 输出 PP rank（模型最后一个 PipelineBlock 所在的 rank）。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parallel_context: "ParallelContext"
        self.config: "NanotronConfigs"
        self.module_id_to_prefix: dict[int, str]

        self.input_pp_rank: int
        self.output_pp_rank: int

        self.module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in self.named_modules()}
        self.module_id_to_prefix[id(self)] = ""

    def get_named_params_with_correct_tied(self) -> Iterator[Tuple[str, "NanotronParameter"]]:
        """返回带有正确绑定参数名称的参数迭代器。

        对于绑定参数，使用其在根模块中的完整名称而非局部名称，
        确保绑定参数在不同位置使用相同的名称标识。

        Yields:
            Tuple[str, NanotronParameter]: (参数完整名称, 参数对象) 元组。
                - 绑定参数：名称从根模块开始，如 "model.layer.0.weight"
                - 普通参数：使用标准 named_parameters() 的名称
        """

        def params_gen():
            for name, param in self.named_parameters():
                if param.is_tied:
                    yield (
                        param.get_tied_info().get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=self.module_id_to_prefix
                        ),
                        param,
                    )
                else:
                    yield name, param

        yield from params_gen()

    @abstractmethod
    def init_model_randomly(self, config):
        """随机初始化模型参数。

        子类必须实现此方法，根据配置初始化所有模型参数。

        Args:
            config: 模型配置对象，包含初始化相关的超参数。
        """
        ...

    def tie_custom_params(self) -> None:
        """标记自定义绑定参数。

        子类可覆盖此方法来标记模型特有的绑定参数。
        例如在 MQA（Multi-Query Attention）中，需要将 KV 头标记为绑定参数。
        默认实现不做任何操作。
        """
        pass

    def get_embeddings_lm_head_tied_names(self) -> list[str]:
        """返回嵌入层和语言模型头之间绑定参数的完整名称列表。

        如果嵌入层和语言模型头的权重共享，返回包含两者参数名的列表；
        否则返回空列表。

        Returns:
            list[str]: 绑定参数的完整名称列表。
                例如 ["model.token_position_embeddings.pp_block.token_embedding.weight",
                       "model.lm_head.pp_block.weight"]
        """
        return []

    def before_tbi_sanity_checks(self) -> None:
        """TBI（Training Batch Iteration）前的健全性检查钩子。"""
        pass

    def after_tbi_sanity_checks(self) -> None:
        """TBI（Training Batch Iteration）后的健全性检查钩子。"""
        pass

    def before_optim_step_sanity_checks(self) -> None:
        """优化器步骤前的健全性检查钩子。"""
        pass

    def after_optim_step_sanity_checks(self) -> None:
        """优化器步骤后的健全性检查钩子。"""
        pass

    def log_modules(self, level: int = logging.DEBUG, group: Optional[ProcessGroup] = None, rank: int = 0):
        """记录模型中所有 PipelineBlock 的 PP rank 分配信息。

        Args:
            level (int): 日志级别。默认为 DEBUG。
            group (Optional[ProcessGroup]): 进程组。默认为 None。
            rank (int): 输出日志的 rank。默认为 0。
        """
        assert hasattr(self, "parallel_context"), "`NanotronModel` needs to have a `parallel_context` attribute"

        for name, module in self.named_modules():
            if not isinstance(module, PipelineBlock):
                continue
            log_rank(
                f"module_name: {name} | PP: {module.rank}/{self.parallel_context.pp_pg.size()}",
                logger=logger,
                level=level,
                group=group,
                rank=rank,
            )

    @property
    def named_modules_in_pp_rank(self) -> Dict[str, nn.Module]:
        """返回当前 PP rank 上的所有叶模块。

        返回当前进程负责的模块（排除 PipelineBlock 本身），
        用于获取当前设备上实际执行计算的模块。

        Returns:
            Dict[str, nn.Module]: 模块名称到模块对象的映射。
                不包含 PipelineBlock 包装器，仅包含实际计算模块。
        """

        def get_leaf_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
            """返回模块中所有叶模块（没有子模块的模块）。"""
            leaf_modules = []
            for n, m in module.named_modules():
                if not list(m.children()):
                    leaf_modules.append((n, m))
            return leaf_modules

        modules = get_leaf_modules(self)
        named_modules_in_current_pp_rank = {}
        for name, module in modules:
            if isinstance(module, PipelineBlock):
                continue
            named_modules_in_current_pp_rank[name] = module

        return named_modules_in_current_pp_rank


class DTypeInvariantTensor(torch.Tensor):
    """禁止修改数据类型的张量子类。

    DTypeInvariantTensor 继承自 torch.Tensor，但禁止任何改变数据类型的操作。
    这用于保护混合精度训练中的关键张量（如 FP8 缩放因子），
    防止自动类型转换破坏数值稳定性。

    注意：张量的数据和其他属性仍然可以修改，仅禁止 dtype 变更。
    同时禁止 detach() 操作，确保张量始终参与计算图。
    """

    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(cls, *args, **kwargs)
        return tensor

    def detach(self, *args, **kwargs):
        """禁止 detach 操作，确保张量始终参与计算图。"""
        raise RuntimeError("Cannot detach an DTypeInvariantTensor")

    def to(self, *args, **kwargs):
        """重写 to() 方法，禁止改变数据类型。

        允许设备转移等操作，但如果检测到 dtype 参数则抛出异常。
        """
        if "dtype" in kwargs or any(isinstance(arg, torch.dtype) for arg in args):
            raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")
        else:
            return super().to(*args, **kwargs)

    def type(self, *args, **kwargs):
        """禁止 type() 方法改变数据类型。"""
        raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")

    def float(self, *args, **kwargs):
        """禁止转换为 float32。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to float")

    def double(self, *args, **kwargs):
        """禁止转换为 float64。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to double")

    def half(self, *args, **kwargs):
        """禁止转换为 float16。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to half")

    def long(self, *args, **kwargs):
        """禁止转换为 int64。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to long")

    def int(self, *args, **kwargs):
        """禁止转换为 int32。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to int")

    def short(self, *args, **kwargs):
        """禁止转换为 int16。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to short")

    def char(self, *args, **kwargs):
        """禁止转换为 int8。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to char")

    def byte(self, *args, **kwargs):
        """禁止转换为 uint8。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to byte")

    def bool(self, *args, **kwargs):
        """禁止转换为 bool。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bool")

    def bfloat16(self, *args, **kwargs):
        """禁止转换为 bfloat16。"""
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bfloat16")


def build_model(
    model_builder: Callable[[], NanotronModel],
    parallel_context: ParallelContext,
    dtype: torch.dtype,
    target_pp_ranks: Optional[List[int]] = None,
    device: Optional[torch.device] = torch.device("cuda"),
) -> NanotronModel:
    """构建模型并为每个 PipelineBlock 分配 PP rank。

    该函数执行以下步骤：
        1. 调用 model_builder 构建模型
        2. 收集所有 PipelineBlock 及其计算成本
        3. 使用负载均衡算法将 PipelineBlock 分配到各 PP rank
        4. 在指定设备和数据类型上初始化参数

    负载均衡算法：
        根据每个 PipelineBlock 的计算成本（FLOPs），将块均匀分配到各 PP rank，
        使得每个 rank 的累计计算成本大致相等。

    Args:
        model_builder (Callable[[], NanotronModel]): 模型构建函数，返回未初始化的模型。
        parallel_context (ParallelContext): 并行上下文。
        dtype (torch.dtype): 参数初始化的数据类型。
        target_pp_ranks (Optional[List[int]], optional): 目标 PP rank 列表。
            默认为 None，表示使用所有 PP rank。
        device (Optional[torch.device], optional): 参数初始化的设备。
            默认为 "cuda"。

    Returns:
        NanotronModel: 构建完成的模型，所有 PipelineBlock 已分配 rank。
    """
    log_rank(
        "Building model", logger=logger, level=logging.INFO, rank=0, group=parallel_context.world_pg, is_separator=True
    )
    model: NanotronModel = model_builder()

    if target_pp_ranks is None:
        pp_size = parallel_context.pp_pg.size()
        target_pp_ranks = list(range(pp_size))
    else:
        pp_size = len(target_pp_ranks)

    log_rank("Setting PP block ranks...", logger=logger, level=logging.INFO, rank=0, group=parallel_context.world_pg)
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    with init_on_device_and_dtype(device=device, dtype=dtype):
        # 获取每个 PipelineBlock 的计算成本，用于负载均衡
        block_compute_costs = model.get_block_compute_costs()
        block_cumulative_costs = np.cumsum(
            [
                block_compute_costs[module.module_builder] if module.module_builder in block_compute_costs else 0
                for module in pipeline_blocks
            ]
        )

        # 计算每个 PP rank 的累计成本阈值，实现均匀分配
        thresholds = [block_cumulative_costs[-1] * ((rank + 1) / pp_size) for rank in range(pp_size)]
        assert thresholds[-1] >= block_cumulative_costs[-1]
        target_pp_rank_idx = 0
        for block, cumulative_cost in zip(pipeline_blocks, block_cumulative_costs):
            assert target_pp_rank_idx < pp_size
            block.build_and_set_rank(target_pp_ranks[target_pp_rank_idx])

            if cumulative_cost > thresholds[target_pp_rank_idx]:
                target_pp_rank_idx += 1

        model.input_pp_rank = target_pp_ranks[0]
        model.output_pp_rank = target_pp_ranks[target_pp_rank_idx]

    if pp_size > 1:
        model.log_modules(level=logging.INFO, group=parallel_context.world_pg, rank=0)
    return model


@contextmanager
def ignore_init_on_device_and_dtype():
    """临时禁用 init_on_device_and_dtype 的设备和数据类型强制设置的上下文管理器。

    在 init_on_device_and_dtype 上下文内部使用此管理器，可以保护特定参数
    不被强制转换为目标数据类型。

    Example:
        with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.float32):
            with ignore_init_on_device_and_dtype():
                # 此参数将保持指定的 dtype (float32)，不被覆盖
                self.weight = nn.Parameter(torch.randn(..., dtype=torch.float32))
    """
    if not hasattr(ignore_init_on_device_and_dtype, "_ignore_flag"):
        ignore_init_on_device_and_dtype._ignore_flag = threading.local()

    old_value = getattr(ignore_init_on_device_and_dtype._ignore_flag, "value", False)
    ignore_init_on_device_and_dtype._ignore_flag.value = True

    try:
        yield
    finally:
        ignore_init_on_device_and_dtype._ignore_flag.value = old_value


@contextmanager
def init_on_device_and_dtype(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
):
    """控制参数初始化设备和数据类型的上下文管理器。

    在此上下文中创建的所有模块参数和缓冲区都会被自动转移到指定的
    设备和数据类型。同时，torch.empty/zeros/ones/full 等张量创建函数
    也会被补丁化，强制使用指定的设备和数据类型。

    这确保了模型初始化时不会在 CPU 上分配大量临时内存，
    直接在目标设备上创建参数。

    Args:
        device (torch.device, optional): 目标设备。默认为 "cpu"。
        dtype (torch.dtype, optional): 目标数据类型。默认为 torch.float。

    Note:
        使用 ignore_init_on_device_and_dtype() 可以在内部临时禁用此行为。
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer

    def should_ignore_init_on_device_and_dtype():
        if not hasattr(ignore_init_on_device_and_dtype, "_ignore_flag"):
            return False
        return getattr(ignore_init_on_device_and_dtype._ignore_flag, "value", False)

    def register_empty_parameter(module, name, param):
        """补丁化的参数注册函数，自动转移设备和数据类型。"""
        old_register_parameter(module, name, param)
        if param is not None:
            if should_ignore_init_on_device_and_dtype():
                pass
            else:
                param.data = param.data.to(device, dtype)

    def register_empty_buffer(module, name, buffer, persistent=True):
        """补丁化的缓冲区注册函数，自动转移设备和数据类型。"""
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            if should_ignore_init_on_device_and_dtype():
                pass
            else:
                module._buffers[name] = module._buffers[name].to(device, dtype)

    tensor_constructors_to_patch = {
        torch_function_name: getattr(torch, torch_function_name)
        for torch_function_name in ["empty", "zeros", "ones", "full"]
    }

    def patch_tensor_constructor(fn):
        """补丁化张量创建函数，强制使用指定的设备和数据类型。"""
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            kwargs["dtype"] = dtype
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def check_model_has_grad(model: NanotronModel, parallel_context: "ParallelContext"):
    """检查当前 PP rank 的模型是否至少有一个需要梯度的参数。

    DDP 要求模型至少有一个可训练参数。如果当前 PP rank 的模型
    没有任何可训练参数，DDP 将无法正常工作。

    Args:
        model (NanotronModel): 待检查的模型。
        parallel_context (ParallelContext): 并行上下文。

    Returns:
        bool: 如果模型有可训练参数则返回 True。

    Raises:
        ValueError: 当模型没有任何可训练参数时。
    """
    for param in model.parameters():
        if param.requires_grad:
            return True
    raise ValueError(
        f"Can't use DDP because model in PP={dist.get_rank(parallel_context.pp_pg)} has no gradient. Consider increasing the number of layers of your model, or put a smaller PP size.\n"
        f"Model: {model}"
    )
