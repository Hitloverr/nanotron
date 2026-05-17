"""
绑定参数管理模块 —— Nanotron 分布式训练中的权重共享与梯度同步。

本模块负责管理模型中的绑定参数（Tied Parameters），即多个位置共享同一参数的情况。
绑定参数在分布式训练中需要特殊处理，以确保所有共享位置的参数值一致，
且梯度正确同步。

核心概念：
    - 绑定参数（Tied Parameter）：模型中多个位置共享同一参数。
      典型场景：
        1. Embedding 层和 LM Head 的权重共享（最常见的权重共享模式）
        2. 张量并行中跨 TP rank 复制的 LayerNorm 权重
        3. 流水线并行中跨 PP stage 共享的参数

    - 绑定参数的两种同步模式：
        1. 同设备绑定：多个位置在同一设备上，通过 Python 引用共享同一 Parameter 对象
        2. 跨设备绑定：多个位置在不同设备上，需要通过梯度归约（AllReduce）同步

    - reduce_op 的含义：
        - None: 不需要梯度归约（同设备内的权重共享，或 TP 中复制的 LayerNorm）
        - dist.ReduceOp.SUM: 需要对梯度求和归约（跨 TP 的绑定参数，如 REDUCE_SCATTER 模式下的 LayerNorm）

绑定参数的处理流程：
    1. tie_parameters(): 在模型构建阶段调用，将多个位置的参数绑定为同一对象
    2. create_pg_for_tied_weights(): 为每组绑定参数创建专用进程组
    3. sync_tied_weights_gradients(): 在反向传播后同步绑定参数的梯度

注意事项：
    - 绑定权重必须在同一 DP rank 内（同一模型副本内），不支持跨 DP 绑定
    - 绑定参数的梯度同步必须在 DDP 梯度同步之前完成
    - 序列化时需要正确标记绑定参数，避免重复保存
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.utils import get_parameter_and_parent_module

logger = logging.get_logger(__name__)

"""
# 绑定参数流程
tie_parameters(root_module, ties, parallel_context, reduce_op)
│
├── 1. 验证所有绑定参数在同一 DP rank
├── 2. 在同一设备上：替换为同一个 Parameter 对象
├── 3. 跨设备：标记为 tied，添加 TiedInfo 元数据
└── 4. 创建专门的进程组用于梯度同步

# 梯度同步
sync_tied_weights_gradients(model, parallel_context)
│
└── 对每个 tied 参数，在绑定 rank 间 AllReduce 梯度

"""

def create_tied_parameter(
    parameter: nn.Parameter,
    name: str,
    global_ranks: Tuple[int, ...],
    reduce_op: Optional[dist.ReduceOp],
    root_module: nn.Module,
) -> NanotronParameter:
    """将普通参数转换为绑定参数，附加绑定元数据。

    Args:
        parameter (nn.Parameter): 待绑定的参数。如果不是 NanotronParameter，
            会自动转换。
        name (str): 参数在 root_module 中的相对名称。
        global_ranks (Tuple[int, ...]): 涉及该绑定参数的所有全局 rank 编号。
        reduce_op (Optional[dist.ReduceOp]): 梯度归约操作。
            - None: 不需要归约（同设备内的权重共享）
            - dist.ReduceOp.SUM: 梯度求和归约（跨设备的绑定参数）
        root_module (nn.Module): 参数所属的根模块。

    Returns:
        NanotronParameter: 带有绑定元数据的参数。
    """
    if not isinstance(parameter, NanotronParameter):
        parameter = NanotronParameter(tensor=parameter)
    parameter.mark_as_tied(name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module)
    return parameter


def tie_parameters(
    root_module: nn.Module,
    ties: List[Tuple[str, Tuple[int, ...]]],
    parallel_context: ParallelContext,
    reduce_op: Optional[dist.ReduceOp],
):
    """将模型中多个位置的参数绑定为同一对象。

    该方法实现了两种级别的绑定：
        1. 同设备绑定：如果多个绑定位置在同一设备上，直接替换为同一 Parameter 对象
        2. 跨设备绑定：如果绑定位置在不同设备上，为每个参数添加绑定元数据，
           后续通过梯度归约同步

    算法流程：
        1. 验证所有绑定位置都在同一 DP rank 内（同一模型副本）
        2. 合并所有绑定涉及的 global_ranks
        3. 遍历每个绑定位置：
           - 如果当前 rank 在该绑定的 ranks 中，获取对应参数
           - 第一个遇到的参数创建绑定 NanotronParameter
           - 后续遇到的参数替换为同一个 NanotronParameter（同设备绑定）

    Args:
        root_module (nn.Module): 根模块（通常是完整模型）。
        ties (List[Tuple[str, Tuple[int, ...]]]): 绑定列表，每个元素为
            (参数路径, 涉及的全局 rank 元组)。参数路径是相对于 root_module 的名称。
        parallel_context (ParallelContext): 并行上下文，用于获取 rank 映射。
        reduce_op (Optional[dist.ReduceOp]): 梯度归约操作。

    Raises:
        ValueError: 当绑定列表为空时。
        AssertionError: 当绑定位置跨越不同 DP rank 时。

    约束条件：
        - 绑定权重必须在同一 DP rank 内。这是因为不同 DP rank 持有的是
          模型的不同副本，它们的参数值在训练过程中可能不同（由于不同的数据），
          因此不应该绑定。
    """
    if len(ties) < 1:
        raise ValueError("Can't tie nothing")

    # 验证所有绑定位置都在同一 DP rank 内
    # 这是绑定权重的基本约束：只有同一模型副本内的参数才能绑定
    dp_ranks = tuple(
        sorted(
            {
                parallel_context.get_local_ranks(world_rank=global_rank)["dp"]
                for _, global_ranks in ties
                for global_rank in global_ranks
            }
        )
    )
    assert (
        len(dp_ranks) == 1
    ), f"Tying weights has to happen with a replica of a model. Got the ranks from the following replicas: {dp_ranks}"

    # 使用第一个绑定的参数名作为绑定组的名称
    name = ties[0][0]
    # 合并所有绑定涉及的 global_ranks（去重排序）
    global_ranks = tuple(sorted(set().union(*(tie[1] for tie in ties))))

    new_param = None
    world_rank = dist.get_rank(parallel_context.world_pg)
    for tie_target, tie_model_ranks in ties:
        # 跳过当前 rank 不参与的绑定
        if world_rank not in tie_model_ranks:
            continue

        # 获取绑定目标参数及其父模块
        param, parent_module, param_name = get_parameter_and_parent_module(target=tie_target, root_module=root_module)

        # 同设备绑定：第一个遇到的参数创建绑定 NanotronParameter
        # 后续遇到的参数直接引用同一个对象，实现物理共享
        if new_param is None:
            new_param = create_tied_parameter(
                parameter=param, name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module
            )

        # 将绑定参数设置到父模块中，替换原始参数
        setattr(parent_module, param_name, new_param)


def create_pg_for_tied_weights(root_module: nn.Module, parallel_context: ParallelContext):
    """为每组绑定权重创建专用的进程组。

    绑定权重可能涉及不同的 rank 集合，每组需要独立的 ProcessGroup
    来进行梯度归约通信。该方法收集所有绑定权重的 rank 集合，
    为每个唯一的 rank 集合创建一个 ProcessGroup。

    算法流程：
        1. 收集当前 rank 上所有绑定参数的 global_ranks 集合
        2. 通过 all_gather_object 收集所有 rank 上的绑定信息
        3. 为每个唯一的 rank 集合创建 ProcessGroup（如果尚未创建）

    Args:
        root_module (nn.Module): 根模块，包含绑定参数。
        parallel_context (ParallelContext): 并行上下文，提供 world_ranks_to_pg 缓存。

    Note:
        - 使用 parallel_context.world_ranks_to_pg 缓存避免重复创建
        - 绑定权重的进程组与 TP/DP/PP 进程组不同，是按需创建的
    """
    group_ranks = {
        param.get_tied_info().global_ranks
        for name, param in root_module.named_parameters()
        if isinstance(param, NanotronParameter) and param.is_tied
    }

    # 收集所有 rank 上的绑定信息
    world_group_ranks = [None] * parallel_context.world_pg.size()
    dist.all_gather_object(world_group_ranks, group_ranks, group=parallel_context.world_pg)
    # 合并去重所有绑定权重的 rank 集合
    all_group_ranks = sorted(
        set().union(*world_group_ranks),
    )

    # 为每个唯一的 rank 集合创建 ProcessGroup
    for global_ranks in all_group_ranks:
        if global_ranks not in parallel_context.world_ranks_to_pg:
            parallel_context.world_ranks_to_pg[global_ranks] = dist.new_group(global_ranks)


def get_tied_id_to_param(
    parameters: List[NanotronParameter], root_module: nn.Module
) -> Dict[Tuple[str, Tuple[int, ...]], NanotronParameter]:
    """构建绑定参数的 ID 到参数对象的映射。

    绑定参数的唯一标识为 (完整参数名, global_ranks) 元组。
    该映射用于在梯度同步时快速查找绑定参数。

    Args:
        parameters (List[NanotronParameter]): 模型中所有需要梯度的参数列表。
        root_module (nn.Module): 根模块，用于构建参数的完整名称。

    Returns:
        Dict[Tuple[str, Tuple[int, ...]], NanotronParameter]:
            键为 (完整参数名, global_ranks)，值为对应的 NanotronParameter。

    Note:
        - 完整参数名通过模块 ID 到前缀的映射构建
        - 同一绑定组的参数在映射中只出现一次（因为它们是同一对象）
    """
    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in root_module.named_modules()}
    module_id_to_prefix[id(root_module)] = ""
    return {
        (
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix),
            param.get_tied_info().global_ranks,  # TODO @nouamane: merge groups which tie the same parameter
        ): param
        for param in parameters
        if param.is_tied
    }


@torch.profiler.record_function("sync_tied_weights_gradients")
def sync_tied_weights_gradients(
    module: nn.Module,  # TODO: NanotronModel
    parallel_context: ParallelContext,
    grad_accumulator: Optional[GradientAccumulator],
):
    """同步绑定权重的梯度。

    在反向传播完成后调用，对绑定权重的梯度进行归约同步。
    不同绑定组可能使用不同的归约操作和进程组，因此需要分组处理。

    算法流程：
        1. 构建绑定参数 ID 到参数对象的映射
        2. 过滤掉不需要归约的绑定参数（reduce_op is None）
        3. 按 (global_ranks, reduce_op) 分组，将相同组和操作的梯度聚合
        4. 对每组使用 all_reduce_coalesced 批量归约，减少通信次数

    性能优化：
        - 使用 OrderedDict 保证所有 rank 上的处理顺序一致
        - 使用 all_reduce_coalesced 将同一组的多个张量合并为一次通信
        - 按 (name, group_ranks) 排序确保跨 rank 一致性

    Args:
        module (nn.Module): 模型对象。
        parallel_context (ParallelContext): 并行上下文，提供进程组映射。
        grad_accumulator (Optional[GradientAccumulator]): 梯度累积器，
            如果提供则从梯度缓冲区获取梯度，否则直接使用 param.grad。

    Note:
        - 必须在 DDP 梯度同步之前调用，否则绑定参数的梯度可能不完整
        - reduce_op 为 None 的绑定参数跳过归约（如 TP 中复制的 LayerNorm）
    """
    tied_id_to_param = get_tied_id_to_param(
        parameters=[param for param in module.parameters() if param.requires_grad], root_module=module
    )

    for rank in [0, parallel_context.world_pg.size() - 1]:
        log_rank(
            f"[Debug Tied Weights] Syncing the following tied weights: {tied_id_to_param.keys()}",
            logger=logger,
            level=logging.DEBUG,
            group=parallel_context.world_pg,
            rank=rank,
        )

    # 按 (进程组, 归约操作) 分组，将同一组的梯度聚合以批量归约
    # 使用 OrderedDict 保证所有 rank 上的处理顺序一致，避免死锁
    group_ranks_and_reduce_op_to_tensors_to_reduce = OrderedDict()
    for (name, group_ranks), tied_param in sorted(tied_id_to_param.items(), key=lambda x: x[0]):
        tied_info = tied_param.get_tied_info()
        # reduce_op 为 None 的绑定参数不需要归约
        # 典型场景：同设备内的权重共享，或 TP ALL_REDUCE 模式下的 LayerNorm
        if tied_info.reduce_op is None:
            continue

        # 获取梯度：优先从梯度累积器获取，否则直接使用 param.grad
        if grad_accumulator is not None:
            tied_grad = grad_accumulator.get_grad_buffer(name=name)
        else:
            tied_grad = tied_param.grad
        log_rank(
            f"Syncing tied weights {name} across ranks {group_ranks} ...",
            logger=logger,
            level=logging.DEBUG,
            group=parallel_context.world_ranks_to_pg[group_ranks],
            rank=0,
        )
        # 按 (进程组, 归约操作) 分组
        key = (group_ranks, tied_info.reduce_op)
        if key in group_ranks_and_reduce_op_to_tensors_to_reduce:
            group_ranks_and_reduce_op_to_tensors_to_reduce[(group_ranks, tied_info.reduce_op)].append(tied_grad)
        else:
            group_ranks_and_reduce_op_to_tensors_to_reduce[(group_ranks, tied_info.reduce_op)] = [tied_grad]

    # 批量归约：每组使用 all_reduce_coalesced 一次通信
    for (group_ranks, reduce_op), tensors in group_ranks_and_reduce_op_to_tensors_to_reduce.items():
        dist.all_reduce_coalesced(tensors=tensors, op=reduce_op, group=parallel_context.world_ranks_to_pg[group_ranks])
