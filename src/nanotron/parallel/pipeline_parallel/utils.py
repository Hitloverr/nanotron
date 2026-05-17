"""
流水线并行工具函数 —— PP rank 查询与模型结构分析。

本模块提供流水线并行中的辅助工具函数，用于查询模型中
各模块/参数所在的 PP rank，以及获取流水线的输入/输出 rank。

核心功能：
    - get_input_output_pp_ranks: 获取模型的流水线输入和输出 rank
    - get_pp_rank_of: 查询指定模块/参数所在的 PP rank

与流水线并行的关系：
    在 PP 训练中，模型被切分到多个 stage，每个 stage 对应一个 PP rank。
    知道模块所在的 PP rank 对于以下场景很重要：
        - 确定哪些 rank 负责数据加载（input_pp_rank）
        - 确定哪些 rank 负责损失计算（output_pp_rank）
        - 绑定参数时确定参数的 PP rank（避免跨 PP 绑定）
"""

from nanotron.models import NanotronModel
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def get_input_output_pp_ranks(model: NanotronModel | DistributedDataParallel):
    """获取模型的流水线输入和输出 PP rank。

    输入 PP rank 负责接收训练数据（通常是第一个 stage），
    输出 PP rank 负责计算损失（通常是最后一个 stage）。

    在 PP 训练中，只有 input_pp_rank 需要 dataloader，
    只有 output_pp_rank 需要 loss 函数。

    Args:
        model (NanotronModel | DistributedDataParallel): 模型对象，
            可能被 DDP 包装。

    Returns:
        Tuple[int, int]: (input_pp_rank, output_pp_rank)
    """
    if isinstance(model, DistributedDataParallel):
        input_pp_rank = model.module.input_pp_rank
        output_pp_rank = model.module.output_pp_rank
    else:
        input_pp_rank = model.input_pp_rank
        output_pp_rank = model.output_pp_rank
    return input_pp_rank, output_pp_rank


def get_pp_rank_of(target: str, module: nn.Module):
    """查询指定名称的模块/参数所在的 PP rank。

    通过递归遍历模块路径，找到包含目标模块的 PipelineBlock，
    返回该 PipelineBlock 的 PP rank。

    算法流程：
        1. 如果 module 本身是 PipelineBlock，直接返回其 rank
        2. 沿 target 路径逐级遍历子模块
        3. 如果遇到 PipelineBlock，返回其 rank
        4. 如果遍历完路径仍未找到 PipelineBlock，抛出异常

    Args:
        target (str): 模块/参数的路径名称（如 "encoder.layer.0.attn"）。
        module (nn.Module): 根模块。

    Returns:
        int: 目标所在的 PP rank。

    Raises:
        AttributeError: 当路径中的属性不存在或不是 nn.Module 时。
        ValueError: 当目标不在任何 PipelineBlock 内时。

    使用场景：
        - 绑定参数时验证参数在同一 PP rank 上
        - 调试时定位模块所在的 stage
    """
    if isinstance(module, PipelineBlock):
        return module.rank

    atoms = target.split(".")
    current_module = module
    for atom in atoms:
        if not hasattr(current_module, atom):
            raise AttributeError(f'{current_module._get_name()} has no attribute `"{atom}"`')

        current_module = getattr(current_module, atom)

        if isinstance(current_module, PipelineBlock):
            return current_module.rank

        if not isinstance(current_module, nn.Module):
            raise AttributeError(f'`"{atom}"` is not an nn.Module')

    raise ValueError(f'`"{target}" is not inside a PipelineBlock and thus does not have a pp_rank')
