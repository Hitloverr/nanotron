"""
并行化配置模块 —— 定义分布式训练的并行策略参数。

本模块定义了 ParallelismArgs 数据类，用于配置 Nanotron 的并行策略组合。
Nanotron 支持多种并行维度的组合，包括 DP、PP、TP、EP、CP。

并行策略概述：
    - DP（数据并行，Data Parallelism）：
        将训练数据分割到多个 GPU，每个 GPU 持有模型副本，
        通过梯度 AllReduce 同步参数更新。dp 表示数据并行度。

    - PP（流水线并行，Pipeline Parallelism）：
        将模型按层切分到多个 GPU，每个 GPU（stage）处理部分层。
        微批次数据依次通过各 stage，形成流水线。pp 表示流水线 stage 数。

    - TP（张量并行，Tensor Parallelism）：
        将单个算子（如线性层）的权重切分到多个 GPU，
        通过集合通信（AllReduce/ReduceScatter）聚合部分结果。tp 表示张量并行度。

    - EP（专家并行，Expert Parallelism）：
        MoE 模型中将不同专家分配到不同 GPU，每个 GPU 处理部分专家。
        expert_parallel_size 表示专家并行度。

    - CP（上下文并行，Context Parallelism）：
        将长序列沿序列维度切分到多个 GPU，使用 Ring Attention 等算法
        计算跨设备的注意力。context_parallel_size 表示上下文并行度。

并行度约束：
    总 GPU 数 = dp × pp × tp × expert_parallel_size × context_parallel_size

    典型配置示例：
        - 8 GPU, TP=2, PP=2, DP=2: 2×2×2=8
        - 16 GPU, TP=2, PP=4, DP=2: 2×4×2=16
        - MoE 模型 16 GPU, TP=2, EP=4, DP=2: 2×4×2=16
"""

from dataclasses import dataclass
from typing import Optional

from nanotron.config.utils_config import (
    cast_str_to_pipeline_engine,
)
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode


@dataclass
class ParallelismArgs:
    """分布式训练并行策略配置参数。

    定义了 Nanotron 中所有并行维度的配置，包括并行度、通信模式和优化选项。

    Attributes:
        dp (int): 数据并行度。每个模型副本的数据被分割到 dp 个 GPU 上。
            增大 dp 可以训练更大的 batch size，但通信量随 dp 线性增长。

        pp (int): 流水线并行度（stage 数）。模型按层切分到 pp 个 GPU 上。
            增大 pp 可以训练更深的模型，但会引入流水线气泡。

        tp (int): 张量并行度。单个算子的权重被切分到 tp 个 GPU 上。
            增大 tp 可以训练更宽的模型，但通信量随 tp 增长。
            tp 通常应等于单个节点内的 GPU 数，以利用 NVLink 的高带宽。

        pp_engine (Optional[PipelineEngine]): 流水线调度引擎。
            - "1f1b"（OneForwardOneBackward）: 交替执行前向和反向传播，
              减少显存占用，是常用的调度策略
            - "afab"（AllForwardAllBackward）: 先执行所有前向传播，
              再执行所有反向传播，显存占用较高但实现简单
            默认为 "afab"。

        tp_mode (Optional[TensorParallelLinearMode]): 张量并行的通信模式。
            - ALL_REDUCE: 标准张量并行，每层 2 次 AllReduce
            - REDUCE_SCATTER: 序列并行模式，每层 1 次 AllGather + 1 次 ReduceScatter，
              通信量更少，适合序列较长时使用
            默认为 ALL_REDUCE。

        tp_linear_async_communication (Optional[bool]): 是否在 TP 线性层中使用异步通信。
            启用后，通信操作与计算重叠执行，可以隐藏通信延迟。
            需要硬件支持异步通信（如 NVLink + NCCL）。
            默认为 False。

        recompute_layer (bool): 是否对每个 Transformer 层使用梯度重计算。
            启用后，前向传播的激活值不保存，反向传播时重新计算，
            以时间换空间，减少约 60% 的激活值显存占用。
            默认为 False。

        tp_recompute_allgather (bool): 是否在 TP 重计算时重新执行 AllGather。
            在 REDUCE_SCATTER 模式下，前向传播的 AllGather 结果可以缓存或重计算。
            启用后选择重计算，节省显存但增加通信。
            默认为 True。

        expert_parallel_size (int): 专家并行度（仅用于 MoE 模型）。
            MoE 模型中的专家被分配到 expert_parallel_size 个 GPU 上。
            通常 expert_parallel_size × tp = 总专家数。
            默认为 1（不使用专家并行）。

        context_parallel_size (int): 上下文并行度（用于长序列训练）。
            将序列沿长度维度切分到 context_parallel_size 个 GPU 上，
            使用 Ring Attention 算法计算跨设备的注意力。
            默认为 1（不使用上下文并行）。
    """

    dp: int
    pp: int
    tp: int
    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    tp_linear_async_communication: Optional[bool] = None
    recompute_layer: bool = False
    tp_recompute_allgather: bool = True

    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    def __post_init__(self):
        """初始化后的默认值设置和类型转换。

        保守默认值策略：
            - pp_engine 默认使用 "afab"（AllForwardAllBackward），
              因为它实现简单且在微批次数较少时性能可接受
            - tp_mode 默认使用 ALL_REDUCE，因为它是标准模式，
              REDUCE_SCATTER 需要序列并行支持
            - tp_linear_async_communication 默认关闭，
              因为异步通信需要硬件支持且调试困难
        """
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        # 支持从字符串配置转换，方便 YAML/JSON 配置文件使用
        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]
