"""
张量并行通信模式枚举 —— 定义张量并行线性层的通信策略。

本模块定义了 TensorParallelLinearMode 枚举，用于指定张量并行线性层
在前向/反向传播中使用的集合通信操作。

两种通信模式：

    ALL_REDUCE（标准张量并行）：
        前向传播：
            - ColumnLinear: 输入通过 identity（占位操作），输出不需要通信
            - RowLinear: 输出通过 AllReduce 求和聚合
        反向传播：
            - ColumnLinear: 输入梯度通过 AllReduce 求和
            - RowLinear: 输入梯度不需要额外通信
        通信量：每个 Transformer 层 2 次 AllReduce

    REDUCE_SCATTER（序列并行优化）：
        前向传播：
            - ColumnLinear: 输入通过 AllGather 聚合，输出通过 ReduceScatter 分散
            - RowLinear: 输出通过 ReduceScatter 分散
        反向传播：
            - ColumnLinear: 输入梯度通过 ReduceScatter 分散
            - RowLinear: 输入梯度通过 AllGather 聚合
        通信量：每个 Transformer 层 1 次 AllGather + 1 次 ReduceScatter
        优势：与标准模式相比，省去了一次 AllReduce，降低通信开销

    典型组合：
        ColumnLinear(ALL_REDUCE) + RowLinear(ALL_REDUCE):
            标准张量并行，2 次 AllReduce/层
        ColumnLinear(REDUCE_SCATTER) + RowLinear(REDUCE_SCATTER):
            序列并行，1 次 AllGather + 1 次 ReduceScatter/层

参考：Megatron-LM 序列并行论文
"""

from enum import Enum, auto


class TensorParallelLinearMode(Enum):
    """张量并行线性层的通信模式。

    Attributes:
        ALL_REDUCE: 标准张量并行模式，使用 AllReduce 聚合部分结果。
        REDUCE_SCATTER: 序列并行模式，使用 ReduceScatter 分散结果，
            与 AllGather 配合使用，减少通信次数。
            
    模式一：ALL_REDUCE（经典 TP）
    ┌──────────┐     ┌──────────┐
    │  GPU 0   │     │  GPU 1   │
    │ W[0:N/2] │     │W[N/2:N]  │
    └────┬─────┘     └────┬─────┘
        │ AllReduce       │
        └────────┬────────┘
                ▼
            完整输出

    模式二：REDUCE_SCATTER（序列并行）
    ┌──────────┐     ┌──────────┐
    │  GPU 0   │     │  GPU 1   │
    │ W[0:N/2] │     │W[N/2:N]  │
    └────┬─────┘     └────┬─────┘
        │ ReduceScatter   │
        ▼                 ▼
    输出[0:S/2]      输出[S/2:S]
    """

    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()

    def __format__(self, format_spec):
        return self.name

    def __str__(self):
        return self.name
