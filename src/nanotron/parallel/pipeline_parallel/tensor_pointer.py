"""
张量指针模块 —— 流水线并行中跨 rank 张量引用的占位符。

TensorPointer 是流水线并行中的核心数据结构，用于表示"某个张量存在于
另一个 PP rank 上"这一语义。当数据不在当前 rank 时，使用 TensorPointer
作为占位符，避免实际的数据传输直到真正需要时。

工作原理：
    在流水线并行中，模型被切分到多个 PP rank 上。当前向传播经过一个
    PipelineBlock 时，如果当前 rank 不是该 block 的 compute rank：
        - 非 compute rank: 将实际张量发送到 compute rank，返回 TensorPointer
        - compute rank: 从 TensorPointer 指向的 rank 接收实际张量

    TensorPointer 只记录目标 rank 的编号，不持有任何实际数据。
    它与 P2P 通信配合使用，实现按需的数据传输。

与 PipelineBlock 的关系：
    PipelineBlock.forward() 中，非 compute rank 返回 TensorPointer 字典，
    compute rank 根据 TensorPointer 的 group_rank 从对应 rank 接收数据。

设计考量：
    - TensorPointer 是轻量级的，只包含一个 int 字段
    - 未来可能扩展：添加进程组标识、通信标签等元数据
"""

import dataclasses


@dataclasses.dataclass
class TensorPointer:
    """张量指针，指示需要从哪个 rank 获取张量数据。

    在流水线并行中，当数据不在当前 rank 时，使用 TensorPointer
    作为占位符，表示该张量存在于 group_rank 指定的 rank 上。

    Attributes:
        group_rank (int): 持有实际张量数据的 rank 在 PP 进程组中的编号。

    示例：
        在 PP=4 的设置中，Rank 0 上的输入数据需要发送到 Rank 1：
        - Rank 0: 返回 TensorPointer(group_rank=1)
        - Rank 1: 根据 TensorPointer 从 Rank 0 接收实际张量
    """

    group_rank: int
