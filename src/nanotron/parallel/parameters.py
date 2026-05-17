"""
参数管理模块 —— Nanotron 分布式参数元数据系统。

本模块定义了 Nanotron 中参数管理的核心数据结构和工具类，用于支持：
    - 分片参数（Sharded Parameters）：参数在多个设备间切分，每个设备只持有部分数据
    - 绑定参数（Tied Parameters）：多个位置共享同一参数，需要梯度同步

核心类：
    - SlicesPair: 描述分片参数的本地切片与全局切片的映射关系
    - TiedInfo: 描述绑定参数的元数据（名称、所属模块、全局 rank、归约操作）
    - ShardedInfo: 描述分片参数的元数据（全局 rank、切片映射、未分片形状）
    - NanotronParameter: 扩展 nn.Parameter，支持分片和绑定元数据
"""

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging

if TYPE_CHECKING:
    from nanotron.models import NanotronModel

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class SlicesPair:
    """分片参数的本地切片与全局切片的映射对。

    在张量并行中，一个完整参数被切分到多个设备上，每个设备持有不同的切片。
    SlicesPair 记录了本地设备上的切片（local_slices）和该切片在完整参数中
    对应的位置（global_slices），用于序列化和反序列化时重建完整参数。

    Attributes:
        local_slices (Tuple[slice, ...]): 本地分片在各维度上的切片描述。
            例如 (slice(0, 512), slice(None)) 表示本地数据的第 0 维
            取 [0:512]，第 1 维取全部。
        global_slices (Tuple[slice, ...]): 本地分片在完整参数中对应的切片描述。
            例如 (slice(0, 512), slice(None)) 表示在完整参数中对应
            第 0 维的 [0:512] 部分。

    序列化格式：
        单个 SlicesPair: "start,stop,step|start,stop,step#start,stop,step|start,stop,step"
        多个 SlicesPair: 用 ";" 分隔

    Example:
        TP=2 时，一个形状为 [1024, 4096] 的权重矩阵在第 0 维切分：
        - Rank 0: SlicesPair(local_slices=(slice(0,512), slice(None)),
                            global_slices=(slice(0,512), slice(None)))
        - Rank 1: SlicesPair(local_slices=(slice(0,512), slice(None)),
                            global_slices=(slice(512,1024), slice(None)))
    """

    local_slices: Tuple[slice, ...]
    global_slices: Tuple[slice, ...]

    @staticmethod
    def slice_to_str(s: slice):
        """将 slice 对象转换为字符串表示。

        Args:
            s (slice): Python slice 对象。

        Returns:
            str: 切片的字符串表示，格式为 "start,stop,step"。
                None 值保留为 "None"。

        Example:
            slice(0, 10, 2) → "0,10,2"
            slice(None, None, None) → "None,None,None"
        """
        return ",".join(str(x) if x is not None else "None" for x in (s.start, s.stop, s.step))

    @staticmethod
    def str_to_slice(s: str):
        """将字符串表示还原为 slice 对象。

        Args:
            s (str): 切片的字符串表示，格式为 "start,stop,step"。

        Returns:
            slice: 还原后的 Python slice 对象。

        Example:
            "0,10,2" → slice(0, 10, 2)
            "None,None,None" → slice(None, None, None)
        """
        return slice(*(int(x) if x != "None" else None for x in s.split(",")))

    def __str__(self):
        """将 SlicesPair 序列化为字符串。

        Returns:
            str: 格式为 "local_dim0|local_dim1#global_dim0|global_dim1"，
                各维度用 "|" 分隔，本地和全局用 "#" 分隔。
        """
        local_slices_str = "|".join(map(self.slice_to_str, self.local_slices))
        global_slices_str = "|".join(map(self.slice_to_str, self.global_slices))
        return f"{local_slices_str}#{global_slices_str}"

    @classmethod
    def from_str(cls, string: str):
        """从字符串反序列化 SlicesPair。

        Args:
            string (str): 序列化后的字符串，格式为 "local#global"。

        Returns:
            SlicesPair: 反序列化后的 SlicesPair 对象。
        """
        local_slices_str, global_slices_str = string.split("#")
        local_slices = tuple(map(cls.str_to_slice, local_slices_str.split("|")))
        global_slices = tuple(map(cls.str_to_slice, global_slices_str.split("|")))
        return cls(local_slices, global_slices)

    @classmethod
    def tuple_to_str(cls, pairs):
        """将多个 SlicesPair 序列化为单个字符串。

        Args:
            pairs: SlicesPair 的元组或列表。

        Returns:
            str: 各 SlicesPair 用 ";" 分隔的字符串。

        Example:
            2 个 SlicesPair → "local0#global0;local1#global1"
        """
        return ";".join(map(str, pairs))

    @classmethod
    def tuple_from_str(cls, string: str):
        """从字符串反序列化多个 SlicesPair。

        Args:
            string (str): 以 ";" 分隔的多个 SlicesPair 字符串。

        Returns:
            Tuple[SlicesPair, ...]: 反序列化后的 SlicesPair 元组。
        """
        return tuple(map(cls.from_str, string.split(";")))


@dataclasses.dataclass
class TiedInfo:
    """绑定参数的元数据信息。

    绑定参数（Tied Parameter）是指模型中多个位置共享同一参数的情况，
    例如 Embedding 层和 LM Head 的权重共享。TiedInfo 记录了绑定参数的
    名称、所属模块、涉及的全局 rank 以及梯度归约操作。

    Attributes:
        name (str): 参数在 root_module 中的相对名称，
            例如 "dense0.dense1.weight"。
        root_module (nn.Module): 参数所属的根模块，用于定位参数的完整路径。
        global_ranks (Tuple[int, ...]): 涉及该绑定参数的所有全局 rank 编号。
            同一绑定参数可能分布在多个 rank 上（例如跨 TP 的 LayerNorm）。
        reduce_op (Optional[dist.ReduceOp]): 梯度归约操作。
            - None: 不需要归约（如同设备内的权重共享）
            - dist.ReduceOp.SUM: 梯度求和归约（如跨 TP 的绑定参数）
    """

    name: str
    root_module: nn.Module
    global_ranks: Tuple[int, ...]
    reduce_op: Optional[dist.ReduceOp]

    def get_full_name_from_model(self, model: nn.Module) -> str:
        """从模型对象获取绑定参数的完整名称。

        Args:
            model (nn.Module): 模型对象，用于构建模块 ID 到前缀的映射。

        Returns:
            str: 参数的完整名称，例如 "model.layer.0.weight"。
        """
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        module_id_to_prefix[id(model)] = ""
        return self.get_full_name_from_module_id_to_prefix(module_id_to_prefix)

    def get_full_name_from_module_id_to_prefix(self, module_id_to_prefix: Dict[int, str]) -> str:
        """根据模块 ID 到前缀的映射获取参数的完整名称。

        Args:
            module_id_to_prefix (Dict[int, str]): 模块对象 ID 到名称前缀的映射。

        Returns:
            str: 参数的完整名称，格式为 "前缀.参数名"。
        """
        return f"{module_id_to_prefix[id(self.root_module)]}{self.name}"


@dataclasses.dataclass
class ShardedInfo:
    """分片参数的元数据信息。

    分片参数（Sharded Parameter）是指被切分到多个设备上的参数，每个设备
    只持有完整参数的一个切片。ShardedInfo 记录了分片涉及的全局 rank、
    本地切片与全局切片的映射关系，以及完整参数的形状。

    Attributes:
        global_ranks (Tuple[int, ...]): 持有该参数分片的所有全局 rank 编号。
        local_global_slices_pairs (Tuple[SlicesPair, ...]): 本地切片与全局切片
            的映射对。描述本地持有的数据在完整参数中的位置。
        unsharded_shape (Tuple[int, ...]): 完整（未分片）参数的形状。
            用于序列化时重建完整参数。

    Example:
        TP=2 时，形状为 [4096, 4096] 的权重在第 0 维切分：
        - Rank 0: global_ranks=(0,1),
                  local_global_slices_pairs=(SlicesPair(local=(slice(0,2048),slice(None)),
                                                        global=(slice(0,2048),slice(None))),)
                  unsharded_shape=(4096, 4096)
    """

    global_ranks: Tuple[int, ...]
    local_global_slices_pairs: Tuple[SlicesPair, ...]
    unsharded_shape: Tuple[int, ...]

    def is_tp_sharded(self, parallel_context) -> bool:
        """判断该分片是否由张量并行产生。

        Args:
            parallel_context: 并行上下文对象，包含 TP 进程组信息。

        Returns:
            bool: 如果持有分片的 rank 集合包含当前 TP 进程组的所有 rank，
                则返回 True。
        """
        return set(dist.get_global_ranks(parallel_context.tp_pg)).issubset(set(self.global_ranks))

    def is_expert_sharded(self, parallel_context) -> bool:
        """判断该分片是否由专家并行产生。

        Args:
            parallel_context: 并行上下文对象，包含 EP 进程组信息。

        Returns:
            bool: 如果持有分片的 rank 集合包含当前 EP 进程组的所有 rank，
                则返回 True。
        """
        return set(dist.get_global_ranks(parallel_context.ep_pg)).issubset(set(self.global_ranks))

    def is_dp_sharded(self, parallel_context):
        """判断该分片是否由数据并行产生（如 ZeRO 优化器分片）。

        Args:
            parallel_context: 并行上下文对象，包含 DP 进程组信息。

        Returns:
            bool: 如果持有分片的 rank 集合包含当前 DP 进程组的所有 rank，
                则返回 True。
        """
        return set(dist.get_global_ranks(parallel_context.dp_pg)).issubset(set(self.global_ranks))


class NanotronParameter(nn.Parameter):
    """Nanotron 模型中所有参数的基类，扩展了 nn.Parameter 以支持分布式元数据。

    NanotronParameter 在标准 Parameter 的基础上增加了两种分布式属性：
     - sharded（分片）: 参数在多个设备间切分，每个设备只持有部分数据。
       典型场景：张量并行中的权重矩阵切分。
     - tied（绑定）: 参数与其他位置的参数共享，需要梯度同步。
       典型场景：Embedding 和 LM Head 的权重共享。

    元数据存储机制：
        使用 __nanotron_metadata__ 字典存储所有元数据，避免与 PyTorch 的
        Parameter 内部属性冲突。该字典以 "tied" 和 "sharded" 为键，
        分别存储 TiedInfo 和 ShardedInfo 对象。

    关于绑定权重的注意事项：
        - 绑定权重意味着需要在同一 DP rank 内同步的权重，无论它们是否参与 TP 策略
          或只是两层之间的共享权重
        - 同步绑定权重通常需要对梯度求和
        - 某些权重不需要跨 rank 归约梯度：它们可能在同一设备上（如同一 PP 阶段的
          编码器/解码器嵌入），或者在 TP 间复制并分担计算量（如传统 TP 的 LayerNorm）
        - 即使某些权重不需要归约梯度，将它们标记为绑定仍然有用，例如当前序列化
          格式需要正确标记它们

    Note:
        NanotronParameter 继承自 nn.Parameter，因此可以无缝替换标准 Parameter，
        但需要通过 mark_as_sharded() 或 mark_as_tied() 方法添加分布式元数据。

    # 分片信息
    sharded_info: ShardedInfo
    │   ├── global_ranks: Tuple[int, ...]        # 持有分片的全局 rank
    │   ├── local_global_slices_pairs            # 本地/全局切片映射
    │   └── unsharded_shape: Tuple[int, ...]     # 未分片的完整形状

    # 绑定信息
    tied_info: TiedInfo
    │   ├── name: str                            # 参数名
    │   ├── root_module: nn.Module               # 所属模块
    │   ├── global_ranks: Tuple[int, ...]        # 绑定的全局 rank
    │   └── reduce_op: Optional[ReduceOp]        # 梯度归约操作
    """

    NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME = "__nanotron_metadata__"
    NANOTRON_PARAMETER_METADATA_TIED_KEY = "tied"
    NANOTRON_PARAMETER_METADATA_SHARDED_KEY = "sharded"

    def __new__(cls, tensor: torch.Tensor, requires_grad: bool = True):
        """创建 NanotronParameter 实例。

        Args:
            tensor (torch.Tensor): 底层张量数据。如果传入的已经是 NanotronParameter，
                则会复制其元数据。
            requires_grad (bool, optional): 是否需要计算梯度。默认为 True。

        Returns:
            NanotronParameter: 新创建的参数实例。

        Note:
            如果输入 tensor 已经是 NanotronParameter，会复制其元数据字典，
            但使用深拷贝以避免共享引用。
        """
        param = nn.Parameter.__new__(cls, data=tensor.data.detach(), requires_grad=requires_grad)

        if isinstance(tensor, NanotronParameter):
            assert type(tensor) == NanotronParameter
            setattr(
                param,
                cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME,
                getattr(tensor, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME).copy(),
            )
        else:
            setattr(param, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME, {})

        return param

    def _set_metadata(self, key: str, value: Any):
        """设置元数据键值对，不允许覆盖已有元数据。

        Args:
            key (str): 元数据键名，如 "tied" 或 "sharded"。
            value (Any): 元数据值，如 TiedInfo 或 ShardedInfo 对象。

        Raises:
            ValueError: 当尝试覆盖已存在的元数据键时抛出。
        """
        metadata = getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)

        if key in metadata:
            raise ValueError(
                f"We shouldn't override previous metadata. Key to be overridden: {key}, current metadata: {metadata}"
            )
        else:
            metadata[key] = value

    def mark_as_tied(
        self,
        name: str,
        global_ranks: Tuple[int, ...],
        reduce_op: Optional[dist.ReduceOp],
        root_module: "NanotronModel",
    ):
        """将参数标记为绑定参数。

        Args:
            name (str): 参数在 root_module 中的相对名称。
            global_ranks (Tuple[int, ...]): 涉及该绑定参数的所有全局 rank。
            reduce_op (Optional[dist.ReduceOp]): 梯度归约操作。
                - None: 不需要归约（同设备内的权重共享）
                - dist.ReduceOp.SUM: 梯度求和归约
            root_module (NanotronModel): 参数所属的根模块。
        """
        self._set_metadata(
            self.NANOTRON_PARAMETER_METADATA_TIED_KEY,
            TiedInfo(name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module),
        )

    def get_tied_info(self) -> TiedInfo:
        """获取绑定参数的元数据信息。

        Returns:
            TiedInfo: 绑定参数的元数据。

        Raises:
            KeyError: 当参数未被标记为绑定时。
        """
        return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
            self.NANOTRON_PARAMETER_METADATA_TIED_KEY
        ]

    @property
    def is_tied(self) -> bool:
        """判断该参数是否为绑定参数。

        Returns:
            bool: 如果参数被标记为绑定则返回 True。
        """
        return self.NANOTRON_PARAMETER_METADATA_TIED_KEY in getattr(
            self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
        )

    def mark_as_sharded(
        self,
        global_ranks: Tuple[int, ...],
        local_global_slices_pairs: Tuple[SlicesPair, ...],
        unsharded_shape: Tuple[int, ...],
    ):
        """将参数标记为分片参数。

        Args:
            global_ranks (Tuple[int, ...]): 持有该参数分片的所有全局 rank。
            local_global_slices_pairs (Tuple[SlicesPair, ...]): 本地切片与全局
                切片的映射对，描述本地数据在完整参数中的位置。
            unsharded_shape (Tuple[int, ...]): 完整（未分片）参数的形状。
        """
        self._set_metadata(
            self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY,
            ShardedInfo(
                global_ranks=global_ranks,
                local_global_slices_pairs=local_global_slices_pairs,
                unsharded_shape=unsharded_shape,
            ),
        )

    def get_sharded_info(self) -> ShardedInfo:
        """获取分片参数的元数据信息。

        Returns:
            ShardedInfo: 分片参数的元数据。

        Raises:
            KeyError: 当参数未被标记为分片时。
        """
        return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
            self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY
        ]

    @property
    def is_sharded(self) -> bool:
        """判断该参数是否为分片参数。

        Returns:
            bool: 如果参数被标记为分片则返回 True。
        """
        return self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY in getattr(
            self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
        )


def sanity_check(root_module: nn.Module):
    """验证模型的所有参数都是 NanotronParameter 类型。

    Nanotron 框架要求模型中的所有参数必须是 NanotronParameter 类型，
    以便正确管理分布式元数据（分片信息、绑定信息等）。
    此函数在模型构建完成后调用，确保格式正确。

    Args:
        root_module (nn.Module): 待检查的根模块（通常是完整模型）。

    Raises:
        ValueError: 当发现非 NanotronParameter 类型的参数时抛出，
            错误信息包含参数名称。
    """
    for name, param in root_module.named_parameters():
        if not isinstance(param, NanotronParameter):
            raise ValueError(
                f"Nanotronrequires model to be in Nanotronformat, ie all parameters are required to be a NanotronParameter. {name} isn't."
            )
