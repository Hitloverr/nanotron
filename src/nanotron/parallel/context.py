"""
并行上下文模块 —— Nanotron 分布式训练的基础设施。

本模块负责初始化和管理所有分布式并行策略的进程组（Process Group），
包括张量并行（TP）、流水线并行（PP）、数据并行（DP）、上下文并行（CP）和专家并行（EP）。

核心设计思路：
    将全局进程按照 5 维张量 [EP, PP, DP, CP, TP] 进行排列，
    通过不同的转置和 reshape 操作派生出各类并行进程组。
    约束条件：EP × PP × DP × CP × TP == WORLD_SIZE
"""

import os
from typing import Dict, Literal

import numpy as np
import torch

import nanotron.distributed as dist

DistributedBackend = Literal["gloo", "mpi", "nccl"]


class ParallelContext:
    """分布式并行上下文管理器，负责创建和管理所有并行进程组。

    ParallelContext 是 Nanotron 分布式训练的核心基础设施，它在初始化时根据
    配置的并行度创建所有必要的进程组，后续的模型构建、通信操作等都依赖
    这些进程组来协调不同并行策略下的 GPU 间通信。

    进程组拓扑结构（5D 排列）：
        ranks = np.arange(WORLD_SIZE).reshape(EP, PP, DP, CP, TP)

    派生的进程组：
        - tp_pg:   张量并行组，同一 PP/DP/CP/EP 内的 TP 组
        - pp_pg:   流水线并行组，同一 TP/DP/CP/EP 内的 PP 组
        - dp_pg:   数据并行组，同一 TP/PP/CP/EP 内的 DP 组
        - cp_pg:   上下文并行组，同一 TP/PP/DP/EP 内的 CP 组
        - ep_pg:   专家并行组，同一 TP/PP/DP/CP 内的 EP 组
        - mp_pg:   模型并行组（TP + PP + EP 的组合）
        - dp_cp_pg: DP+CP 组合进程组
        - tp_and_ep_pg: TP+EP 组合进程组

    Attributes:
        tensor_parallel_size: 张量并行度
        pipeline_parallel_size: 流水线并行度
        data_parallel_size: 数据并行度
        context_parallel_size: 上下文并行度
        expert_parallel_size: 专家并行度
        world_size: 全局进程数
        local_world_size: 单节点进程数
        world_pg: 全局进程组
        tp_pg: 张量并行进程组
        pp_pg: 流水线并行进程组
        dp_pg: 数据并行进程组
        cp_pg: 上下文并行进程组
        ep_pg: 专家并行进程组
        mp_pg: 模型并行进程组
        dp_cp_pg: 数据+上下文并行进程组
        tp_and_ep_pg: 张量+专家并行进程组
        world_rank_matrix: 5D 进程排列矩阵
        parallel_order: 维度顺序标识
    """

    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        context_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        backend: DistributedBackend = "nccl",
    ):
        """初始化并行上下文，创建所有分布式进程组。

        Args:
            tensor_parallel_size (int): 张量并行度，将模型权重矩阵切分到多少个 GPU 上。
            pipeline_parallel_size (int): 流水线并行度，将模型层切分到多少个 GPU 上。
            data_parallel_size (int): 数据并行度，数据副本数量。
            context_parallel_size (int, optional): 上下文并行度，用于超长序列训练。
                默认为 1（不启用）。
            expert_parallel_size (int, optional): 专家并行度，用于 MoE 模型的专家切分。
                默认为 1（不启用）。
            backend (DistributedBackend, optional): 分布式通信后端，目前仅支持 "nccl"。
                默认为 "nccl"。

        Raises:
            AssertionError: 当 TP × PP × DP × CP × EP ≠ WORLD_SIZE 时抛出。
            ValueError: 当 torch.distributed 不可用时抛出。

        Note:
            环境变量 WORLD_SIZE 和 LOCAL_WORLD_SIZE 必须在调用前设置。
            当前仅支持 nccl 后端，因为 GPU 间高速通信依赖 NCCL。
        """
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8")) if world_size > 8 else world_size

        assert (
            tensor_parallel_size * pipeline_parallel_size * context_parallel_size * data_parallel_size
        ) == world_size, f"TP*CP*DP*PP={tensor_parallel_size}*{pipeline_parallel_size}*{context_parallel_size}*{data_parallel_size}={tensor_parallel_size * pipeline_parallel_size * context_parallel_size * data_parallel_size} != WORLD_SIZE={world_size}"

        if not dist.is_available():
            raise ValueError("torch.distributed is not available as a package, please install it.")

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.context_parallel_size = context_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.world_size = world_size
        self.local_world_size = local_world_size

        self._groups = {}

        self.set_device()

        assert backend == "nccl", "Only nccl backend is supported for now."

        if not dist.is_initialized():
            dist.initialize_torch_distributed()

        ranks = list(range(self.world_size))
        process_group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(),
        )
        self.world_pg = process_group

        self._init_parallel_groups()

    def _init_parallel_groups(self):
        """初始化 3D（实际为 5D）并行策略的所有进程组。

        核心算法：
            1. 将全局 rank 按 [EP, PP, DP, CP, TP] 的 5 维形状排列
            2. 通过 numpy.transpose + reshape 操作，将同一并行维度的 rank
               组合到同一行，从而派生出各类进程组
            3. 使用 create_new_group 创建去重的 ProcessGroup 对象

        进程组派生逻辑示例（以 TP 为例）：
            ranks.transpose((0,1,2,3,4)).reshape(-1, TP_SIZE)
            → 将 TP 维度放最后，reshape 后每行即为一个 TP 进程组

        性能考量：
            - 所有进程组创建前后都有 barrier 同步，确保全局一致性
            - world_ranks_to_pg 字典避免重复创建相同的 ProcessGroup
        """
        dist.barrier()
        ranks = np.arange(0, self.world_size).reshape(
            (
                self.expert_parallel_size,
                self.pipeline_parallel_size,
                self.data_parallel_size,
                self.context_parallel_size,
                self.tensor_parallel_size,
            )
        )
        self.world_ranks_to_pg = {}
        self.local_pg = self.create_new_group(ranks.reshape((-1, self.local_world_size)))
        assert int(os.environ.get("LOCAL_RANK")) == dist.get_rank(self.local_pg), "Local rank mismatch"

        # 通过转置将目标维度放到最内层，再 reshape 为 (组数, 组大小) 的形式
        # 每行代表一个进程组，包含该组内所有 rank
        self.tp_pg = self.create_new_group(ranks.transpose((0, 1, 2, 3, 4)).reshape((-1, self.tensor_parallel_size)))
        self.cp_pg = self.create_new_group(ranks.transpose((4, 0, 1, 2, 3)).reshape((-1, self.context_parallel_size)))
        self.dp_pg = self.create_new_group(ranks.transpose((3, 4, 0, 1, 2)).reshape((-1, self.data_parallel_size)))
        self.pp_pg = self.create_new_group(ranks.transpose((2, 3, 4, 0, 1)).reshape((-1, self.pipeline_parallel_size)))
        self.ep_pg = self.create_new_group(
            ranks.transpose((1, 2, 3, 4, 0)).reshape((-1, self.expert_parallel_size))
        )

        # 模型并行组 = TP + PP + EP 的组合（同一 DP 和 CP rank 内）
        # 用于标识哪些 rank 持有模型的不同部分
        self.mp_pg = self.create_new_group(
            [
                ranks[:, :, dp_rank, cp_rank, :].reshape(-1)
                for cp_rank in range(self.context_parallel_size)
                for dp_rank in range(self.data_parallel_size)
            ]
        )

        # DP+CP 组合进程组：用于需要同时跨 DP 和 CP 维度通信的场景
        self.dp_cp_pg = self.create_new_group(
            [
                ranks[ep_rank, pp_rank, :, :, tp_rank].reshape(-1)
                for tp_rank in range(self.tensor_parallel_size)
                for pp_rank in range(self.pipeline_parallel_size)
                for ep_rank in range(self.expert_parallel_size)
            ]
        )

        # TP+EP 组合进程组：用于 MoE 中专家并行的 TP 通信
        self.tp_and_ep_pg = self.create_new_group(
            [
                ranks[:, pp_rank, dp_rank, cp_rank, :].reshape(-1)
                for cp_rank in range(self.context_parallel_size)
                for pp_rank in range(self.pipeline_parallel_size)
                for dp_rank in range(self.data_parallel_size)
            ]
        )

        self.world_rank_matrix: np.ndarray = ranks
        self.parallel_order = ["ep", "pp", "dp", "cp", "tp"]

    def create_new_group(self, all_groups_ranks: np.ndarray) -> dist.ProcessGroup:
        """根据给定的 rank 分组方案创建进程组，返回当前 rank 所属的进程组。

        该方法会遍历所有分组方案，为每个唯一的 rank 集合创建一个 ProcessGroup，
        并缓存到 world_ranks_to_pg 中避免重复创建。

        Args:
            all_groups_ranks (np.ndarray): 形状为 (组数, 组大小) 的数组，
                每行包含一个进程组内所有 rank 的编号。

        Returns:
            dist.ProcessGroup: 当前 rank 所属的进程组。

        Note:
            - 使用 sorted tuple 作为缓存键，确保相同 rank 集合只创建一个 ProcessGroup
            - 创建前后都有 barrier 同步，保证所有进程的一致性
        """
        dist.barrier()
        rank = int(os.environ["RANK"])
        new_group_containing_rank = None
        for group_ranks in all_groups_ranks:
            sorted_ranks = tuple(sorted(group_ranks))

            if sorted_ranks not in self.world_ranks_to_pg:
                new_group = dist.new_group(ranks=group_ranks)
                self.world_ranks_to_pg[sorted_ranks] = new_group
            else:
                new_group = self.world_ranks_to_pg[sorted_ranks]

            if rank in sorted_ranks:
                new_group_containing_rank = new_group
        dist.barrier()
        return new_group_containing_rank

    def set_device(self):
        """设置当前进程使用的 CUDA 设备。

        根据 LOCAL_RANK 环境变量将当前进程绑定到对应的 GPU 上。
        假设各节点 GPU 数量相同（同构节点）。
        """
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))

    def get_local_ranks(self, world_rank: int) -> Dict[str, int]:
        """根据全局 rank 获取在各并行维度上的本地 rank。

        Args:
            world_rank (int): 全局 rank 编号。

        Returns:
            Dict[str, int]: 各并行维度上的本地 rank，键为维度名称
                （"ep", "pp", "dp", "cp", "tp"），值为对应的本地 rank。

        Example:
            >>> ctx.get_local_ranks(5)
            {'ep': 0, 'pp': 1, 'dp': 0, 'cp': 0, 'tp': 1}
        """
        local_ranks = np.where(self.world_rank_matrix == world_rank)
        return {ax: local_ranks[i].item() for i, ax in enumerate(self.parallel_order)}

    def destroy(self):
        """销毁所有进程组并清理分布式资源。

        在训练结束后调用，释放 ProcessGroup 占用的资源。
        如果分布式尚未初始化则直接返回。
        """
        if not dist.is_initialized():
            return

        dist.barrier()
        dist.destroy_process_group()

    def get_global_rank(
        self,
        ep_rank: int,
        pp_rank: int,
        dp_rank: int,
        cp_rank: int,
        tp_rank: int,
    ) -> np.int64:
        """根据各并行维度上的本地 rank 计算全局 rank。

        Args:
            ep_rank (int): 专家并行维度上的 rank。
            pp_rank (int): 流水线并行维度上的 rank。
            dp_rank (int): 数据并行维度上的 rank。
            cp_rank (int): 上下文并行维度上的 rank。
            tp_rank (int): 张量并行维度上的 rank。

        Returns:
            numpy.int64: 对应的全局 rank 编号。

        Example:
            >>> ctx.get_global_rank(ep_rank=0, pp_rank=1, dp_rank=0, cp_rank=0, tp_rank=0)
            2
        """
        return self.world_rank_matrix[ep_rank, pp_rank, dp_rank, cp_rank, tp_rank]
