"""
混合专家（MoE）层实现 —— 专家并行（EP）的核心模块。

本模块实现了 Qwen2 模型的 MoE（Mixture of Experts）层，
包括路由器（Router）、分组 MLP（GroupedMLP）和 MoE 层（Qwen2MoELayer）。

MoE 与专家并行（EP）的关系：
    MoE 模型包含多个专家（Expert），每个 token 只被路由到 top-k 个专家。
    当专家数量较多时，单个 GPU 无法容纳所有专家，需要将专家分配到多个 GPU 上，
    这就是专家并行（Expert Parallelism）。

    在 EP 中：
        - 每个 GPU 持有 num_experts / expert_parallel_size 个专家
        - Token 根据路由结果被发送到持有对应专家的 GPU
        - 计算完成后，结果被收集回原始 GPU

    当前实现的限制：
        - 本实现假设 token 已经按专家分组排列（通过 ops.permute）
        - 跨设备的 token 路由（All-to-All 通信）尚未实现
        - 每个 GPU 独立处理本地专家，无需跨设备通信

核心组件：
    - Router: 计算每个 token 到各专家的路由权重和索引
    - GroupedMLP: 使用 Grouped GEMM 高效计算多个专家的 MLP
    - Qwen2MoELayer: 完整的 MoE 层，包括路由、分发、计算和合并

Grouped GEMM 优化：
    传统的 MoE 实现为每个专家单独计算 MLP，效率较低。
    GroupedMLP 使用 grouped_gemm 库的批量矩阵乘法（ops.gmm），
    将多个专家的计算合并为一次 GEMM 调用，显著提高 GPU 利用率。

    前提条件：
        - Token 必须按专家分组排列（通过 ops.permute 实现）
        - 需要知道每个专家的 token 数量（num_tokens_per_expert）
        - 权重形状为 [num_local_experts, in_features, out_features]

共享专家（Shared Expert）：
    Qwen2 MoE 支持共享专家机制：所有 token 都会经过一个共享的 MLP，
    其输出通过门控（sigmoid）与路由专家的输出相加。
    共享专家不参与路由，始终被使用。
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import Qwen2Config
from nanotron.models.base import ignore_init_on_device_and_dtype
from nanotron.nn.activations import ACT2FN

logger = logging.get_logger(__name__)


try:
    import grouped_gemm.ops as ops
except ImportError:
    raise RuntimeError(
        "Grouped GEMM is not available. Please run `pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main` (takes less than 5 minutes)"
    )


class Router(nn.Module):
    """MoE 路由器，计算 token 到专家的路由权重和索引。

    路由算法：
        1. 线性变换：logits = x @ W（W 形状 [num_experts, hidden_size]）
        2. Softmax 归一化：probs = softmax(logits)
        3. Top-k 选择：选择概率最高的 k 个专家

    注意事项：
        - 路由权重和计算使用 float32 精度，与 Qwen2 官方实现一致
        - 路由索引转换为 int32，因为 ops.permute 要求 int32 索引

    Attributes:
        num_experts (int): 专家总数。
        num_experts_per_token (int): 每个 token 选择的专家数（top-k）。
        weight (nn.Parameter): 路由权重，形状 [num_experts, hidden_size]，float32。
    """

    def __init__(
        self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs], tp_pg: dist.ProcessGroup, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.layer_idx = layer_idx

        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.top_k

        # float32 routing weights
        # NOTE: qwen keep the routing weights in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L608
        with ignore_init_on_device_and_dtype():
            self.weight = nn.Parameter(
                torch.randn(self.num_experts, config.hidden_size, dtype=torch.float32, device="cuda")
            )
        assert self.weight.dtype == torch.float32

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        """计算所有专家的路由 logits（未归一化）。

        Args:
            x: 输入张量，形状 [num_tokens, hidden_size]。

        Returns:
            torch.Tensor: 路由 logits，形状 [num_tokens, num_experts]，float32。
        """
        # NOTE: qwen keep the routing logits in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L613
        return F.linear(x.to(torch.float32), self.weight, bias=None)

    def routing(self, logits: torch.Tensor):
        """Top-k softmax 归一化路由。

        Args:
            logits: 路由 logits，形状 [num_tokens, num_experts]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_weights: Top-k 路由权重，形状 [num_tokens, top_k]，float32。
                - routing_indices: Top-k 路由索引，形状 [num_tokens, top_k]，int32。
        """
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, routing_indices = torch.topk(routing_weights, k=self.num_experts_per_token, dim=-1)
        routing_indices = routing_indices.to(torch.int32)  # NOTE: ops.permute requires indices to be int32
        return routing_weights, routing_indices

    def forward(self, x: torch.Tensor):
        """前向传播：计算路由权重和索引。"""
        logits = self.gating(x)
        return self.routing(logits)


class GroupedMLP(nn.Module):
    """分组 MLP，使用 Grouped GEMM 高效计算多个专家的前馈网络。

    与传统逐专家计算不同，GroupedMLP 将所有专家的权重合并为单个张量，
    使用 grouped_gemm 的批量矩阵乘法（ops.gmm）一次性计算所有专家的输出。

    权重组织方式：
        - merged_gate_up_proj: [num_local_experts, hidden_size, 2*intermediate_size]
          合并了 gate 投影和 up 投影，支持 SwiGLU 激活函数
        - merged_down_proj: [num_local_experts, intermediate_size, hidden_size]
          下投影权重

    与专家并行（EP）的关系：
        num_local_experts = num_experts / expert_parallel_size
        每个 GPU 只持有部分专家的权重，减少显存占用。

    Attributes:
        merged_gate_up_proj (nn.Parameter): 合并的 gate+up 投影权重。
        merged_down_proj (nn.Parameter): 下投影权重。
        act: 激活函数（如 SiLU/Swish）。
    """

    def __init__(self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()

        # 专家并行：每个设备只持有部分专家
        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size
        self.merged_gate_up_proj = nn.Parameter(
            torch.randn(num_local_experts, config.hidden_size, 2 * config.moe_config.moe_intermediate_size)
        )
        self.merged_down_proj = nn.Parameter(
            torch.randn(num_local_experts, config.moe_config.moe_intermediate_size, config.hidden_size)
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ):
        """前向传播：使用 Grouped GEMM 计算所有专家的 MLP。

        假设 hidden_states 已经按专家分组排列（通过 ops.permute）。

        Args:
            hidden_states: 已排列的输入张量，形状 [total_tokens, hidden_size]。
            num_tokens_per_expert: 每个专家的 token 数量，形状 [num_local_experts]。

        Returns:
            dict: {"hidden_states": 输出张量，形状 [total_tokens, hidden_size]}。

        注意：
            - ops.gmm 要求 num_tokens_per_expert 在 CPU 上
            - ops.gmm 要求输入为 bfloat16
        """
        # NOTE: ops.gemm requires "batch_sizes" (aka: num_tokens_per_expert here) to be on cpu
        num_tokens_per_expert = num_tokens_per_expert.to("cpu")
        # 合并的 gate+up 投影：一次 GEMM 计算两个投影
        merged_states = ops.gmm(hidden_states, self.merged_gate_up_proj, num_tokens_per_expert, trans_b=False)
        # SwiGLU 激活：gate_states * up_states
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states
        # 下投影
        hidden_states = ops.gmm(hidden_states, self.merged_down_proj, num_tokens_per_expert, trans_b=False)

        return {"hidden_states": hidden_states}


class Qwen2MoELayer(nn.Module):
    """Qwen2 模型的 MoE（混合专家）层。

    实现了完整的 MoE 前向传播流程：
        1. 路由：计算每个 token 到各专家的路由权重和索引
        2. 分发：将 token 按路由索引重排（permute），按专家分组
        3. 计算：使用 GroupedMLP 计算各专家的输出
        4. 合并：将专家输出按路由权重加权合并（unpermute）

    与专家并行（EP）的关系：
        - num_local_experts = num_experts / expert_parallel_size
        - 每个 GPU 只持有部分专家
        - 当前实现假设 token 已经在正确的 GPU 上
          （完整的 EP 需要 All-to-All 通信来路由 token）

    共享专家机制：
        如果启用共享专家（enable_shared_expert=True），
        所有 token 还会经过一个共享的 MLP，其输出通过 sigmoid 门控
        与路由专家的输出相加：output = routed_output + gate * shared_output

    梯度重计算：
        如果 recompute_layer=True，MoE 层的前向传播会在反向传播时重新计算，
        以节省激活值的显存占用。

    Attributes:
        num_experts (int): 专家总数。
        num_local_experts (int): 当前 GPU 持有的专家数。
        num_experts_per_token (int): 每个 token 选择的专家数（top-k）。
        expert_parallel_size (int): 专家并行度。
        router (Router): 路由器模块。
        experts (GroupedMLP): 分组 MLP 模块。
        shared_expert (Optional[Qwen2MLP]): 共享专家 MLP（如果启用）。
        shared_expert_gate (Optional[nn.Linear]): 共享专家门控（如果启用）。
        recompute_layer (bool): 是否使用梯度重计算。
    """

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # MoE specific configurations
        self.num_experts = config.moe_config.num_experts  # Total number of experts
        self.num_local_experts = (
            config.moe_config.num_experts // parallel_config.expert_parallel_size
        )  # Experts per device
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = self.num_experts // self.expert_parallel_size  # Experts per device

        # Router for selecting experts
        self.router = Router(config, parallel_config, tp_pg, layer_idx)

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                intermediate_size=config.moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        # Create the expert MLPs
        self.experts = GroupedMLP(config, parallel_config)
        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer

    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
    ):
        """将 token 按路由索引分发到对应的专家。

        使用 ops.permute 将 token 重排为按专家分组的顺序，
        并返回逆映射用于后续合并。

        Args:
            hidden_states: 输入张量，形状 [num_tokens, hidden_size]。
            routing_indices: 路由索引，形状 [num_tokens, top_k]，int32。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dispatched_inputs: 按专家分组的输入，形状 [total_routed_tokens, hidden_size]。
                - inverse_permute_mapping: 逆映射，用于合并时恢复原始顺序。
                - num_tokens_per_expert: 每个专家的 token 数量，形状 [num_local_experts]。
        """
        # NOTE: start from expert 0 to expert n
        num_tokens_per_expert = torch.bincount(
            routing_indices.flatten(), minlength=self.num_local_experts
        )  # [num_local_experts]
        dispatched_inputs, inverse_permute_mapping = ops.permute(hidden_states, routing_indices)
        return dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert

    def _combine_expert_outputs(self, expert_outputs, inverse_mapping, routing_weights):
        """将各专家的输出按路由权重加权合并，恢复原始 token 顺序。

        Args:
            expert_outputs: 专家输出，形状 [total_routed_tokens, hidden_size]。
            inverse_mapping: 逆映射索引。
            routing_weights: 路由权重，用于加权合并。

        Returns:
            torch.Tensor: 合并后的输出，形状 [num_tokens, hidden_size]。
        """
        hidden_states = ops.unpermute(expert_outputs, inverse_mapping, routing_weights)
        return hidden_states

    def _core_forward(self, hidden_states):
        """MoE 层的核心前向逻辑。

        流程：
            1. 路由：计算 top-k 路由权重和索引
            2. 分发：按路由索引重排 token
            3. 计算：GroupedMLP 计算各专家输出
            4. 合并：按路由权重加权合并，恢复原始顺序
            5. 共享专家：如果启用，加上共享专家的贡献

        Args:
            hidden_states: 输入张量。

        Returns:
            torch.Tensor: MoE 层输出。
        """
        # Get top-k routing weights and indices
        routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]

        # Dispatch tokens to experts
        dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert = self._dispatch_tokens(
            hidden_states, routing_indices
        )

        expert_outputs = self.experts(dispatched_inputs, num_tokens_per_expert)

        output = self._combine_expert_outputs(
            expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights
        )

        # Add shared expert contribution if enabled
        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

        return output

    def _checkpointed_forward(self, hidden_states):
        """使用梯度重计算的前向传播，节省显存。"""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states):
        """MoE 层前向传播。

        根据 recompute_layer 配置选择是否使用梯度重计算。

        Args:
            hidden_states: 输入张量。

        Returns:
            dict: {"hidden_states": MoE 层输出}。
        """
        if self.recompute_layer and self.training:
            hidden_states = self._checkpointed_forward(hidden_states)
        else:
            hidden_states = self._core_forward(hidden_states)

        return {"hidden_states": hidden_states}
