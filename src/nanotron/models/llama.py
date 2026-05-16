# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLaMA 模型实现 —— Nanotron 中的 LLaMA/Llama2/Llama3 大语言模型。

本模块实现了完整的 LLaMA 模型架构，包括：
    - RotaryEmbedding / LlamaRotaryEmbedding: 旋转位置编码（RoPE）的两种实现
    - GLUActivation: 门控线性单元激活函数（SwiGLU）
    - MLP: 前馈网络（gate_up_proj + down_proj）
    - CoreAttention: 基于 FlashAttention 的核心注意力计算
    - CausalSelfAttention: 因果自注意力（支持 GQA/MQA）
    - LlamaDecoderLayer: LLaMA 解码器层（Pre-Norm 架构）
    - Embedding: 词嵌入层
    - LlamaModel: LLaMA 模型主体（PipelineBlock 组装）
    - Loss / LossWithZLoss: 交叉熵损失（可选 Z-loss 正则化）
    - LlamaForTraining: 训练用模型（NanotronModel 子类）

架构特点：
    - Pre-Norm: LayerNorm 在注意力/MLP 之前应用
    - SwiGLU: 使用门控线性单元作为激活函数
    - GQA/MQA: 支持分组查询注意力和多查询注意力
    - RoPE: 旋转位置编码，支持交错和非交错两种模式
    - 张量并行: QKV 投影和 MLP 使用列/行切分
    - 流水线并行: 所有子模块通过 PipelineBlock 包装
"""

from typing import Dict, List, Optional, Union

import torch
from flash_attn import bert_padding
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
)
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, LlamaConfig, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method

logger = logging.get_logger(__name__)


class RotaryEmbedding(nn.Module):
    """交错式旋转位置编码（Interleaved RoPE）。

    将旋转位置编码应用于查询和键向量，使模型能够感知序列中
    token 的相对位置。此实现使用交错模式（相邻维度配对旋转）。

    该实现支持动态扩展位置编码长度：当输入序列超过当前 end 值时，
    自动将 end 翻倍并重新计算频率缓冲区。

    Attributes:
        dim (int): 旋转编码的维度（必须为偶数），通常等于 d_qk。
        end (int): 支持的最大序列长度，可动态扩展。
        theta (float): RoPE 的基础频率，默认为 10000.0。
        freqs_cis (torch.Tensor): 预计算的复数频率缓冲区，形状为 [end, dim//2, 2]。
        _initialized_buffer (bool): 缓冲区是否已初始化。
    """

    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        """初始化交错式旋转位置编码。

        Args:
            dim (int): 旋转编码的维度（必须为偶数）。
            end (int): 初始最大序列长度。
            theta (float, optional): 基础频率参数。默认为 10000.0。
        """
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self):
        """初始化旋转位置编码的频率缓冲区。

        计算复数频率并存储到 GPU 缓冲区中。频率计算在 CPU 上完成
        以确保数值精度，然后拷贝到 GPU。
        """
        if self._initialized_buffer is True:
            return
        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device="cuda"),
            persistent=False,
        )
        assert self.freqs_cis.device.type == "cuda"
        if self.freqs_cis.dtype != torch.float:
            self.freqs_cis = self.freqs_cis.to(torch.float)
        assert self.freqs_cis.dtype == torch.float
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu")[: (self.dim // 2)] / self.dim)
        ).to(
            "cuda"
        )
        t = torch.arange(self.end, device="cuda")
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
    ):
        """应用旋转位置编码。

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_length, num_heads, d_qk]。
            position_ids (Optional[torch.LongTensor]): 位置 ID，形状为 [batch_size, seq_length]。
                None 表示使用连续位置 [0, seq_length)。

        Returns:
            torch.Tensor: 应用旋转编码后的张量，形状与输入相同。
        """
        batch_size, seq_length, num_heads, inner_dim = x.shape
        while (
            position_ids is not None and position_ids[-1, -1] >= self.end
        ) or seq_length >= self.end:
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            print(f"Initializing rotary embeddings with end={self.end}")
            self.init_rotary_embeddings()
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(
            batch_size, seq_length, num_heads, inner_dim // 2, 2
        )
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)
        if position_ids is None:
            freqs_cis = self.freqs_cis[None, :seq_length, None, :]
        else:
            if position_ids[-1, -1] < 0 or position_ids[-1, -1] >= self.end:
                raise ValueError(f"Position ids must be in the range [0, {self.end}), but got {position_ids}")
            freqs_cis = self.freqs_cis[position_ids][:, :, None, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)


class LlamaRotaryEmbedding(nn.Module):
    """非交错式旋转位置编码（Non-Interleaved RoPE），与 HuggingFace Transformers 兼容。

    与 RotaryEmbedding 的区别：
        - RotaryEmbedding 使用交错模式（相邻维度配对），通过复数乘法实现
        - LlamaRotaryEmbedding 使用非交错模式（前半/后半配对），通过 cos/sin 分离实现

    非交错模式是 Llama2/Llama3 的标准实现，便于与 HuggingFace 权重互转。

    Attributes:
        dim (int): 旋转编码的维度。
        end (int): 支持的最大序列长度。
        theta (float): 基础频率参数，默认为 500000.0（Llama2/3 的默认值）。
        inv_freq (torch.Tensor): 逆频率缓冲区，形状为 [dim//2]。
    """

    def __init__(self, dim: int, end: int, theta: float = 500000.0):
        """初始化非交错式旋转位置编码。

        Args:
            dim (int): 旋转编码的维度。
            end (int): 最大序列长度。
            theta (float, optional): 基础频率参数。默认为 500000.0。
        """
        super().__init__()
        self.dim = dim
        self.end = end
        self.theta = theta
        self.init_rotary_embeddings()

    def init_rotary_embeddings(self):
        """初始化逆频率缓冲区。

        逆频率在 CPU 上计算以确保数值精度，然后拷贝到 GPU。
        """
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu") / self.dim)
        )
        self.register_buffer(
            "inv_freq", torch.empty(self.dim // 2, dtype=torch.float, device="cuda"), persistent=False
        )
        self.inv_freq = self.inv_freq.to(
            torch.float
        )
        self.inv_freq.copy_(inv_freq)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
    ):
        """计算旋转位置编码的 cos 和 sin 值。

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_length, num_heads, d_qk]。
            position_ids (Optional[torch.LongTensor]): 位置 ID，形状为 [batch_size, seq_length]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos, sin) 张量对，
                形状均为 [batch_size, seq_length, head_dim]。
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        """将输入张量的后半部分旋转到前半部分。

        将 x 沿最后一维分为两半，返回 (-x2, x1) 的拼接结果。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 旋转后的张量。
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        """将旋转位置编码应用于查询和键张量。

        Args:
            q (torch.Tensor): 查询张量。
            k (torch.Tensor): 键张量。
            cos (torch.Tensor): 余弦部分。
            sin (torch.Tensor): 正弦部分。
            unsqueeze_dim (int, optional): 在哪个维度上扩展 cos/sin 以便广播。
                默认为 2（对应 [batch, seq, heads, dim] 布局）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 应用旋转编码后的 (q, k) 张量对。
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class GLUActivation(nn.Module):
    """门控线性单元（GLU）激活函数，实现 SwiGLU。

    将输入沿最后一维分为两半（gate 和 up），对 gate 应用激活函数后
    与 up 逐元素相乘。这是 LLaMA 中使用的激活函数。

    公式: output = act(gate) * up

    Attributes:
        act (Callable): 激活函数（如 SiLU/Swish）。
    """

    def __init__(self, act_fn_name: str):
        """初始化 GLU 激活函数。

        Args:
            act_fn_name (str): 激活函数名称，如 "silu"。
        """
        super().__init__()
        self.act = ACT2FN[act_fn_name]

    def forward(self, merged_states: torch.Tensor):
        """执行 GLU 激活。

        Args:
            merged_states (torch.Tensor): 合并的 gate 和 up 状态，
                形状为 [..., 2 * intermediate_size]。

        Returns:
            torch.Tensor: 激活后的输出，形状为 [..., intermediate_size]。
        """
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states


class MLP(nn.Module):
    """LLaMA 的前馈网络（MLP），使用 SwiGLU 激活函数。

    结构: gate_up_proj → GLUActivation → down_proj

    gate_up_proj 将 hidden_size 映射到 2 * intermediate_size，
    然后通过 GLU 激活函数分为 gate 和 up 两路，
    最后通过 down_proj 映射回 hidden_size。

    张量并行：
        - gate_up_proj: 列切分（ColumnLinear），输出维度切分
        - down_proj: 行切分（RowLinear），输入维度切分

    Attributes:
        gate_up_proj (TensorParallelColumnLinear): 合并的 gate 和 up 投影层。
        down_proj (TensorParallelRowLinear): 下投影层。
        split_silu_mul (GLUActivation): GLU 激活函数。
    """

    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        """初始化 MLP。

        Args:
            config (LlamaConfig): LLaMA 模型配置。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
            tp_pg (dist.ProcessGroup): 张量并行进程组。
        """
        super().__init__()

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,
            config.intermediate_size,
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.split_silu_mul = GLUActivation(config.hidden_act)

    def forward(self, hidden_states):
        """MLP 前向传播。

        Args:
            hidden_states (torch.Tensor): 输入隐藏状态，形状为 [seq_length, batch_size, hidden_dim]。

        Returns:
            Dict[str, torch.Tensor]: 包含 "hidden_states" 键的字典。
        """
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}


class CoreAttention(nn.Module):
    """核心注意力计算模块，基于 FlashAttention 实现。

    使用 flash_attn_varlen_func 进行高效的变长序列注意力计算，
    支持因果掩码和 µTransfer 缩放。

    Attributes:
        d_qk (int): 查询/键的头部维度。
        d_v (int): 值的头部维度。
        is_using_mup (bool): 是否使用 µTransfer 参数化。
        checkpoint_attention (bool): 是否使用梯度检查点（FlashAttention 已内置）。
    """

    def __init__(self, config: LlamaConfig, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        """初始化核心注意力模块。

        Args:
            config (LlamaConfig): LLaMA 模型配置。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
            layer_idx (int): 层索引。
        """
        super().__init__()
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.is_using_mup = config.is_using_mup

        self.checkpoint_attention = False

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_sequence_mask: torch.Tensor,
        kv_sequence_mask: torch.Tensor,
    ):
        """执行核心注意力计算。

        Args:
            query_states (torch.Tensor): 查询状态，形状为 [batch_size * q_length, n_local_q_heads, d_qk]。
            key_states (torch.Tensor): 键状态，形状为 [batch_size * kv_length, n_local_kv_heads, d_qk]。
            value_states (torch.Tensor): 值状态，形状为 [batch_size * kv_length, n_local_kv_heads, d_v]。
            q_sequence_mask (torch.Tensor): 查询序列掩码，形状为 [batch_size, q_length]。
            kv_sequence_mask (torch.Tensor): 键值序列掩码，形状为 [batch_size, kv_length]。

        Returns:
            torch.Tensor: 注意力输出，形状为 [total_q, n_local_q_heads, d_v]。
        """
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
        torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

        causal = False if q_sequence_mask.shape[1] == 1 else True

        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_sequence_mask.shape[1],
            max_seqlen_k=kv_sequence_mask.shape[1],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output


def pad_to_right(tensor, mask, new_tensor=None):
    """将左填充张量转换为右填充张量。

    在推理的 prefill 阶段，需要将左填充的 key/value 状态转换为右填充格式，
    以便正确写入 KV 缓存。

    Args:
        tensor (torch.Tensor): 输入张量，形状为 [batch_size, seqlen, d1, d2]。
        mask (torch.Tensor): 掩码张量，形状为 [batch_size, seqlen]，
            True 表示有效 token，False 表示填充。
        new_tensor (Optional[torch.Tensor]): 可选的输出张量，形状为 [batch_size, new_seqlen, d1, d2]。
            如果为 None，则创建与输入相同形状的零张量。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - new_tensor (torch.Tensor): 右填充后的张量。
            - right_padded_mask (torch.Tensor): 右填充后的掩码，形状为 [batch_size, seqlen]。
    """
    unpad_seqlens = mask.sum(1)
    max_seqlen = mask.shape[1]
    indices = torch.arange(max_seqlen, device=mask.device)
    right_padded_mask = indices < unpad_seqlens[:, None]
    useful_values = tensor[mask]
    new_tensor = torch.zeros_like(tensor) if new_tensor is None else new_tensor
    new_tensor[:, : right_padded_mask.shape[1], :, :][right_padded_mask] = useful_values
    return new_tensor, right_padded_mask


class CausalSelfAttention(nn.Module, AttachableStore):
    """因果自注意力模块，支持 GQA（分组查询注意力）和 MQA（多查询注意力）。

    该模块实现了 LLaMA 的自注意力机制，包括：
        - QKV 投影（列切分张量并行）
        - 旋转位置编码（RoPE）
        - FlashAttention 高效注意力计算
        - 输出投影（行切分张量并行）
        - KV 缓存（推理模式）

    张量并行策略：
        注意力头沿 TP 维度切分，每个 GPU 持有一部分 Q/K/V 头。
        - n_local_q_heads = num_attention_heads / TP_SIZE
        - n_local_kv_heads = num_key_value_heads / TP_SIZE

    GQA 支持：
        当 num_key_value_heads < num_attention_heads 时启用 GQA，
        多个 Q 头共享同一组 K/V 头（n_repeats = Q_heads / KV_heads）。

    推理模式：
        通过 AttachableStore 接口支持 KV 缓存，实现自回归生成。
        首次推理（prefill）使用 flash_attn_varlen_func，
        后续推理（decode）使用 flash_attn_with_kvcache。

    Attributes:
        n_local_q_heads (int): 当前 rank 持有的查询头数。
        n_local_kv_heads (int): 当前 rank 持有的键值头数。
        n_repeats (int): GQA 中每组 KV 头对应的 Q 头数。
        is_gqa (bool): 是否使用 GQA。
        d_qk (int): 查询/键的头部维度。
        d_v (int): 值的头部维度。
        d_model (int): 模型隐藏维度。
        is_using_mup (bool): 是否使用 µTransfer 参数化。
        qkv_proj (TensorParallelColumnLinear): QKV 合并投影层。
        rotary_embedding: 旋转位置编码模块。
        o_proj (TensorParallelRowLinear): 输出投影层。
        attention (CoreAttention): 核心注意力计算模块。
        prefill_kv_len (int): KV 缓存预分配长度。
    """

    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        """初始化因果自注意力模块。

        Args:
            config (LlamaConfig): LLaMA 模型配置。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
            tp_pg (dist.ProcessGroup): 张量并行进程组。
            layer_idx (int): 层索引。
        """
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk,
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        if config.rope_interleaved:
            self.rotary_embedding = RotaryEmbedding(
                dim=self.d_qk,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        else:
            self.rotary_embedding = LlamaRotaryEmbedding(
                dim=self.d_qk,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        self.rope_interleaved = config.rope_interleaved

        self.flash_rotary_embedding = FlashRotaryEmbedding(
            dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved
        )

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = config.max_position_embeddings

    def forward(
        self,
        hidden_states,
        sequence_mask,
    ):
        """自注意力前向传播，自动区分训练和推理模式。

        Args:
            hidden_states (torch.Tensor): 输入隐藏状态，形状为 [seq_length, batch_size, hidden_size]。
            sequence_mask (torch.Tensor): 序列掩码，形状为 [batch_size, seq_length]。

        Returns:
            Dict[str, torch.Tensor]: 包含 "hidden_states" 和 "sequence_mask" 的字典。
        """
        qkv_states = self.qkv_proj(hidden_states)
        q_length, batch_size, _ = qkv_states.shape

        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [
                    self.n_local_q_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                ],
                dim=-1,
            )

            query_states = (
                query_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
            )
            key_states = (
                key_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
            value_states = (
                value_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
        else:
            query_states, key_states, value_states = (
                qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )

        store = self.get_local_store()
        if store is not None:
            return self._forward_inference(
                query_states, key_states, value_states, sequence_mask, batch_size, q_length, store
            )
        else:
            return self._forward_training(query_states, key_states, value_states, sequence_mask, batch_size, q_length)

    def _forward_inference(self, query_states, key_states, value_states, sequence_mask, batch_size, q_length, store):
        """推理模式前向传播，支持 KV 缓存的自回归生成。

        分两个阶段：
            1. Prefill（首次推理）：处理完整输入序列，初始化 KV 缓存
            2. Decode（后续推理）：每次处理一个 token，使用 KV 缓存

        Args:
            query_states (torch.Tensor): 查询状态。
            key_states (torch.Tensor): 键状态。
            value_states (torch.Tensor): 值状态。
            sequence_mask (torch.Tensor): 序列掩码。
            batch_size (int): 批大小。
            q_length (int): 查询序列长度。
            store (dict): KV 缓存存储。

        Returns:
            Dict[str, torch.Tensor]: 包含 "hidden_states" 和 "sequence_mask" 的字典。
        """
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache

        assert key_states.requires_grad is False
        assert value_states.requires_grad is False

        if "position_offsets" in store:
            old_position_offsets = store["position_offsets"]
            position_ids = old_position_offsets[:, None] + sequence_mask
        else:
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        position_offsets = position_ids[:, -1]

        old_rotary_embed_end = self.rotary_embedding.end
        if self.rope_interleaved:
            query_states = self.rotary_embedding(query_states, position_ids=position_ids)
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)
        else:
            cos, sin = self.rotary_embedding(value_states, position_ids)
            query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            old_rotary_embed_end = self.rotary_embedding.end
            if self.rope_interleaved:
                query_states = self.rotary_embedding(query_states, position_ids=position_ids)
                key_states = self.rotary_embedding(key_states, position_ids=position_ids)
            else:
                cos, sin = self.rotary_embedding(value_states, position_ids)
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            if "key" not in store:
                # Prefill 阶段：首次推理，初始化 KV 缓存
                assert ~(
                    sequence_mask[:, :-1] & (~sequence_mask[:, 1:])
                ).any(), "Can't mask in the middle of sequence, please make sure that pads are at the left of the sequence if existing"

                k_cache = torch.zeros(
                    (
                        batch_size,
                        self.prefill_kv_len,
                        self.n_local_kv_heads,
                        self.d_qk,
                    ),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                v_cache = torch.zeros(
                    (batch_size, self.prefill_kv_len, self.n_local_kv_heads, self.d_v),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(
                    query_states,
                    sequence_mask,
                )
                (key_unpad, indices_k, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(
                    key_states, sequence_mask
                )
                (value_unpad, _, _, _) = bert_padding.unpad_input(value_states, sequence_mask)

                softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
                output_unpad = flash_attn_varlen_func(
                    q=query_unpad,
                    k=key_unpad,
                    v=value_unpad,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=True,
                    return_attn_probs=False,
                )

                attention_output = bert_padding.pad_input(
                    output_unpad, indices_q, batch_size, q_length
                )

                pad_to_right(key_states, sequence_mask, new_tensor=k_cache)
                pad_to_right(value_states, sequence_mask, new_tensor=v_cache)

            else:
                # Decode 阶段：后续推理，使用 KV 缓存
                k_cache = store["key"]
                v_cache = store["value"]

                if self.rotary_embedding.end > old_rotary_embed_end:
                    k_cache = torch.cat(
                        [
                            k_cache,
                            torch.zeros(
                                (
                                    batch_size,
                                    self.rotary_embedding.end - old_rotary_embed_end,
                                    self.n_local_kv_heads,
                                    self.d_qk,
                                ),
                                dtype=query_states.dtype,
                                device=query_states.device,
                            ),
                        ],
                        dim=1,
                    )

                    v_cache = torch.cat(
                        [
                            v_cache,
                            torch.zeros(
                                (
                                    batch_size,
                                    self.rotary_embedding.end - old_rotary_embed_end,
                                    self.n_local_kv_heads,
                                    self.d_v,
                                ),
                                dtype=query_states.dtype,
                                device=query_states.device,
                            ),
                        ],
                        dim=1,
                    )

                assert (
                    k_cache.shape[1] == self.rotary_embedding.end
                ), f"Cache size {k_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"
                assert (
                    v_cache.shape[1] == self.rotary_embedding.end
                ), f"Cache size {v_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"

                query_states = query_states.view(
                    batch_size, q_length, self.n_local_q_heads, self.d_qk
                )
                kv_length = key_states.shape[1]
                key_states = key_states.view(
                    batch_size, kv_length, self.n_local_kv_heads, self.d_qk
                )
                value_states = value_states.view(
                    batch_size, kv_length, self.n_local_kv_heads, self.d_v
                )

                softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
                attention_output = flash_attn_with_kvcache(
                    query_states,
                    k_cache,
                    v_cache,
                    key_states,
                    value_states,
                    rotary_cos=None,
                    rotary_sin=None,
                    cache_seqlens=position_offsets.contiguous(),
                    softmax_scale=softmax_scale,
                    causal=True,
                    rotary_interleaved=False,
                )

            store.update(
                {
                    "key": k_cache,
                    "value": v_cache,
                    "position_offsets": position_offsets,
                }
            )

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}

    def _forward_training(self, query_states, key_states, value_states, sequence_mask, batch_size, q_length):
        """训练模式前向传播，使用 FlashAttention 的融合 RoPE。

        训练时使用 flash_attn 的融合旋转位置编码实现，
        避免显式计算 cos/sin，提高计算效率。

        Args:
            query_states (torch.Tensor): 查询状态，形状为 [batch_size, seq_length, n_local_q_heads, d_qk]。
            key_states (torch.Tensor): 键状态，形状为 [batch_size, seq_length, n_local_kv_heads, d_qk]。
            value_states (torch.Tensor): 值状态，形状为 [batch_size, seq_length, n_local_kv_heads, d_v]。
            sequence_mask (torch.Tensor): 序列掩码，形状为 [batch_size, seq_length]。
            batch_size (int): 批大小。
            q_length (int): 查询序列长度。

        Returns:
            Dict[str, torch.Tensor]: 包含 "hidden_states" 和 "sequence_mask" 的字典。
        """
        key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
        key_states, value_states = torch.split(key_value_states, 1, dim=2)

        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        query_states = query_states.view(
            batch_size * q_length, self.n_local_q_heads, self.d_qk
        )

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}


class LlamaDecoderLayer(nn.Module):
    """LLaMA 解码器层，采用 Pre-Norm 架构。

    每层包含两个子模块，每个子模块前都有 RMSNorm：
        1. 自注意力: RMSNorm → CausalSelfAttention → 残差连接
        2. 前馈网络: RMSNorm → MLP → 残差连接

    支持梯度检查点（activation checkpointing）以减少内存占用。

    Attributes:
        input_layernorm (TritonRMSNorm): 注意力前的层归一化。
        attn (CausalSelfAttention): 因果自注意力模块。
        post_attention_layernorm (TritonRMSNorm): MLP 前的层归一化。
        mlp (MLP): 前馈网络模块。
        recompute_layer (bool): 是否对该层启用梯度检查点。
    """

    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        """初始化 LLaMA 解码器层。

        Args:
            config (LlamaConfig): LLaMA 模型配置。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
            tp_pg (dist.ProcessGroup): 张量并行进程组。
            layer_idx (int): 层索引。
        """
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        """核心前向计算（Pre-Norm 架构）。

        Args:
            hidden_states (Union[torch.Tensor, TensorPointer]): 输入隐藏状态。
            sequence_mask (Union[torch.Tensor, TensorPointer]): 序列掩码。

        Returns:
            List[Union[torch.Tensor, TensorPointer]]: [hidden_states, sequence_mask]。
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, output["sequence_mask"]

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """带梯度检查点的前向计算。

        使用 PyTorch 的 CheckpointFunction 在前向时不保存中间激活，
        反向时重新计算，以时间换空间。

        Args:
            hidden_states (torch.Tensor): 输入隐藏状态。
            sequence_mask (torch.Tensor): 序列掩码。

        Returns:
            List[torch.Tensor]: [hidden_states, sequence_mask]。
        """
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, sequence_mask)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """解码器层前向传播。

        当 recompute_layer 为 True 且输入不是 TensorPointer 时，
        使用梯度检查点以减少内存占用。

        Args:
            hidden_states (Union[torch.Tensor, TensorPointer]): 输入隐藏状态。
            sequence_mask (Union[torch.Tensor, TensorPointer]): 序列掩码。

        Returns:
            Dict[str, Union[torch.Tensor, TensorPointer]]: 包含 "hidden_states" 和 "sequence_mask" 的字典。
        """

        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, sequence_mask = self._checkpointed_forward(hidden_states, sequence_mask)
        else:
            hidden_states, sequence_mask = self._core_forward(hidden_states, sequence_mask)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,
        }


class Embedding(nn.Module, AttachableStore):
    """词嵌入层，将 token ID 映射为连续向量表示。

    使用 TensorParallelEmbedding 进行张量并行的词嵌入查找。
    在推理模式下，通过 AttachableStore 跟踪已处理的 token 数量。

    Attributes:
        token_embedding (TensorParallelEmbedding): 张量并行词嵌入层。
        pg (dist.ProcessGroup): 张量并行进程组。
    """

    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional[ParallelismArgs]):
        """初始化词嵌入层。

        Args:
            tp_pg (dist.ProcessGroup): 张量并行进程组。
            config (LlamaConfig): LLaMA 模型配置。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
        """
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        """词嵌入前向传播。

        Args:
            input_ids (torch.Tensor): 输入 token ID，形状为 [batch_size, seq_length]。
            input_mask (torch.Tensor): 输入掩码，形状为 [batch_size, seq_length]。

        Returns:
            Dict[str, torch.Tensor]: 包含 "input_embeds" 键的字典，
                嵌入形状为 [seq_length, batch_size, hidden_size]。
        """
        store = self.get_local_store()
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(input_ids.shape[0])

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            store["past_length"] = past_length + cumsum_mask[:, -1]

        input_ids = input_ids.transpose(0, 1)
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds}


class LlamaModel(nn.Module):
    """LLaMA 模型主体，构建流水线并行图。

    将模型的所有组件通过 PipelineBlock 包装，形成流水线并行的计算图。
    模型结构：Embedding → [DecoderLayer × N] → FinalLayerNorm → LM_Head → CastFP32

    每个 PipelineBlock 指定了模块构建器、输入/输出键名，
    由流水线引擎负责跨设备的数据传输和调度。

    Attributes:
        p2p (P2P): 流水线并行点对点通信器。
        config (LlamaConfig): LLaMA 模型配置。
        parallel_config (Optional[ParallelismArgs]): 并行配置。
        parallel_context (ParallelContext): 并行上下文。
        token_position_embeddings (PipelineBlock): 词嵌入层（PipelineBlock 包装）。
        decoder (nn.ModuleList): 解码器层列表（PipelineBlock 包装）。
        final_layer_norm (PipelineBlock): 最终层归一化（PipelineBlock 包装）。
        lm_head (PipelineBlock): 语言模型头（PipelineBlock 包装）。
        cast_to_fp32 (PipelineBlock): FP32 转换层（PipelineBlock 包装）。
    """

    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        """初始化 LLaMA 模型。

        Args:
            config (LlamaConfig): LLaMA 模型配置。
            parallel_context (ParallelContext): 并行上下文。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
        """
        super().__init__()

        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={"input_embeds"},
        )
        log_rank(f"Initialize RoPE Theta = {config.rope_theta}", logger=logger, level=logging.INFO, rank=0)
        if config.rope_interleaved:
            log_rank(
                "The RoPE interleaved version differs from the Transformers implementation. It's better to set rope_interleaved=False if you need to convert the weights to Transformers",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlamaDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "sequence_mask"},
                    module_output_keys={"hidden_states", "sequence_mask"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
    ):
        """LLaMA 模型前向传播。

        Args:
            input_ids (Union[torch.Tensor, TensorPointer]): 输入 token ID，形状为 [batch_size, seq_length]。
            input_mask (Union[torch.Tensor, TensorPointer]): 输入掩码，形状为 [batch_size, seq_length]。

        Returns:
            torch.Tensor: FP32 精度的分片 logits。
        """
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
    ):
        """LLaMA 模型前向传播，同时返回隐藏状态。

        Args:
            input_ids (Union[torch.Tensor, TensorPointer]): 输入 token ID，形状为 [batch_size, seq_length]。
            input_mask (Union[torch.Tensor, TensorPointer]): 输入掩码，形状为 [batch_size, seq_length]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - fp32_sharded_logits: FP32 精度的分片 logits。
                - hidden_states: 最后一层的隐藏状态。
        """
        output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

        hidden_encoder_states = {
            "hidden_states": output["input_embeds"],
            "sequence_mask": input_mask,
        }
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states

    def get_block_compute_costs(self):
        """计算模型中每种块的浮点计算量，用于流水线并行的负载均衡。

        Returns:
            Dict[type, float]: 块类型到计算量的映射。
                - LlamaDecoderLayer: 包含注意力（4 * heads * d_qkv * hidden）+ MLP（3 * d_ff * hidden）
                - TensorParallelColumnLinear: lm_head 的计算量（vocab * hidden）
        """
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """计算模型的每秒浮点运算量（FLOPS）。

        Args:
            iteration_time_in_sec (float): 单次迭代的耗时（秒）。
            sequence_length (int): 序列长度。
            global_batch_size (int): 全局批大小。

        Returns:
            Tuple[float, float]:
                - model_flops_per_s: 模型理论 FLOPS（单位: TFLOPS）。
                - hardware_flops_per_s: 硬件实际 FLOPS（单位: TFLOPS）。
        """
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    """计算掩码加权平均损失。

    Args:
        loss (torch.Tensor): 逐 token 的损失值。
        label_mask (torch.Tensor): 标签掩码，1 表示有效 token，0 表示忽略。
        dtype (torch.dtype): 计算精度。

    Returns:
        torch.Tensor: 掩码加权平均后的标量损失。
    """
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    """交叉熵损失计算模块，支持张量并行的分片 logits。

    使用 sharded_cross_entropy 在 TP 组内直接计算分片 logits 的交叉熵，
    避免先 AllGather 再计算的高内存开销。

    Attributes:
        tp_pg (dist.ProcessGroup): 张量并行进程组。
    """

    def __init__(self, tp_pg: dist.ProcessGroup):
        """初始化损失模块。

        Args:
            tp_pg (dist.ProcessGroup): 张量并行进程组。
        """
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,
        label_ids: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算交叉熵损失。

        Args:
            sharded_logits (torch.Tensor): 分片 logits，形状为 [seq_length, batch_size, vocab_size/TP]。
            label_ids (torch.Tensor): 标签 ID，形状为 [batch_size, seq_length]。
            label_mask (torch.Tensor): 标签掩码，形状为 [batch_size, seq_length]。

        Returns:
            Dict[str, torch.Tensor]: 包含 "loss" 键的字典。
        """
        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
        ).transpose(0, 1)
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        return {"loss": loss}


class LossWithZLoss(Loss):
    """带 Z-loss 正则化的交叉熵损失。

    Z-loss 鼓励 logits 保持较小的数值，有助于训练稳定性。
    参考: https://arxiv.org/abs/2305.18290

    total_loss = cross_entropy_loss + z_loss_coef * z_loss

    Attributes:
        z_loss_coef (float): Z-loss 的系数。
    """

    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float):
        """初始化带 Z-loss 的损失模块。

        Args:
            tp_pg (dist.ProcessGroup): 张量并行进程组。
            z_loss_coefficient (float): Z-loss 系数。
        """
        super().__init__(tp_pg)
        self.z_loss_coef = z_loss_coefficient

    def forward(
        self,
        sharded_logits: torch.Tensor,
        label_ids: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算带 Z-loss 的交叉熵损失。

        Args:
            sharded_logits (torch.Tensor): 分片 logits，形状为 [seq_length, batch_size, vocab_size/TP]。
            label_ids (torch.Tensor): 标签 ID，形状为 [batch_size, seq_length]。
            label_mask (torch.Tensor): 标签掩码，形状为 [batch_size, seq_length]。

        Returns:
            Dict[str, torch.Tensor]: 包含 "loss" 和 "z_loss" 键的字典。
        """
        loss, z_loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
            z_loss_coef=self.z_loss_coef,
        )
        loss = masked_mean(loss.transpose(0, 1), label_mask, dtype=torch.float)
        z_loss = masked_mean(z_loss.detach().transpose(0, 1), label_mask, dtype=torch.float)
        return {"loss": loss, "z_loss": z_loss}


class LlamaForTraining(NanotronModel):
    """LLaMA 训练模型，NanotronModel 的具体实现。

    将 LlamaModel 和 Loss 组合在一起，提供完整的训练前向传播。
    损失计算也通过 PipelineBlock 包装，以支持流水线并行。

    继承自 NanotronModel，必须实现 init_model_randomly() 方法。

    Attributes:
        model (LlamaModel): LLaMA 模型主体。
        loss (PipelineBlock): 损失计算模块（PipelineBlock 包装）。
        parallel_context (ParallelContext): 并行上下文。
        config (LlamaConfig): LLaMA 模型配置。
        parallel_config (Optional[ParallelismArgs]): 并行配置。
    """

    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        """初始化 LLaMA 训练模型。

        Args:
            config (LlamaConfig): LLaMA 模型配置。
            parallel_context (ParallelContext): 并行上下文。
            parallel_config (Optional[ParallelismArgs]): 并行配置。
            random_states (Optional[RandomStates]): 随机状态管理器。
        """
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        loss_kwargs = {
            "tp_pg": parallel_context.tp_pg,
        }
        if config.z_loss_enabled:
            loss_kwargs["z_loss_coefficient"] = config.z_loss_coefficient

        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=LossWithZLoss if config.z_loss_enabled else Loss,
            module_kwargs=loss_kwargs,
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss", "z_loss"} if config.z_loss_enabled else {"loss"},
        )

        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """训练前向传播。

        Args:
            input_ids (Union[torch.Tensor, TensorPointer]): 输入 token ID。
            input_mask (Union[torch.Tensor, TensorPointer]): 输入掩码。
            label_ids (Union[torch.Tensor, TensorPointer]): 标签 ID。
            label_mask (Union[torch.Tensor, TensorPointer]): 标签掩码。

        Returns:
            Dict[str, Union[torch.Tensor, TensorPointer]]:
                包含 "loss" 的字典，如果启用 Z-loss 还包含 "z_loss"。
        """
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """随机初始化模型参数。

        根据配置选择参数化策略：
            - RandomInit: 标准参数化（StandardParametrizator）
            - SpectralMupInit: Spectral µP 参数化（SpectralMupParametrizator）

        处理绑定参数（tied parameters）时，确保只初始化一次。

        Args:
            config (Config): 全局配置，包含模型初始化方法。
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config.model)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_embeddings_lm_head_tied_names(self):
        """获取绑定词嵌入和语言模型头的参数名称。

        当 tie_word_embeddings 为 True 时，词嵌入权重和 lm_head 权重共享，
        返回两者的参数名称列表。

        Returns:
            List[str]: 绑定参数的名称列表，空列表表示不绑定。
        """
        if self.config.tie_word_embeddings is True:
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_block_compute_costs(self):
        """获取模型各块的浮点计算量，用于流水线并行负载均衡。

        Returns:
            Dict[type, float]: 块类型到计算量的映射。
        """
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """获取模型的每秒浮点运算量。

        Args:
            iteration_time_in_sec (float): 单次迭代的耗时（秒）。
            sequence_length (int): 序列长度。
            global_batch_size (int): 全局批大小。

        Returns:
            Tuple[float, float]: (模型理论 FLOPS, 硬件实际 FLOPS)，单位为 TFLOPS。
        """
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    """计算解码器模型的浮点运算量（FLOPs）。

    模型 FLOPs 包括前向和反向传播的计算量。
    反向传播的计算量约为前向的 2 倍（需要计算对输入和权重的梯度），
    因此总 FLOPs = 3 × 前向 FLOPs。

    计算分解：
        - QKV 投影: 2 × L × B × S × H × (Q_heads + 2 × KV_heads) × d_head
        - QK 注意力: 2 × L × B × Q_heads × S × d_head × S
        - V 注意力: 2 × L × B × Q_heads × S × S × d_head
        - 输出投影: 2 × L × B × Q_heads × S × d_head × H
        - FFN 第 1 层: 4 × L × B × S × H × d_ff （gate_up_proj 的 2 倍）
        - FFN 第 2 层: 2 × L × B × S × d_ff × H
        - LM Head: 2 × B × S × H × V

    Args:
        num_layers (int): 解码器层数。
        hidden_size (int): 隐藏维度。
        num_heads (int): 注意力头数。
        num_key_value_heads (int): 键值头数（GQA）。
        vocab_size (int): 词表大小。
        seq_len (int): 序列长度。
        ffn_hidden_size (int): FFN 中间维度。
        batch_size (int, optional): 批大小。默认为 1。

    Returns:
        Tuple[float, float]:
            - model_flops: 模型理论 FLOPs（与硬件和实现无关）。
            - hardware_flops: 硬件实际 FLOPs（参考: https://arxiv.org/pdf/2205.05198.pdf 6.3 节）。
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)

    hardware_flops = model_flops

    return model_flops, hardware_flops
