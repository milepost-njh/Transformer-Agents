import math
import sys
from collections.abc import Callable
from typing import Any, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

# Mock the problematic torchao module before importing transformers
# This prevents the ModuleNotFoundError from torchao.prototype.safetensors
try:
    import torchao.prototype.safetensors.safetensors_utils
except ImportError:
    # Create a mock module structure
    if 'torchao' not in sys.modules:
        sys.modules['torchao'] = MagicMock()
    if 'torchao.prototype' not in sys.modules:
        sys.modules['torchao.prototype'] = MagicMock()
    if 'torchao.prototype.safetensors' not in sys.modules:
        sys.modules['torchao.prototype.safetensors'] = MagicMock()
    if 'torchao.prototype.safetensors.safetensors_utils' not in sys.modules:
        mock_module = MagicMock()
        mock_module.is_metadata_torchao = MagicMock(return_value=False)
        sys.modules['torchao.prototype.safetensors.safetensors_utils'] = mock_module

import transformers
from einops import rearrange
from packaging import version
from torch import nn
from loguru import logger
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except (ImportError, AttributeError):
    ALL_ATTENTION_FUNCTIONS = None

try:
    from transformers.processing_utils import Unpack
except ImportError:
    # Fallback for older Python or transformers versions
    from typing import Unpack

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (TransformersKwargs, auto_docstring,
                                can_return_tuple, logging)
from transformers.utils.generic import OutputRecorder, check_model_inputs

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    raise ImportError("Plese run `pip install -U fla-core`")

# Utility functions for handling variable-length sequences
def get_unpad_data(attention_mask):
    """
    Get indices, cumulative sequence lengths, and maximum sequence length from attention mask.
    
    Args:
        attention_mask: Binary mask of shape [batch_size, seq_len] where 1 = valid token, 0 = padding
        
    Returns:
        indices: Indices of valid (non-padded) tokens
        cu_seqlens: Cumulative sequence lengths
        max_seqlen_in_batch: Maximum sequence length in the batch
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    return indices, cu_seqlens, max_seqlen_in_batch


def index_first_axis(x, indices):
    """
    Index the first axis of a tensor.
    
    Args:
        x: Tensor of shape [total_tokens, ...]
        indices: Indices to select
        
    Returns:
        Selected tensor
    """
    return x[indices]


def pad_input(hidden_states, indices, batch_size, seqlen):
    """
    Pad the hidden states back to the original shape.
    
    Args:
        hidden_states: Unpadded hidden states of shape [total_valid_tokens, hidden_dim]
        indices: Indices of valid tokens
        batch_size: Original batch size
        seqlen: Original sequence length
        
    Returns:
        Padded hidden states of shape [batch_size, seqlen, hidden_dim]
    """
    dim = hidden_states.shape[-1]
    output = torch.zeros(
        batch_size * seqlen, dim, dtype=hidden_states.dtype, device=hidden_states.device
    )
    output[indices] = hidden_states
    return rearrange(output, "(b s) d -> b s d", b=batch_size)

# Handle both relative and absolute imports
try:
    from .configuration_kimi import KimiLinearConfig
except ImportError:
    from configuration_kimi import KimiLinearConfig

assert version.parse(transformers.__version__) >= version.parse("4.56.0"), \
    "Please upgrade transformers to >= 4.56.0"

logger = logging.get_logger(__name__)


class KimiDynamicCache:
    """
    Dynamic cache for Kimi model.
    Inspired by Qwen3-Next
    """
    is_compileable = False

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config

        if config.linear_attn_config is not None:
            self.layer_types = []
            for i in range(config.num_hidden_layers):
                if config.is_kda_layer(i):
                    self.layer_types.append("linear_attention")
                else:
                    self.layer_types.append("full_attention")
        else:
            self.layer_types = ["full_attention"] * config.num_hidden_layers

        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]

        linear_layers = [i for i in range(
            config.num_hidden_layers) if self.layer_types[i] == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else -1

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx),
                    k_conv.index_select(0, beam_idx),
                    v_conv.index_select(0, beam_idx)
                )
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(
                    0, beam_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        if self.last_linear_layer == -1:
            return False
        return self.conv_states[self.last_linear_layer] is not None


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        KimiRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(KimiRMSNorm)


class KimiBlockSparseMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)   # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # up

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(
            self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class KimiMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class KimiMLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.rope_theta = config.rope_theta
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(
                f"Kimi MLA config is not found or not properly formatted: {e}")

        assert self.q_lora_rank is None
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.q_head_dim, bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.is_causal = True
        assert self.use_nope

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1,
                     self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(
            k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx)

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            value_states = F.pad(
                value_states, [0, self.q_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if ALL_ATTENTION_FUNCTIONS is not None:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            else:
                # Fallback to eager if ALL_ATTENTION_FUNCTIONS is not available
                attention_interface = eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in [
            'chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu'
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation='silu'
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(
            self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[KimiDynamicCache] = None,
        **kwargs: Unpack[dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask", None)

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here."
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens', None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[
                    self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k = map(lambda x: rearrange(
            x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (
                conv_state_q, conv_state_k, conv_state_v)

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim))
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty((self.num_experts))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(
                torch.float32), None
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}"
            )

        # select top-k experts
        # Training mode is now supported
        scores_for_choice = scores.view(bsz * seq_len, -1)
        scores_for_choice = scores_for_choice + self.e_score_correction_bias.unsqueeze(0)  # 避免 inplace 操作破坏梯度
        group_scores = (
            scores_for_choice.view(
                bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, num_expert_group]
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len, self.num_expert_group, self.num_experts // self.num_expert_group
            )
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0)  # [n, e]
        _, topk_idx = torch.topk(
            tmp_scores, k=self.top_k, dim=-1, sorted=False
        )
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V3's MOE implementation
    The namings are consistent with Kimi's version.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.ep_size = 1
        self.experts_per_rank = config.num_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(config.num_experts)
            ]
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx,
                               topk_weight).view(*orig_shape)
        else:
            # Training mode: use gradient-enabled forward
            y = self.moe_train(hidden_states, topk_idx,
                              topk_weight).view(*orig_shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_train(self, x, topk_ids, topk_weight):
        """
        Training-mode forward for MoE block with gradient support.
        """
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        # Keep tokens_per_expert on GPU for gradient flow
        tokens_per_expert_list = tokens_per_expert.long().tolist()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_list):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(
            outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(
            outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = KimiDeltaAttention(
                config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = KimiMLAAttention(
                config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class KimiPreTrainedModel(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(KimiBlockSparseMLP, index=1),
        "hidden_states": KimiDecoderLayer,
        "attentions": KimiMLAAttention,
    }
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KimiLinearModel(KimiPreTrainedModel):
    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([KimiDecoderLayer(
            config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "_attn_implementation", None) is not None:
            if config._attn_implementation != "flash_attention_2":
                logger.warning_once(
                    f"Ignoring the provided attention implementation {config._attn_implementation}")
                logger.warning_once("Using flash_attention_2 backend instead.")
                config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "flash_attention_2"

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        debug_mode: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if debug_mode:
            logger.warning("🔍 [KimiLinearModel] forward 开始")
            if input_ids is not None:
                logger.warning(f"  input_ids: {input_ids.shape}")

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        # Get inputs_embeds
        # input_ids: (batch_size, seq_len) -> inputs_embeds: (batch_size, seq_len, hidden_size)
        # 例如：(26, 74) -> (26, 74, 512)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (batch_size, seq_len, hidden_size) 例如 (26, 74, 512)
        
        if debug_mode:
            logger.warning(f"  inputs_embeds: {inputs_embeds.shape}")

        if use_cache and past_key_values is None:
            past_key_values = KimiDynamicCache(config=self.config)

        # ============================================================================
        # 计算 cache_position（当前输入在完整序列中的位置索引）
        # ============================================================================
        # 用于增量生成（KV-cache）：记录当前 token 在整个序列中的绝对位置
        # 
        # 工作流程：
        #   1. 获取已缓存的序列长度（past_seen_tokens）
        #      - 训练时：past_key_values=None，past_seen_tokens=0（从头开始）
        #      - 推理时（使用 cache）：past_seen_tokens=之前生成的 token 数量
        #   
        #   2. 生成当前输入的位置索引
        #      - cache_position = [past_seen_tokens, past_seen_tokens+1, ..., past_seen_tokens+seq_len-1]
        #      - 例如：如果已生成 10 个 token，当前输入 5 个，则 cache_position = [10, 11, 12, 13, 14]
        #      - 训练时（past_seen_tokens=0, seq_len=74）：cache_position = [0, 1, 2, ..., 73]（形状 (74,)）
        #
        # 为什么需要 cache_position？
        #   - RoPE 位置编码需要知道每个 token 的绝对位置
        #   - 创建 mask 时需要知道当前处理的是序列的哪一部分
        # ============================================================================
        if cache_position is None:
            # 获取已缓存的序列长度（训练时为 0，推理时为已生成的 token 数）
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0  # -> int
            
            # 生成当前输入的绝对位置索引：[past_seen_tokens, past_seen_tokens+1, ..., past_seen_tokens+seq_len-1]
            cache_position: torch.Tensor = torch.arange(  # (seq_len,)
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )  # -> (seq_len,)

        if position_ids is None:
            # cache_position: (seq_len,) 例如 (74,) -> position_ids: (1, seq_len) 例如 (1, 74)
            # 在实际计算时会广播到 (batch_size, seq_len) 例如 (26, 74)
            position_ids = cache_position.unsqueeze(0)  # (seq_len,) -> (1, seq_len) 例如 (74,) -> (1, 74)

        # ============================================================================
        # 创建因果 mask（Causal Mask）- 用于自回归语言模型
        # ============================================================================
        # 作用：确保每个位置只能看到当前及之前的 token，不能看到未来的 token
        #      这是自回归生成的核心机制（防止信息泄露）
        #
        # 输入参数：
        #   - input_embeds: (batch_size, seq_len, hidden_size) 例如 (26, 74, 512)
        #   - attention_mask: (batch_size, seq_len) 例如 (26, 74) - 标记哪些位置是 padding（1=有效，0=padding）
        #   - position_ids: (batch_size, seq_len) 例如 (26, 74) - 每个 token 的位置索引 [0, 1, 2, ..., 73]
        #   - cache_position: (seq_len,) 例如 (74,) - 当前序列在 cache 中的位置索引
        #
        # 输出：causal_mask (batch_size, 1, seq_len, seq_len) 或 None
        #   - 4D tensor 例如 (26, 1, 74, 74)，其中第二维是 1（会广播到所有注意力头）
        #   - mask[b, :, i, j] 表示样本 b 中位置 i 是否能看到位置 j
        #   - 值为 0（可见）或 -inf（不可见，会被 softmax 过滤掉）
        #
        # Mask 创建过程（内部逻辑）：
        #   1. 创建基础因果 mask：上三角矩阵，确保 i 只能看到 j <= i 的位置
        #      对于 seq_len=74 的序列，生成 (74, 74) 的矩阵
        #      简化示例（取前4个位置）：
        #        [[0, 1, 1, 1],    位置0只能看到自己
        #         [0, 0, 1, 1],    位置1能看到0和1
        #         [0, 0, 0, 1],    位置2能看到0,1,2
        #         [0, 0, 0, 0]]    位置3能看到0,1,2,3（所有历史）
        #      实际是 (74, 74) 的矩阵，遵循相同规律
        #
        #   2. 结合 attention_mask（padding mask）：
        #      假设 attention_mask = [1, 1, ..., 1, 0, 0, 0]（前 54 个有效，后 20 个是 padding）
        #      则 padding 位置（54-73）在 mask 中被设为 -inf，表示任何位置都不能看到它们
        #      同时 padding 位置自己也看不到任何东西（整行都是 -inf）
        #
        #   3. 转换为 4D：(batch_size, 1, seq_len, seq_len) 例如 (26, 1, 74, 74)
        #      - batch_size=26：每个样本独立的 mask
        #      - 第二维=1：会在注意力计算时广播到所有头（例如 8 个头）
        #      - (74, 74)：每个位置对所有位置的可见性
        #
        # 特殊情况：
        #   - 如果使用 Flash Attention，可能返回 None（Flash Attention 内部处理因果性）
        #   - 如果所有 token 都有效且没有 cache，某些实现会优化为 is_causal=True 而不是显式 mask
        # ============================================================================
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,  # (26, 74, 512)
            attention_mask=attention_mask,  # (26, 74)
            cache_position=cache_position,  # (74,)
            past_key_values=past_key_values,
            position_ids=position_ids,  # (26, 74)
        )  # -> (26, 1, 74, 74) 或 None
        
        # ============================================================================
        # 更新线性注意力 mask（Linear Attention Mask）- 用于线性注意力层
        # ============================================================================
        # 作用：线性注意力层（KDA - Kernel Decomposed Attention）使用更简单的 2D mask
        #      不需要完整的 4D 因果 mask，因为线性注意力的计算方式不同
        #
        # 输入参数：
        #   - attention_mask: (batch_size, seq_len) 例如 (26, 74) - 标记 padding 位置（1=有效，0=padding）
        #   - cache_position: (seq_len,) 例如 (74,) - 当前位置索引
        #
        # 输出：linear_attn_mask (batch_size, seq_len) 或 None
        #   - 2D mask 例如 (26, 74)，直接标记哪些位置需要被忽略
        #   - None 表示不需要 mask（优化情况）：
        #     * 如果使用了 cache（cache_position[0] > 0），说明是增量生成，不需要 mask
        #     * 如果所有位置都是有效 token（attention_mask 全为 1），不需要 padding mask
        #
        # 为什么线性注意力只需要 2D mask？
        #   - 线性注意力（如 KDA）使用线性复杂度的计算方式
        #   - 不需要计算完整的注意力矩阵（seq_len × seq_len）
        #   - 只需要知道哪些位置是 padding，在计算时跳过即可
        # ============================================================================
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, cache_position)  # -> (26, 74) 或 None
        

        hidden_states = inputs_embeds  # (batch_size, seq_len, hidden_size) 例如 (26, 74, 512)
        if past_key_values is not None:
            assert isinstance(past_key_values, KimiDynamicCache)

        # 逐层处理：hidden_states: (batch_size, seq_len, hidden_size) 例如 (26, 74, 512)
        for layer_idx, decoder_layer in enumerate(self.layers):
            # 根据层类型选择 mask：线性注意力层用 linear_attn_mask，全注意力层用 causal_mask
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask

            # decoder_layer 输入输出都是 (batch_size, seq_len, hidden_size)
            hidden_states = decoder_layer(
                hidden_states,  # (26, 74, 512)
                attention_mask=layer_mask,  # (26, 1, 74, 74) 或 (26, 74) 或 None
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )  # -> (26, 74, 512)
        
        if debug_mode:
            logger.warning(f"  hidden_states (after {len(self.layers)} layers): {hidden_states.shape}")

        # 最终的 RMSNorm：(batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)  # (26, 74, 512) -> (26, 74, 512)
        
        if debug_mode:
            logger.warning(f"  hidden_states (after {len(self.layers)} layers + norm): {hidden_states.shape}")
            logger.warning("🔍 [KimiLinearModel] forward 完成")
            assert 1 == 0, "Debug mode: 在 KimiLinearModel.forward() 结束前中断"

        # 返回最后的 hidden_states: (batch_size, seq_len, hidden_size) 例如 (26, 74, 512)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,  # (26, 74, 512)
            past_key_values=past_key_values,
        )


class KimiLinearForCausalLM(KimiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = KimiLinearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        debug_mode: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example of log:

       """
        # ============================================================================
        # 输入数据说明（forward 函数接收到的数据）
        # ============================================================================
        # 注意：数据已经在 DataLoader 的 collate_fn 中完成了 padding 处理
        # 
        # 输入张量形状（以 DataParallel 分片后的真实数据为例）：
        #   - input_ids: (batch_size, seq_len) 例如 (26, 74)
        #     * batch_size=26: DataParallel 分片后的数量（原始 batch_size=128，分到 5 个 GPU）
        #     * seq_len=74: 经过 padding 后的序列长度（动态 padding，每个 batch 可能不同）
        #     * padding 位置的值 = pad_token_id（通常是 1）
        #
        #   - attention_mask: (batch_size, seq_len) 例如 (26, 74)
        #     * 形状与 input_ids 相同
        #     * 1 表示真实 token，0 表示 padding 位置
        #     * 例如：[1,1,1,...,1,0,0,0] 表示前 54 个是真实 token，后 20 个是 padding
        #
        #   - labels: (batch_size, seq_len) 例如 (26, 74) 或 None
        #     * 形状与 input_ids 相同
        #     * 包含真实 token id 或 -100（用于 mask）
        #     * -100 的位置会被损失函数忽略（PyTorch CrossEntropyLoss 的 ignore_index）
        #     * 在翻译任务中：
        #       - prompt 部分（前 34 个位置）：-100（不计算损失）
        #       - 英文部分（接下来 6 个位置）：真实 token id（计算损失）
        #       - padding 部分（最后 34 个位置）：-100（不计算损失）
        #     * 例如：[-100,-100,...,-100,6866,299,...,2342,-100,-100,...,-100]
        #
        # 输出 logits 形状：
        #   - logits: (batch_size, seq_len, vocab_size) 例如 (26, 74, 8192)
        #     * 每个位置 i 的 logits 预测该位置的 token（因果语言模型）
        #     * logits[b, i, :] 是位置 i 在词表上的概率分布（未归一化）
        #     * 在训练时，logits[b, i, :] 与 labels[b, i] 计算交叉熵
        #
        # 损失计算：
        #   - 使用 logits 和 labels 计算交叉熵损失
        #   - labels 中 -100 的位置会被自动忽略
        #   - 只对非 -100 的位置（即英文部分）计算损失
        # ============================================================================

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 KimiLinearModel
        # input_ids: (batch_size, seq_len) 例如 (26, 74)
        # attention_mask: (batch_size, seq_len) 例如 (26, 74)
        outputs = self.model(
            input_ids=input_ids,  # (26, 74)
            attention_mask=attention_mask,  # (26, 74)
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # 注意：不传递 debug_mode，让 KimiLinearModel 的 debug_mode 独立控制
        )  # -> last_hidden_state: (26, 74, 512)

        # outputs[0] 即 last_hidden_state: (batch_size, seq_len, hidden_size) 例如 (26, 74, 512)
        logits = outputs[0]  # (26, 74, 512)
        if generation_mode:
            logits = logits[:, -1:]  # 只取最后一个位置 (26, 1, 512)
        # lm_head: 线性层投影到词表大小
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(logits)  # (26, 74, 512) -> (26, 74, 8192)

        # Debug mode: 打印 logits 形状并中断
        if debug_mode:
            logger.warning(f"🔍 [KimiLinearForCausalLM] logits: {logits.shape}, vocab_size={self.vocab_size}")
            logger.warning(f"  ⚠️ 注意：使用 DataParallel 时 batch_size 是分片后的（原始128可能分为24/26等）")
            assert 1 == 0, "Debug mode: 在 KimiLinearForCausalLM.forward() 中断"

        loss = None
        if labels is not None:
            # 计算损失
            # logits: (batch_size, seq_len, vocab_size) 例如 (26, 74, 8192)
            # labels: (batch_size, seq_len) 例如 (26, 74)，其中 -100 位置会被忽略
            # 损失函数会：
            #   1. 对每个位置计算交叉熵：比较 logits[b, i, :] 和 labels[b, i]
            #   2. 忽略 labels[b, i] = -100 的位置（prompt 和 padding）
            #   3. 只对有效位置（英文部分）计算平均损失
            loss = self.loss_function(
                logits, labels, self.vocab_size, **kwargs)  # -> scalar

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    """
    调试入口：通过 KimiLinearForCausalLM 贯穿所有核心组件
    执行路径：
        KimiLinearForCausalLM
        └── KimiLinearModel
            ├── embed_tokens (Embedding)
            ├── layers (多个 KimiDecoderLayer)
            │   ├── input_layernorm (KimiRMSNorm)
            │   ├── self_attn (KimiMLAAttention 或 KimiDeltaAttention)
            │   ├── post_attention_layernorm (KimiRMSNorm)
            │   └── block_sparse_moe (KimiSparseMoeBlock)
            │       ├── gate (KimiMoEGate)
            │       ├── experts (KimiBlockSparseMLP)
            │       └── shared_experts (KimiMLP)
            └── norm (KimiRMSNorm)
            └── lm_head (Linear)
    """
    
    # ===== 配置 =====
    device = torch.device("cuda")
    attn_implementation = "flash_attention_2"
    dtype = torch.bfloat16
    
    print(f"Device: {device} | Attention: {attn_implementation} | Dtype: {dtype}")
    
    # ===== 配置参数 =====
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    config = KimiLinearConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        intermediate_size=2048,
        hidden_act="silu",
        q_lora_rank=None,
        kv_lora_rank=4 * head_dim,
        qk_nope_head_dim=head_dim // 2,
        qk_rope_head_dim=head_dim // 2,
        v_head_dim=head_dim,
        mla_use_nope=True,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=1536,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=2,
        routed_scaling_factor=1.0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        use_cache=False,
        _attn_implementation=attn_implementation,
    )
    
    # ===== 创建模型 =====
    model = KimiLinearForCausalLM(config)
    model.eval()
    model.to(device=device, dtype=dtype)
    
    # ===== 测试完整前向传播 =====
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    print(f"\n✅ Full model test passed!")
    print(f"   Input: {input_ids.shape} → Output: {logits.shape}")
    
    # ===== 单独测试各个组件 =====
    print("\n📦 Testing individual components...")
    
    # 1. 测试 MLA Attention
    mla_layer = KimiMLAAttention(config=config, layer_idx=0).to(device=device, dtype=dtype)
    mla_layer.eval()
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
    with torch.no_grad():
        mla_output = mla_layer(hidden_states=hidden_states)
    print(f"   ✓ MLA Attention: {hidden_states.shape} → {mla_output.shape}")
    
    # 2. 测试 Decoder Layer
    decoder_layer = KimiDecoderLayer(config=config, layer_idx=0).to(device=device, dtype=dtype)
    decoder_layer.eval()
    with torch.no_grad():
        decoder_output = decoder_layer(hidden_states=hidden_states)
    print(f"   ✓ Decoder Layer: {hidden_states.shape} → {decoder_output[0].shape}")
    
    # 3. 测试 MoE Block
    moe_block = KimiSparseMoeBlock(config=config).to(device=device, dtype=dtype)
    moe_block.eval()
    with torch.no_grad():
        moe_output = moe_block(hidden_states)
    print(f"   ✓ MoE Block: {hidden_states.shape} → {moe_output[0].shape}")
    
    print("\n🎉 All tests passed! Ready for debugging!")
