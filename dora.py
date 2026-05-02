import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.base = base
        self.scaling = alpha / r

        self.base.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        W = base.weight.detach()
        self.magnitude = nn.Parameter(W.norm(p=2, dim=1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.base.weight
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        W_adapted = W + lora_update

        W_norm = W_adapted.norm(p=2, dim=1, keepdim=True)
        W_dora = (self.magnitude / W_norm) * W_adapted

        return F.linear(x, W_dora, self.base.bias)


class DoRALlamaMHA(nn.Module):
    def __init__(self, original_attn, rotary_emb, r: int, alpha: float = 16.0):
        super().__init__()

        cfg = original_attn.config
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = DoRALinear(original_attn.q_proj, r, alpha)
        self.k_proj = DoRALinear(original_attn.k_proj, r, alpha)
        self.v_proj = DoRALinear(original_attn.v_proj, r, alpha)
        self.o_proj = DoRALinear(original_attn.o_proj, r, alpha)

        self.rotary_emb = rotary_emb
        self.layer_idx: Optional[int] = getattr(original_attn, "layer_idx", None)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        B, num_kv_heads, S, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, num_kv_heads, n_rep, S, head_dim)
            .reshape(B, num_kv_heads * n_rep, S, head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_value.update(K, V, self.layer_idx, cache_kwargs)

        K = self._repeat_kv(K, self.num_kv_groups)
        V = self._repeat_kv(V, self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


def apply_dora_to_llama(model: LlamaForCausalLM, r: int = 16, alpha: float = 32.0):
    rotary_emb = (
        getattr(model.model, "rotary_emb", None) or
        getattr(model.model.layers[0].self_attn, "rotary_emb", None) or
        getattr(model.model.layers[0], "rotary_emb", None)
    )
    for layer_idx, layer in enumerate(model.model.layers):
        dora_attn = DoRALlamaMHA(layer.self_attn, rotary_emb, r=r, alpha=alpha)
        dora_attn.layer_idx = layer_idx
        layer.self_attn = dora_attn
    return model


def dora_trainable_params(model) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if p.requires_grad]


def dora_param_count(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100 * trainable / total, 4),
    }
