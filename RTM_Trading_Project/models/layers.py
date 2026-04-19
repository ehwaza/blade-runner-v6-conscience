from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import einops
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a: int, b: int) -> int:
    # ceil(a/b) * b
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim halves (x1, x2) -> (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, S, H, D], cos/sin: [S, D]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros((out_features, ))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype),
                        bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float, device=None):
        super().__init__()
        # S'assurer que dim est pair
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False):
        super().__init__()
        self.hidden_size = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # Calcul des dimensions
        self.q_dim = num_heads * head_dim
        self.kv_dim = num_key_value_heads * head_dim
        self.total_qkv_dim = self.q_dim + 2 * self.kv_dim
        
        # Vérification que les dimensions sont cohérentes
        assert dim == self.q_dim, f"Hidden size {dim} must equal q_dim {self.q_dim}"
        
        self.qkv_proj = CastedLinear(dim, self.total_qkv_dim, bias=False)
        self.o_proj = CastedLinear(self.q_dim, dim, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S, D]
        B, S, D = hidden_states.shape
        
        # Projection QKV
        qkv = self.qkv_proj(hidden_states)  # [B, S, total_qkv_dim]
        
        # Séparation Q, K, V
        q = qkv[:, :, :self.q_dim]  # [B, S, H*Dh]
        k = qkv[:, :, self.q_dim:self.q_dim + self.kv_dim]  # [B, S, KVH*Dh]
        v = qkv[:, :, self.q_dim + self.kv_dim:]  # [B, S, KVH*Dh]
        
        # Reshape pour avoir les têtes séparées
        q = q.view(B, S, self.num_heads, self.head_dim)  # [B, S, H, Dh]
        k = k.view(B, S, self.num_key_value_heads, self.head_dim)  # [B, S, KVH, Dh]
        v = v.view(B, S, self.num_key_value_heads, self.head_dim)  # [B, S, KVH, Dh]

        # Application rotary embedding si fourni
        if cos_sin is not None:
            cos, sin = cos_sin
            # S'assurer que cos/sin ont la bonne dimension
            cos = cos[:S]  # [S, D]
            sin = sin[:S]  # [S, D]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Réorganisation pour l'attention: [B, H, S, Dh]
        q = einops.rearrange(q, 'B S H D -> B H S D')
        k = einops.rearrange(k, 'B S H D -> B H S D')
        v = einops.rearrange(v, 'B S H D -> B H S D')

        # Attention - utiliser la même clé/valeur pour toutes les têtes de requête si nécessaire
        if self.num_heads != self.num_key_value_heads:
            # Répéter les têtes K/V pour correspondre aux têtes Q
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # Attention
        attn_output = scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            is_causal=self.causal and S > 1
        )

        # Réorganisation retour: [B, S, H, Dh] -> [B, S, H*Dh]
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(B, S, self.q_dim)
        
        # Projection de sortie
        return self.o_proj(attn_output)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse: bool = False):
        super().__init__()
        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(self.linear(x)) if self.reverse else self.linear(F.silu(x)))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """RMSNorm without learned weight (since many blocks do post-norm + residual)."""
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    var = x.square().mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + variance_epsilon)
    return x.to(input_dtype)