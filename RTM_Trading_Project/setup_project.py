"""
Script d'installation automatique du projet RTM Trading
Crée la structure de dossiers et tous les fichiers nécessaires
"""

import os
from pathlib import Path

def create_project_structure():
    """Crée la structure complète du projet"""
    
    print("=" * 70)
    print("🚀 INSTALLATION DU PROJET RTM TRADING")
    print("=" * 70)
    
    # Créer le dossier principal
    project_dir = Path("RTM_Trading_Project")
    project_dir.mkdir(exist_ok=True)
    
    # Créer le dossier models
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Création de la structure dans: {project_dir.absolute()}")
    
    # ========== FICHIER 1: models/__init__.py ==========
    print("\n1️⃣ Création de models/__init__.py...")
    init_content = """# Models package"""
    
    with open(models_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(init_content)
    print("   ✅ models/__init__.py créé")
    
    # ========== FICHIER 2: models/common.py ==========
    print("\n2️⃣ Création de models/common.py...")
    common_content = """import math

import torch
from torch import nn


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor
"""
    
    with open(models_dir / "common.py", "w", encoding="utf-8") as f:
        f.write(common_content)
    print("   ✅ models/common.py créé")
    
    # ========== FICHIER 3: models/ema.py ==========
    print("\n3️⃣ Création de models/ema.py...")
    ema_content = """import copy
import torch.nn as nn

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
"""
    
    with open(models_dir / "ema.py", "w", encoding="utf-8") as f:
        f.write(ema_content)
    print("   ✅ models/ema.py créé")
    
    # ========== FICHIER 4: models/layers.py ==========
    print("\n4️⃣ Création de models/layers.py...")
    layers_content = """from typing import Tuple
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
    \"\"\"Rotate last-dim halves (x1, x2) -> (-x2, x1).\"\"\"
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
        self.output_size = head_dim * num_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S, D]
        B, S, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)  # [B, S, (H + 2*KVH)*Dh]
        qkv = qkv.view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # [B, H, S, Dh]
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(B, S, self.output_size)
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
    \"\"\"RMSNorm without learned weight (since many blocks do post-norm + residual).\"\"\"
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    var = x.square().mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + variance_epsilon)
    return x.to(input_dtype)
"""
    
    with open(models_dir / "layers.py", "w", encoding="utf-8") as f:
        f.write(layers_content)
    print("   ✅ models/layers.py créé")
    
    # ========== FICHIER 5: models/losses.py ==========
    print("\n5️⃣ Création de models/losses.py...")
    losses_content = """from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
"""
    
    with open(models_dir / "losses.py", "w", encoding="utf-8") as f:
        f.write(losses_content)
    print("   ✅ models/losses.py créé")
    
    # ========== FICHIER 6: models/sparse_embedding.py ==========
    print("\n6️⃣ Création de models/sparse_embedding.py...")
    sparse_embedding_content = """# models/sparse_embedding.py  — CPU-only, sans init CUDA, tolère des kwargs

from typing import Optional, Any
import torch
from torch import nn

from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    \"\"\"
    Table d'embeddings (lookup) castée vers un dtype donné.
    Implémentation CPU-only : on force la création et le stockage sur CPU
    pour éviter toute initialisation CUDA.

    Args:
        num_embeddings (int): taille du vocab.
        embedding_dim  (int): dimension de l'embedding.
        init_std     (float): écart-type pour l'init tronquée.
        cast_to    (torch.dtype): dtype de calcul (ex: torch.float32/bfloat16).
        trainable     (bool): True => Paramètre appris ; False => buffer fixe.
        **kwargs       : absorbtion des paramètres optionnels (ex: batch_size, etc.)
                        transmis par l'architecture mais non utilisés ici en CPU-only.
    \"\"\"
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype,
        trainable: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.cast_to = cast_to

        # IMPORTANT : création explicite sur CPU pour éviter l'init CUDA
        weight_cpu = trunc_normal_init_(
            torch.empty((num_embeddings, embedding_dim), device="cpu"), std=init_std
        )

        if trainable:
            self.weight = nn.Parameter(weight_cpu)
        else:
            self.register_buffer("weight", weight_cpu, persistent=True)

        # On mémorise les kwargs à toutes fins utiles (debug/inspection)
        self.extra_kwargs = dict(kwargs)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        \"\"\"
        indices: LongTensor (...), valeurs dans [0, num_embeddings)
        Retour:  Tensor (..., embedding_dim) casté vers self.cast_to
        \"\"\"
        if indices.device.type != "cpu":
            indices = indices.to("cpu")

        flat = indices.view(-1).to(torch.long)
        out = self.weight.index_select(0, flat)
        out = out.view(*indices.shape, self.weight.shape[1])

        return out.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(CastedSparseEmbedding):
    \"\"\"
    Variante attendue par pretrain.py.
    Ici CPU-only, on absorbe aussi les kwargs (p.ex. liés au mode distribué).
    Toute logique signSGD/distrib doit rester côté optimiseur/runner.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


__all__ = [
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
]
"""
    
    with open(models_dir / "sparse_embedding.py", "w", encoding="utf-8") as f:
        f.write(sparse_embedding_content)
    print("   ✅ models/sparse_embedding.py créé")
    
    # ========== CRÉER UN README ==========
    print("\n7️⃣ Création du README...")
    readme_content = """# RTM Trading Project

## Structure du projet

```
RTM_Trading_Project/
├── models/
│   ├── __init__.py           # Package models
│   ├── common.py             # Fonctions communes (trunc_normal_init)
│   ├── ema.py                # Exponential Moving Average
│   ├── layers.py             # Couches du transformer (Attention, SwiGLU, etc.)
│   ├── losses.py             # Fonctions de loss
│   └── sparse_embedding.py   # Embeddings spécialisés
│
├── rtm_trading_system.py     # Système principal (à copier depuis artifacts)
├── rtm_training.py           # Script d'entraînement (à copier depuis artifacts)
├── rtm_test_simple.py        # Tests unitaires (à copier depuis artifacts)
└── README.md                 # Ce fichier
```

## Installation

1. **Dépendances Python:**
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm einops
```

2. **Fichiers manquants:**
Les fichiers suivants doivent être copiés depuis les artifacts Claude:
- `rtm_trading_system.py`
- `rtm_training.py`
- `rtm_test_simple.py`

3. **Bot MQL5:**
Le fichier `RTM_Adaptive_Bot.mq5` doit être placé dans:
```
C:/Users/[User]/AppData/Roaming/MetaQuotes/Terminal/[ID]/MQL5/Experts/
```

## Utilisation rapide

### Test du système
```bash
python rtm_test_simple.py
```

### Lancement du serveur
```bash
python rtm_trading_system.py
```

### Entraînement
```bash
python rtm_training.py
```

## Notes

- Tous les fichiers `models/` sont maintenant créés
- Assurez-vous d'avoir PyTorch installé
- Pour le trading réel, utilisez d'abord un compte démo

## Support

Ce projet utilise un Recursive Tiny Model (RTM) pour le trading adaptatif.
"""
    
    with open(project_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("   ✅ README.md créé")
    
    # ========== CRÉER requirements.txt ==========
    print("\n8️⃣ Création de requirements.txt...")
    requirements_content = """torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
einops>=0.7.0
"""
    
    with open(project_dir / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("   ✅ requirements.txt créé")
    
    # ========== RÉSUMÉ ==========
    print("\n" + "=" * 70)
    print("✅ INSTALLATION TERMINÉE!")
    print("=" * 70)
    
    print(f"\n📁 Structure créée dans: {project_dir.absolute()}\n")
    
    print("📦 Fichiers créés:")
    print("   ✅ models/__init__.py")
    print("   ✅ models/common.py")
    print("   ✅ models/ema.py")
    print("   ✅ models/layers.py")
    print("   ✅ models/losses.py")
    print("   ✅ models/sparse_embedding.py")
    print("   ✅ README.md")
    print("   ✅ requirements.txt")
    
    print("\n⚠️  FICHIERS À COPIER MANUELLEMENT:")
    print("   📄 rtm_trading_system.py (depuis l'artifact 'RTM Trading System - Python Bridge')")
    print("   📄 rtm_training.py (depuis l'artifact 'RTM Training - Apprentissage...')")
    print("   📄 rtm_test_simple.py (depuis l'artifact 'Test Simple du Système RTM')")
    print("   📄 RTM_Adaptive_Bot.mq5 (depuis l'artifact 'RTM Trading Bot MQL5')")
    
    print("\n🚀 PROCHAINES ÉTAPES:")
    print(f"   1. cd {project_dir}")
    print("   2. pip install -r requirements.txt")
    print("   3. Copier les fichiers Python depuis les artifacts Claude")
    print("   4. python rtm_test_simple.py")
    
    print("\n💡 Pour copier les fichiers depuis Claude:")
    print("   - Cliquez sur chaque artifact à droite")
    print("   - Copiez le contenu")
    print("   - Créez le fichier correspondant dans RTM_Trading_Project/")
    
    print("\n" + "=" * 70)
    
    return project_dir


if __name__ == "__main__":
    create_project_structure()
