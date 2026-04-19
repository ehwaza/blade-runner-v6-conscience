# models/sparse_embedding.py  — CPU-only, sans init CUDA, tolère des kwargs

from typing import Optional, Any
import torch
from torch import nn

from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    """
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
    """
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
        """
        indices: LongTensor (...), valeurs dans [0, num_embeddings)
        Retour:  Tensor (..., embedding_dim) casté vers self.cast_to
        """
        if indices.device.type != "cpu":
            indices = indices.to("cpu")

        flat = indices.view(-1).to(torch.long)
        out = self.weight.index_select(0, flat)
        out = out.view(*indices.shape, self.weight.shape[1])

        return out.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(CastedSparseEmbedding):
    """
    Variante attendue par pretrain.py.
    Ici CPU-only, on absorbe aussi les kwargs (p.ex. liés au mode distribué).
    Toute logique signSGD/distrib doit rester côté optimiseur/runner.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


__all__ = [
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
]
