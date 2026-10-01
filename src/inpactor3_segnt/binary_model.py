"""
Clasificador binario SegNT para ventanas de 50 kb.

Arquitectura:
    Ventana 50 kb
      ↓ divide en 10 chunks de 5 kb
      ↓ cada chunk tokenizado (~800 tokens)
    [10 × encoder NT] (compartido, mismos pesos)
      ↓ promedio por token (pooling intra-chunk)
    [10 vectores de 512]
      ↓ max-pool entre chunks (si ALGÚN chunk tiene LTR-RT → ventana positiva)
    [vector de 512]
      ↓ cabeza binaria
    [sigmoid → 1 scalar]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class NTBinaryClassifier(nn.Module):
    """SegNT adaptado a clasificación binaria con pooling multi-chunk."""

    def __init__(
        self,
        base_model_name: str,
        freeze_encoder: bool = True,
        head_hidden: int = 256,
        dropout: float = 0.1,
        pool: str = "max",   # "max" (MIL) o "mean"
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.pool = pool

        # Patch config con atributos que ESM nuevo exige y NT no trae
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        for k, v in {
            "rope_theta": 10000.0, "is_decoder": False,
            "add_cross_attention": False, "use_cache": False,
            "output_attentions": False, "output_hidden_states": False,
            "tie_word_embeddings": False,
            "position_embedding_type": getattr(config, "position_embedding_type", "rotary"),
            "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-12),
            "attention_probs_dropout_prob": getattr(config, "attention_probs_dropout_prob", 0.0),
            "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.0),
        }.items():
            if not hasattr(config, k) or getattr(config, k) is None:
                setattr(config, k, v)

        self.encoder = AutoModel.from_pretrained(
            base_model_name, config=config,
            trust_remote_code=True, ignore_mismatched_sizes=True,
        )
        embed_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Cabeza binaria
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids:      (B, n_chunks, T)
        attention_mask: (B, n_chunks, T)

        Devuelve: (B,) logits para BCE.
        """
        B, C, T = input_ids.shape
        # Aplanar chunks: (B*C, T)
        ids = input_ids.view(B * C, T)
        mask = attention_mask.view(B * C, T) if attention_mask is not None else None

        out = self.encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=False, return_dict=True)
        h = out.last_hidden_state  # (B*C, T, D)

        # Pooling intra-chunk (mean sobre tokens válidos)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            summed = (h * m).sum(dim=1)
            counts = m.sum(dim=1).clamp(min=1)
            chunk_emb = summed / counts
        else:
            chunk_emb = h.mean(dim=1)

        # (B*C, D) → (B, C, D)
        chunk_emb = chunk_emb.view(B, C, -1)

        # Pooling entre chunks: MIL-style max por defecto
        if self.pool == "max":
            win_emb, _ = chunk_emb.max(dim=1)
        elif self.pool == "mean":
            win_emb = chunk_emb.mean(dim=1)
        else:
            raise ValueError(f"pool desconocido: {self.pool}")

        logits = self.head(win_emb).squeeze(-1)  # (B,)
        return logits


def binary_loss(logits: torch.Tensor, labels: torch.Tensor,
                pos_weight: float = 1.0) -> torch.Tensor:
    """BCE con pos_weight para desbalance."""
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pw)
