"""
Inpactor2 Detect · modelo de deteccion de TE sobre ventanas 50 kb.

Arquitectura:
- Backbone: Nucleotide Transformer (InstaDeep 500M) congelado (fase 1) o fine-tuned (fase 2)
- Encabezado: MLP para clasificacion binaria (positive = ventana contiene TE con solapamiento >= umbral)

Input:  secuencia DNA de 50 kb (chunked en bloques de 6 kb por el tokenizer de NT)
Output: probabilidad de ventana positiva + logits por familia TE (opcional multi-task)
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Inpactor2Detect(nn.Module):
    def __init__(
        self,
        nt_model_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        n_families: int = 0,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(nt_model_name)
        hidden = self.backbone.config.hidden_size  # 1280 para 500m

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.n_families = n_families
        if n_families > 0:
            self.fam_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_families),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        input_ids: (B, L) tokens de NT para el chunk
        attention_mask: (B, L)
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # (B, L, H)

        # mean pooling con mascara
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        feat = self.dropout(torch.relu(self.pool_proj(pooled)))

        result = {"logits": self.cls_head(feat).squeeze(-1)}
        if self.n_families > 0:
            result["fam_logits"] = self.fam_head(feat)
        return result


def load_tokenizer(nt_model_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref"):
    return AutoTokenizer.from_pretrained(nt_model_name)
