"""
NTSegmentation: Nucleotide Transformer + cabeza de segmentación per-token.

Toma un modelo pre-entrenado de HuggingFace (Nucleotide Transformer o DNABERT-2)
y añade una cabeza convolucional 1D que produce logits de clase por cada
token de la ventana. En inferencia, los tokens se re-mapean a coordenadas
de nucleótido usando el mapeo token→bp que provee el tokenizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class SegmentationHead(nn.Module):
    """Cabeza convolucional 1D sobre embeddings de tokens → logits por token."""

    def __init__(self, in_dim: int, hidden_dims: list[int], num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Conv1d(prev, h, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Conv1d(prev, num_classes, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, C)"""
        # (B, T, D) → (B, D, T) para Conv1D
        y = self.net(x.transpose(1, 2))
        return y.transpose(1, 2)


class NTSegmentation(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_classes: int = 2,
        head_channels: list[int] | None = None,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        # Nucleotide Transformer usa ESM bajo el capó. Versiones nuevas de
        # transformers exigen `rope_theta` en el config; el config oficial
        # del NT no lo trae → lo añadimos con el valor por defecto de ESM.
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        if not hasattr(config, "rope_theta") or config.rope_theta is None:
            config.rope_theta = 10000.0
        self.encoder = AutoModel.from_pretrained(
            base_model_name, config=config, trust_remote_code=True
        )
        embed_dim = self.encoder.config.hidden_size
        self.head = SegmentationHead(
            in_dim=embed_dim,
            hidden_dims=head_channels or [256, 128],
            num_classes=num_classes,
            dropout=dropout,
        )
        self.num_classes = num_classes
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: (B, T) tokens
        attention_mask: (B, T) 1 donde válido, 0 en padding

        Devuelve: (B, T, num_classes) logits por token.
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=False, return_dict=True)
        # HF devuelve last_hidden_state (B, T, D)
        h = out.last_hidden_state
        return self.head(h)


def token_loss(
    logits: torch.Tensor,       # (B, T, C)
    labels: torch.Tensor,       # (B, T) long, -100 = ignorar
    attention_mask: torch.Tensor,  # (B, T)
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy por token, ponderando la clase positiva."""
    B, T, C = logits.shape
    # weight: dar mayor peso a clase 1 (LTR-RT) para compensar desbalance
    weight = torch.ones(C, device=logits.device)
    if C >= 2:
        weight[1:] = pos_weight  # cualquier clase != fondo pesa más
    loss = F.cross_entropy(
        logits.view(-1, C),
        labels.view(-1),
        weight=weight,
        ignore_index=-100,
        reduction="mean",
    )
    return loss


@torch.no_grad()
def decode_predictions(
    logits: torch.Tensor,       # (B, T, C)
    token_to_bp: list[list[tuple[int, int]]],  # por batch, por token → (bp_start, bp_end)
    threshold: float = 0.5,
    cell_size: int = 100,
    min_length: int = 1000,
) -> list[list[tuple[int, int, int, float]]]:
    """
    Convierte logits por token en cajas contiguas de LTR-RT.

    1. Softmax → prob de "no-fondo" por token.
    2. Umbraliza y proyecta a bp.
    3. Agrupa runs contiguos como una caja.
    """
    probs = F.softmax(logits, dim=-1)
    # prob de que sea cualquier clase != fondo (suma sobre clases 1..C-1)
    prob_pos = 1.0 - probs[..., 0]
    argmax_cls = probs.argmax(-1)

    out: list[list[tuple[int, int, int, float]]] = []
    for b in range(logits.size(0)):
        boxes: list[tuple[int, int, int, float]] = []
        current_start = None
        current_class = 0
        current_scores: list[float] = []

        for t, (bp_s, bp_e) in enumerate(token_to_bp[b]):
            p = float(prob_pos[b, t])
            cls = int(argmax_cls[b, t])
            is_pos = p > threshold and cls != 0

            if is_pos:
                if current_start is None:
                    current_start = bp_s
                    current_class = cls
                    current_scores = [p]
                else:
                    current_scores.append(p)
            else:
                if current_start is not None:
                    length = bp_e - current_start
                    if length >= min_length:
                        boxes.append((
                            current_start,
                            token_to_bp[b][t - 1][1],
                            current_class,
                            sum(current_scores) / len(current_scores),
                        ))
                    current_start = None
                    current_scores = []

        # cierre al final de la ventana
        if current_start is not None:
            last_end = token_to_bp[b][-1][1]
            if last_end - current_start >= min_length:
                boxes.append((current_start, last_end, current_class,
                              sum(current_scores) / len(current_scores)))
        out.append(boxes)
    return out
