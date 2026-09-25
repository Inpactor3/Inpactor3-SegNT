"""
Dataset para segmentación de LTR-RTs a nivel token.

Corta el genoma en ventanas de ~6 kb, tokeniza con el tokenizer del modelo
base (Nucleotide Transformer usa BPE con k-mers de 6 bp aprox), y genera
una etiqueta por token: 0 = fondo, 1..K = clase de LTR-RT.

El mapeo token→bp se calcula porque BPE agrupa bases de forma variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class Annotation:
    """Anotación puntual de un LTR-RT: (start, end, class_id)."""
    start: int
    end: int
    cls: int  # 1..K (0 reservado a fondo)


def parse_fasta(path: Path) -> dict[str, str]:
    """Devuelve {scaffold_id: secuencia mayúsculas}."""
    seqs: dict[str, str] = {}
    cur, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf).upper()
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            seqs[cur] = "".join(buf).upper()
    return seqs


def parse_inpactor2_tab(
    path: Path,
    class_map: dict[str, int] | None = None,
) -> dict[str, list[Annotation]]:
    """
    Lee Inpactor2_predictions.tab (formato:
        seqid  start  end  length  lineage  det  filt  cls
    ).

    class_map: {"RLC/ALE/RETROFIT": 1, ...} — si None, todo es clase 1 (binario).
    """
    ann: dict[str, list[Annotation]] = {}
    with open(path) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue
            scaf, s, e, _, lin = cols[0], int(cols[1]), int(cols[2]), cols[3], cols[4]
            cls = 1 if class_map is None else class_map.get(lin, 1)
            ann.setdefault(scaf, []).append(Annotation(s, e, cls))
    return ann


class NucleotideSegDataset(Dataset):
    """
    Ventanas de secuencia + etiquetas por token.

    Argumentos:
      sequences:  {scaffold: DNA string} filtrado a los cromosomas del split.
      annotations: {scaffold: [Annotation]}
      tokenizer:  HF tokenizer (Nucleotide Transformer).
      window_size: bp por ventana (default 6000, ~1000 tokens BPE).
      window_stride: paso entre ventanas.
      max_tokens: máximo tokens que el modelo acepta (default 1000).
      min_positive_frac: si una ventana positiva tiene menos de X% cubierto,
                         se descarta (control de calidad).
    """

    def __init__(
        self,
        sequences: dict[str, str],
        annotations: dict[str, list[Annotation]],
        tokenizer,
        window_size: int = 6000,
        window_stride: int = 3000,
        max_tokens: int = 1000,
        min_positive_frac: float = 0.02,
        include_all_windows: bool = True,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.window_stride = window_stride
        self.max_tokens = max_tokens
        self.min_positive_frac = min_positive_frac
        # pre-generar índice de ventanas
        self.windows: list[tuple[str, int, int]] = []  # (scaf, start, end)
        for scaf, seq in sequences.items():
            L = len(seq)
            for w_start in range(0, L, window_stride):
                w_end = min(w_start + window_size, L)
                if w_end - w_start < window_size // 2:
                    continue
                self.windows.append((scaf, w_start, w_end))
        self.sequences = sequences
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.windows)

    def _labels_per_bp(self, scaf: str, w_start: int, w_end: int) -> np.ndarray:
        """Array de tamaño window_size con la clase por bp (0 = fondo)."""
        labels = np.zeros(w_end - w_start, dtype=np.int8)
        for ann in self.annotations.get(scaf, []):
            # solape con la ventana
            s = max(ann.start, w_start)
            e = min(ann.end, w_end)
            if s < e:
                labels[s - w_start : e - w_start] = ann.cls
        return labels

    def __getitem__(self, idx: int):
        scaf, w_start, w_end = self.windows[idx]
        seq = self.sequences[scaf][w_start:w_end]

        # tokenizar (Nucleotide Transformer no soporta return_offsets_mapping;
        # calculamos las posiciones bp manualmente a partir de los strings)
        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Reconstruir offsets manualmente
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        offsets: list[tuple[int, int]] = []
        cursor = 0
        specials = set(self.tokenizer.all_special_tokens)
        for tok in tokens:
            if tok in specials or tok is None:
                offsets.append((cursor, cursor))  # sin span en la secuencia
            else:
                # k-mer típico del NT: alfabéticos ACGT
                tlen = sum(1 for c in tok if c.upper() in "ACGTN")
                if tlen == 0:
                    offsets.append((cursor, cursor))
                else:
                    offsets.append((cursor, cursor + tlen))
                    cursor += tlen

        # etiquetas por bp → etiqueta por token (mayoría dentro del span)
        bp_labels = self._labels_per_bp(scaf, w_start, w_end)
        T = len(offsets)
        token_labels = np.full(T, -100, dtype=np.int64)  # -100 = ignorar
        for t, (bp_s, bp_e) in enumerate(offsets):
            if bp_s == bp_e:  # token especial
                continue
            if bp_s >= len(bp_labels):
                continue
            span = bp_labels[bp_s : min(bp_e, len(bp_labels))]
            if len(span) == 0:
                continue
            vals, counts = np.unique(span, return_counts=True)
            token_labels[t] = int(vals[counts.argmax()])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.from_numpy(token_labels),
            "meta": {
                "scaffold": scaf,
                "window_start": w_start,
                "window_end": w_end,
                "offsets": offsets,
            },
        }


def collate(batch: list[dict]) -> dict:
    """Collate para DataLoader. Mantiene meta como lista de dicts."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def build_splits(
    sequences: dict[str, str],
    annotations: dict[str, list[Annotation]],
    train_chroms: list[str],
    val_chroms: list[str],
    test_chroms: list[str],
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Divide secuencias y anotaciones por cromosoma (por scaffold_id)."""
    def _sub(keys):
        return (
            {k: sequences[k] for k in keys if k in sequences},
            {k: annotations.get(k, []) for k in keys if k in sequences},
        )
    tr_seq, tr_ann = _sub(train_chroms)
    va_seq, va_ann = _sub(val_chroms)
    te_seq, te_ann = _sub(test_chroms)
    return tr_seq, tr_ann, va_seq, va_ann, te_seq, te_ann
