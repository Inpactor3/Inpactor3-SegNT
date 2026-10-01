"""
Dataset lazy que lee los .npy del paquete inpactor2_detect_panteon_v1 (PanteonDataSet.rar).

Formato:
    X_*.npy  →  (N, 50000)  uint8  alfabeto: {0:A, 1:C, 2:G, 3:T, 4:N}
    y_*.npy  →  (N,)        int8   {0 fondo, 1 con LTR-RT plantado}

Clasificación BINARIA por ventana de 50 kb (no segmentación por token).

Para SegNT dividimos cada ventana de 50 kb en 10 chunks de 5 kb,
tokenizamos cada chunk y promediamos las representaciones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ALPHABET = ["A", "C", "G", "T", "N"]
_LOOKUP = np.array(ALPHABET, dtype="U1")  # vectorized uint8 → char


def uint8_to_dna(arr: np.ndarray) -> str:
    """Convierte una fila de uint8 (0..4) a string DNA."""
    return "".join(_LOOKUP[arr].tolist())


class PanteonBinaryDataset(Dataset):
    """
    Dataset de ventanas de 50 kb con etiqueta binaria.

    Argumentos:
      npy_dir:       carpeta con X_{split}.npy y y_{split}.npy
      split:         'train', 'dev' o 'test'
      tokenizer:     tokenizer del Transformer (si None, devuelve raw DNA)
      chunk_size:    tamaño de cada sub-ventana (default 5000)
      n_chunks:      cantidad de chunks por ventana (default 10 → cubre 50 kb)
      max_tokens:    máximo tokens por chunk
    """

    def __init__(
        self,
        npy_dir: Path,
        split: str,
        tokenizer=None,
        chunk_size: int = 5000,
        n_chunks: int = 10,
        max_tokens: int = 1000,
    ):
        npy_dir = Path(npy_dir)
        self.X = np.load(npy_dir / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(npy_dir / f"y_{split}.npy", mmap_mode="r")
        assert self.X.shape[0] == self.y.shape[0]
        self.window_size = self.X.shape[1]
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.max_tokens = max_tokens
        assert chunk_size * n_chunks <= self.window_size, (
            f"chunk_size*n_chunks={chunk_size*n_chunks} > "
            f"window_size={self.window_size}"
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = np.asarray(self.X[idx])
        label = int(self.y[idx])

        # Decodificar uint8 → string
        dna = uint8_to_dna(row)

        if self.tokenizer is None:
            return {
                "sequence": dna,
                "label": label,
            }

        # Dividir en n_chunks de chunk_size cada uno
        chunks = [dna[i * self.chunk_size : (i + 1) * self.chunk_size]
                  for i in range(self.n_chunks)]

        enc = self.tokenizer(
            chunks,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
        )

        return {
            "input_ids": enc["input_ids"],         # (n_chunks, max_tokens)
            "attention_mask": enc["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate(batch):
    """Collate: apila input_ids y attention_mask de todos los chunks de todas las muestras."""
    input_ids = torch.stack([b["input_ids"] for b in batch])      # (B, n_chunks, T)
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])              # (B,)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
