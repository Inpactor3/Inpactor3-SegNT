"""
Dataset lazy que materializa ventanas on-the-fly a partir del manifest.jsonl.

Uso desde train_teon.py:
    ds = TeonWindowDataset(
        manifest="data/common_dataset/manifest.jsonl",
        genomes_dir="data/common_dataset/genomes",
        panteon_subset="data/common_dataset/panteon_ltr_subset.fasta",
        split="train",
        tokenizer=tok,
    )
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowRecipe:
    window_id: str
    kingdom: str
    background_genome: str
    background_scaffold: str
    background_start: int
    background_end: int
    has_ltr_rt: bool
    insertions: list[dict]
    split: str


def load_manifest(path: Path, split: str | None = None) -> list[WindowRecipe]:
    """Lee JSONL y devuelve lista de recetas, opcionalmente filtradas por split."""
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if split is not None and d.get("split") != split:
                continue
            out.append(WindowRecipe(**{k: d[k] for k in WindowRecipe.__annotations__}))
    return out


class FastaSlicer:
    """Lee un FASTA una vez y saca subsecuencias por scaffold + rango."""
    def __init__(self, path: Path):
        self.path = path
        self.index: dict[str, tuple[int, int]] = {}  # scaffold → (byte_offset, length)
        self._build_index()

    def _build_index(self):
        # Índice simple: guarda offset y longitud de cada scaffold
        offset = 0
        cur_name = None
        cur_start = 0
        cur_len = 0
        with open(self.path, "rb") as f:
            for line in f:
                if line.startswith(b">"):
                    if cur_name is not None:
                        self.index[cur_name] = (cur_start, cur_len)
                    cur_name = line[1:].split()[0].decode()
                    cur_start = offset + len(line)
                    cur_len = 0
                else:
                    cur_len += len(line.strip())
                offset += len(line)
            if cur_name is not None:
                self.index[cur_name] = (cur_start, cur_len)

    def slice(self, scaffold: str, start: int, end: int) -> str:
        """Devuelve subsecuencia [start, end) de un scaffold, saltando saltos de línea."""
        if scaffold not in self.index:
            raise KeyError(f"scaffold {scaffold} no está en {self.path}")
        offset, total = self.index[scaffold]
        end = min(end, total)
        result = []
        collected = 0
        with open(self.path, "rb") as f:
            f.seek(offset)
            pos = 0  # posición dentro del scaffold (sin newlines)
            for line in f:
                if line.startswith(b">"):
                    break
                bases = line.strip()
                line_len = len(bases)
                if pos + line_len > start and pos < end:
                    lo = max(0, start - pos)
                    hi = min(line_len, end - pos)
                    result.append(bases[lo:hi].decode())
                pos += line_len
                if pos >= end:
                    break
        return "".join(result).upper()


class PanteonSubset:
    """Carga en memoria las secuencias del subset PanTEon (500 MB máximo)."""
    def __init__(self, path: Path):
        self.seqs: dict[str, str] = {}
        cur, buf = None, []
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if cur is not None:
                        self.seqs[cur] = "".join(buf).upper()
                    cur = line[1:].split()[0]
                    buf = []
                else:
                    buf.append(line)
            if cur is not None:
                self.seqs[cur] = "".join(buf).upper()

    def get(self, panteon_id: str) -> str:
        return self.seqs[panteon_id]


class TeonWindowDataset(Dataset):
    """
    Dataset lazy. Cada __getitem__ construye la ventana de 50 kb según la
    receta del manifiesto: baja el fondo del genoma correspondiente, planta
    el(los) TE(s) en su posición, y devuelve tokens + labels.
    """

    def __init__(
        self,
        manifest: Path,
        genomes_dir: Path,
        panteon_subset: Path,
        tokenizer=None,
        split: str = "train",
        window_size: int = 6000,
        max_tokens: int = 1000,
        cell_size: int = 100,
    ):
        self.recipes = load_manifest(manifest, split=split)
        self.slicers: dict[str, FastaSlicer] = {}
        for g in {r.background_genome for r in self.recipes}:
            gpath = genomes_dir / f"{g}.fasta"
            if gpath.exists():
                print(f"[index] {gpath}")
                self.slicers[g] = FastaSlicer(gpath)
        self.panteon = PanteonSubset(panteon_subset)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.cell_size = cell_size

    def __len__(self) -> int:
        return len(self.recipes)

    def _materialize(self, r: WindowRecipe) -> tuple[str, list[int]]:
        """Construye la secuencia + labels por bp."""
        seq = self.slicers[r.background_genome].slice(
            r.background_scaffold, r.background_start, r.background_end
        )
        # padding si el scaffold era más corto
        if len(seq) < (r.background_end - r.background_start):
            seq = seq + "N" * ((r.background_end - r.background_start) - len(seq))
        seq = list(seq)
        labels = [0] * len(seq)

        for ins in r.insertions:
            te_seq = self.panteon.get(ins["panteon_id"])
            pos = ins["insert_position"]
            end = pos + len(te_seq)
            # sobrescribir en la ventana
            for i, c in enumerate(te_seq):
                if pos + i < len(seq):
                    seq[pos + i] = c
                    labels[pos + i] = ins["lineage_id"]

        return "".join(seq), labels

    def __getitem__(self, idx: int):
        r = self.recipes[idx]
        # Recorte al tamaño de ventana del modelo (6 kb por default)
        full_seq, full_labels = self._materialize(r)
        # elegir sub-ventana de window_size centrada en la inserción si es positiva
        if r.has_ltr_rt and r.insertions:
            ins = r.insertions[0]
            center = ins["insert_position"] + ins["insert_length"] // 2
            start = max(0, center - self.window_size // 2)
            end = min(len(full_seq), start + self.window_size)
            start = end - self.window_size
        else:
            # negativo: ventana aleatoria
            rng = random.Random(idx)
            start = rng.randint(0, max(0, len(full_seq) - self.window_size))
            end = start + self.window_size
        sub_seq = full_seq[start:end]
        sub_labels = full_labels[start:end]

        if self.tokenizer is None:
            # modo raw
            return {
                "sequence": sub_seq,
                "bp_labels": sub_labels,
                "meta": {"window_id": r.window_id, "kingdom": r.kingdom},
            }

        enc = self.tokenizer(
            sub_seq,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        specials = set(self.tokenizer.all_special_tokens)
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            if tok in specials or tok is None:
                offsets.append((cursor, cursor))
            else:
                tlen = sum(1 for c in tok if c.upper() in "ACGTN")
                offsets.append((cursor, cursor + tlen))
                cursor += tlen

        token_labels = np.full(len(offsets), -100, dtype=np.int64)
        for t, (bs, be) in enumerate(offsets):
            if bs == be or bs >= len(sub_labels):
                continue
            span = sub_labels[bs : min(be, len(sub_labels))]
            if not span:
                continue
            vals, counts = np.unique(span, return_counts=True)
            token_labels[t] = int(vals[counts.argmax()])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.from_numpy(token_labels),
            "meta": {"window_id": r.window_id, "kingdom": r.kingdom},
        }
