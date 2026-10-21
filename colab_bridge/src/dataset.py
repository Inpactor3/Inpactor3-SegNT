"""
Dataset Inpactor2 Detect.

Lee JSONL de ventanas (producido por el downstream pipeline) + genomas .fna.gz,
extrae la secuencia nucleotidica de cada ventana y la tokeniza para Nucleotide Transformer.

Para validacion con TAIR10 (fase prototype): usa un JSONL generado a partir de la
anotacion curada de TAIR10 sobre el genoma de Arabidopsis thaliana.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Dataset de ventanas 50 kb desde JSONL + genomas .fna.gz locales."""

    def __init__(
        self,
        jsonl_paths: list[Path],
        genomes_dir: Path,
        tokenizer,
        chunk_bp: int = 6000,
        max_tokens: int = 1000,
        max_chunks_per_window: int = 1,
    ):
        self.rows: list[dict] = []
        for p in jsonl_paths:
            with open(p) as f:
                for line in f:
                    row = json.loads(line)
                    row["_jsonl"] = str(p)
                    self.rows.append(row)

        self.genomes_dir = Path(genomes_dir)
        self.tokenizer = tokenizer
        self.chunk_bp = chunk_bp
        self.max_tokens = max_tokens
        self.max_chunks_per_window = max_chunks_per_window
        self._genome_cache: dict[str, dict[str, str]] = {}

    def __len__(self) -> int:
        return len(self.rows) * self.max_chunks_per_window

    def _load_genome(self, species: str) -> Optional[dict[str, str]]:
        if species in self._genome_cache:
            return self._genome_cache[species]
        matches = list(self.genomes_dir.rglob(f"{species}*.fna.gz"))
        if not matches:
            return None
        seqs: dict[str, str] = {}
        cur, buf = None, []
        with gzip.open(matches[0], "rt") as f:
            for line in f:
                if line.startswith(">"):
                    if cur:
                        seqs[cur] = "".join(buf)
                    cur = line[1:].strip().split()[0]
                    buf = []
                else:
                    buf.append(line.strip())
            if cur:
                seqs[cur] = "".join(buf)
        if len(self._genome_cache) > 3:
            self._genome_cache.pop(next(iter(self._genome_cache)))
        self._genome_cache[species] = seqs
        return seqs

    def __getitem__(self, idx: int) -> dict:
        row_idx = idx // self.max_chunks_per_window
        chunk_idx = idx % self.max_chunks_per_window
        row = self.rows[row_idx]

        species = row["species"]
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        label = row["label"]

        seqs = self._load_genome(species)
        if seqs is None or chrom not in seqs:
            seq = "N" * self.chunk_bp
        else:
            full = seqs[chrom][start:end].upper()
            chunk_start = chunk_idx * self.chunk_bp
            seq = full[chunk_start:chunk_start + self.chunk_bp]
            if len(seq) < 100:
                seq = "N" * self.chunk_bp

        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(label)),
            "species": species,
        }


def tair10_to_jsonl(
    tair_gff_path: Path,
    tair_fasta_path: Path,
    output_jsonl: Path,
    window_size: int = 50000,
    min_overlap_frac: float = 0.2,
    min_overlap_bp: int = 50,
) -> None:
    """
    Convierte la anotacion curada TAIR10 (GFF3 de TEs) + genoma FASTA
    en un JSONL de ventanas 50 kb etiquetadas, para validar el prototype.
    """
    import gzip
    from collections import defaultdict

    sizes: dict[str, int] = {}
    cur, acc = None, 0
    opener = gzip.open if str(tair_fasta_path).endswith(".gz") else open
    with opener(tair_fasta_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    sizes[cur] = acc
                cur = line[1:].strip().split()[0]
                acc = 0
            else:
                acc += len(line.strip())
    if cur:
        sizes[cur] = acc

    hits: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with open(tair_gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, kind, s, e, *_ = parts
            if "transposable" not in kind.lower() and "retro" not in kind.lower() and "te" not in kind.lower():
                continue
            hits[chrom].append((int(s) - 1, int(e), kind))

    with open(output_jsonl, "w") as fo:
        for chrom, L in sizes.items():
            for start in range(0, L - window_size + 1, window_size):
                end = start + window_size
                bp_total = 0
                is_pos = False
                for hs, he, _ in hits.get(chrom, []):
                    if hs < end and he > start:
                        overlap = min(he, end) - max(hs, start)
                        if overlap <= 0:
                            continue
                        bp_total += overlap
                        elem_len = he - hs
                        thr = max(min_overlap_bp, int(min_overlap_frac * elem_len))
                        if overlap >= thr:
                            is_pos = True
                rec = {
                    "window_id": f"TAIR10_{chrom}_{start}",
                    "species": "Arabidopsis_thaliana",
                    "reino": "Plantas",
                    "split": "test",
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "label": 1 if is_pos else 0,
                    "bp_TE": bp_total,
                }
                fo.write(json.dumps(rec) + "\n")
