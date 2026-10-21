"""
embed_pipeline.py
Corre DENTRO del runtime Colab (conectado via VSCode SSH).
Pre-computa embeddings de Nucleotide Transformer para cada ventana del JSONL.

Flujo:
1. Lee un JSONL de ventanas (producido por el downstream pipeline).
2. Para cada ventana, extrae la secuencia nucleotidica del genoma correspondiente.
3. Pasa la secuencia por Nucleotide Transformer (modelo InstaDeep 500M).
4. Guarda un .npy con los embeddings alineado con el JSONL.

Uso:
    python embed_pipeline.py --jsonl hongos_train.jsonl \\
                             --genomes_dir /content/genomes \\
                             --out hongos_train_emb.npy

Requisitos (ya instalados por celda 3 del notebook):
    transformers torch biopython numpy tqdm
"""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
WINDOW_BP = 50000
MAX_TOKENS = 1000  # Limite del modelo (~6 kb por chunk, se chunkea)
CHUNK_BP = 6000


def load_genome(gz_path: Path) -> dict[str, str]:
    seqs = {}
    cur, buf = None, []
    with gzip.open(gz_path, "rt") as f:
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
    return seqs


def embed_window(seq: str, tokenizer, model, device) -> np.ndarray:
    """Divide la ventana en chunks de 6 kb, embebe cada uno, promedia."""
    chunks = [seq[i:i + CHUNK_BP] for i in range(0, len(seq), CHUNK_BP)]
    chunk_embs = []
    for ch in chunks:
        if len(ch) < 100:
            continue
        toks = tokenizer(ch, return_tensors="pt", truncation=True,
                         max_length=MAX_TOKENS).to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        # mean pooling sobre tokens
        last = out.hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
        chunk_embs.append(last)
    if not chunk_embs:
        return np.zeros(1280, dtype=np.float32)
    return np.mean(chunk_embs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--genomes_dir", required=True,
                    help="Carpeta con los .fna.gz, un archivo por especie")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Procesar solo las primeras N ventanas (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Cargando modelo {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

    with open(args.jsonl) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[:args.limit]
    print(f"Ventanas a procesar: {len(rows)}")

    genome_cache: dict[str, dict[str, str]] = {}
    embeddings = []

    for row in tqdm(rows):
        species = row["species"]
        chrom = row["chrom"]
        start, end = row["start"], row["end"]

        if species not in genome_cache:
            # Buscar .fna.gz en el directorio
            matches = list(Path(args.genomes_dir).rglob(f"{species}*.fna.gz"))
            if not matches:
                print(f"[WARN] no genoma para {species}, skip")
                embeddings.append(np.zeros(1280, dtype=np.float32))
                continue
            genome_cache[species] = load_genome(matches[0])
            # Liberar memoria si acumula demasiado
            if len(genome_cache) > 5:
                oldest = next(iter(genome_cache))
                if oldest != species:
                    del genome_cache[oldest]

        seqs = genome_cache[species]
        if chrom not in seqs:
            embeddings.append(np.zeros(1280, dtype=np.float32))
            continue
        window_seq = seqs[chrom][start:end].upper()
        emb = embed_window(window_seq, tokenizer, model, device)
        embeddings.append(emb)

    arr = np.stack(embeddings)
    np.save(args.out, arr)
    print(f"Guardado {args.out}  shape={arr.shape}  size={arr.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
