"""Extrae la secuencia de 6 kb usada por cada ventana y la empaqueta por especie.

Uso:
  python make_sequence_bundle.py --jsonl_dir <consolidated/R> --genomes_dir <genomes/R> \
      --out_dir <seq_bundle> [--chunk_bp 6000] [--splits train,val,test]

Salida: <out_dir>/<split>/<tag>.npz con seq (N,chunk_bp) uint8, label (N,) int8,
window_id (N,) unicode.
"""
import argparse, gzip, json
from pathlib import Path

import numpy as np


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def stream_genome(gz_path, wanted):
    cur, buf = None, []
    with gzip.open(gz_path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None and cur in wanted:
                    yield cur, "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None and cur in wanted:
        yield cur, "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_dir", type=Path, required=True)
    ap.add_argument("--genomes_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--chunk_bp", type=int, default=6000)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    for split in args.splits.split(","):
        files = sorted((args.jsonl_dir / split).glob("*.jsonl"))
        outdir = args.out_dir / split
        outdir.mkdir(parents=True, exist_ok=True)
        for jf in files:
            tag = jf.stem
            out = outdir / f"{tag}.npz"
            if out.exists():
                print("SKIP", split, tag, flush=True)
                continue
            rows = read_rows(jf)
            by_chrom = {}
            for i, r in enumerate(rows):
                by_chrom.setdefault(r["chrom"], []).append(i)

            arr = np.full((len(rows), args.chunk_bp), ord("N"), dtype=np.uint8)
            labels = np.zeros(len(rows), dtype=np.int8)
            ids = []
            for i, r in enumerate(rows):
                labels[i] = int(r.get("label", 0))
                ids.append(r.get("window_id", ""))

            hits = list(args.genomes_dir.rglob(f"{tag}*.fna.gz"))
            if hits:
                for chrom, seq in stream_genome(hits[0], set(by_chrom)):
                    for i in by_chrom[chrom]:
                        st = rows[i]["start"]
                        chunk = seq[st:st + args.chunk_bp].upper()
                        if len(chunk) < 100:
                            chunk = "N" * args.chunk_bp
                        chunk = chunk[: args.chunk_bp].ljust(args.chunk_bp, "N")
                        arr[i] = np.frombuffer(chunk.encode(), dtype=np.uint8)
                    del seq

            np.savez_compressed(out, seq=arr, label=labels,
                                window_id=np.array(ids))
            print("OK", split, tag, arr.shape, flush=True)


if __name__ == "__main__":
    main()
