"""Extrae los bloques de secuencia usados por cada ventana y los empaqueta por especie.

Layout por ventana (guardado en seq como (N, chunks, chunk_bp)):
  * por defecto: `--chunks 4` bloques de 6 kb dentro de la ventana de 50 kb
    (offsets 0, 14k, 28k, 42k) => 24 kb de entrada -> max pooling.
  * `--mode eval_full --chunks 8`: 8 bloques contiguos (0..48 kb)
    => 48 kb de entrada, usado para la evaluacion en GPU A100.

Uso:
  python make_sequence_bundle.py --jsonl_dir <consolidated/R> --genomes_dir <genomes/R> \
      --out_dir <seq_bundle> --chunks 4 --splits train,val

  # paquete de evaluacion (48 kb):
  python make_sequence_bundle.py --jsonl_dir <consolidated/R> --genomes_dir <genomes/R> \
      --out_dir <seq_bundle_eval8> --chunks 8 --mode eval_full --splits val,test

Salida: <out_dir>/<split>/<tag>.npz con
  seq        (N, chunks, chunk_bp) uint8
  label      (N,) int8
  window_id  (N,) unicode
  offsets    (chunks,) int64 (offsets en bp dentro de la ventana)
  chunk_bp   () int64
"""
import argparse
import gzip
import json
import os
from pathlib import Path

import re

import numpy as np

from block_layout import block_offsets


def norm(name):
    """Colapsa runs de guiones bajos: 'Sp___GCF_1' y 'Sp__GCF_1' son la misma."""
    return re.sub(r"_+", "_", name)


def index_genomes(genomes_dir):
    """Mapa nombre-normalizado -> ruta del .fna.gz, y por especie (sin accession)."""
    idx, por_especie = {}, {}
    for p in Path(genomes_dir).rglob("*.fna.gz"):
        base = norm(p.name[:-7])
        idx[base] = p
        esp = re.sub(r"_(GC[AF]_\d+\.\d+)$", "", base)   # quita el accession
        por_especie.setdefault(esp, []).append(p)
    return idx, por_especie


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
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk_bp", type=int, default=6000)
    ap.add_argument("--mode", choices=("windowed", "eval_full"), default="windowed")
    ap.add_argument("--window_size", type=int, default=50000)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    offs = block_offsets(args.chunks, args.window_size, args.chunk_bp, args.mode)
    print(f"chunks={args.chunks} chunk_bp={args.chunk_bp} modo={args.mode} "
          f"offsets={offs.tolist()} total={args.chunks * args.chunk_bp} pb", flush=True)

    gidx, por_especie = index_genomes(args.genomes_dir)
    print(f"genomas indexados: {len(gidx)}", flush=True)
    faltantes = []

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

            arr = np.full((len(rows), args.chunks, args.chunk_bp), ord("N"), dtype=np.uint8)
            labels = np.zeros(len(rows), dtype=np.int8)
            ids = []
            for i, r in enumerate(rows):
                labels[i] = int(r.get("label", 0))
                ids.append(r.get("window_id", ""))

            gpath = gidx.get(norm(tag))
            if gpath is None:
                # la misma especie puede estar con otro accession de ensamblado
                esp = re.sub(r"_(GC[AF]_\d+\.\d+)$", "", norm(tag))
                cand = por_especie.get(esp, [])
                if len(cand) == 1:
                    gpath = cand[0]
                    print(f"AVISO {tag}: sin genoma con ese accession, se usa "
                          f"{gpath.name} (mismo nombre de especie)", flush=True)
                elif len(cand) > 1:
                    print(f"ERROR {tag}: {len(cand)} genomas candidatos para la "
                          f"especie, hay que elegir a mano", flush=True)
            hits = [gpath] if gpath is not None else []
            if hits:
                for chrom, seq in stream_genome(hits[0], set(by_chrom)):
                    for i in by_chrom[chrom]:
                        st = rows[i]["start"]
                        for c, off in enumerate(offs):
                            piece = seq[st + int(off): st + int(off) + args.chunk_bp].upper()
                            if len(piece) < 100:
                                continue
                            arr[i, c] = np.frombuffer(
                                piece[: args.chunk_bp].ljust(args.chunk_bp, "N").encode(),
                                dtype=np.uint8)
                    del seq
            else:
                faltantes.append(tag)
                print(f"ERROR sin genoma para {tag}: se escribiria una ventana "
                      f"llena de N. Se omite el archivo.", flush=True)
                if os.path.exists(out):
                    os.remove(out)
                continue

            np.savez_compressed(out, seq=arr, label=labels,
                                window_id=np.array(ids), offsets=offs,
                                chunk_bp=np.int64(args.chunk_bp))
            # guarda: si quedo practicamente todo N, la extraccion fallo
            # (genoma distinto, contigs con otros nombres). No se guarda.
            frac_n = float((arr == ord("N")).mean())
            if frac_n > 0.99:
                os.remove(out)
                faltantes.append(tag)
                print(f"ERROR {tag}: {frac_n*100:.1f}% de N, no se guarda "
                      f"(contigs del genoma no coinciden con el JSONL)", flush=True)
                continue
            print("OK", split, tag, arr.shape, flush=True)

    if faltantes:
        print(f"\nATENCION: {len(faltantes)} especies sin genoma (no se generaron):",
              flush=True)
        for t in faltantes:
            print("  ", t, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
