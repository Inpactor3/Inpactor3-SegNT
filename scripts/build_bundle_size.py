"""Genera jsonl + bundle npz con un tamano de ventana distinto al de 50 kb.

Uso:
  python build_bundle_size.py --reino Hongos --window 24000 \
     --bed_dir /media/.../annotations/Hongos \
     --genomes_dir /media/.../genomes_449/Hongos \
     --out_dir /tmp/bundle24_Hongos
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sizes_de(genome):
    out = []
    cur, n = None, 0
    with gzip.open(genome, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    out.append((cur, n))
                cur = line[1:].split()[0]
                n = 0
            else:
                n += len(line.strip())
    if cur is not None:
        out.append((cur, n))
    return out


def _una_especie(work):
    """Genera sizes + jsonl de una especie. Devuelve (ok, sp, motivo)."""
    bed, sizes_path, dst, reino, split, window, genome = work
    sizes_path, dst, genome = Path(sizes_path), Path(dst), Path(genome)
    try:
        if not (dst.exists() and dst.stat().st_size > 0):
            if not sizes_path.exists():
                with open(sizes_path, "w") as f:
                    for c, n in sizes_de(genome):
                        f.write(f"{c}\t{n}\n")
            subprocess.run([sys.executable, str(HERE / "build_windows_var.py"),
                            str(bed), str(sizes_path), str(dst), reino, split,
                            str(window)], check=True, stdout=subprocess.DEVNULL)
        return (True, Path(dst).stem, "")
    except Exception as e:
        return (False, Path(dst).stem, str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reino", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--bed_dir", default=None)
    ap.add_argument("--bed_dirs", default=None,
                    help="lista separada por ':' de directorios donde buscar el BED")
    ap.add_argument("--consolidated", default=None,
                    help="directorio consolidated/<Reino> para tomar la lista exacta de especies")
    ap.add_argument("--genomes_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    jsonl = out / "jsonl"
    bundle = out / f"bundle{args.chunks}_{args.reino}"
    jsonl.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)

    dirs = [Path(d) for d in (args.bed_dirs or args.bed_dir or "").split(":") if d]

    def buscar_bed(split, sp):
        for d in dirs:
            for cand in (d / split / f"{sp}.bed", d / split / f"{sp}.bed.gz"):
                if cand.exists():
                    return cand
        return None

    for split in args.splits.split(","):
        if args.consolidated:
            sps = sorted(p.stem for p in (Path(args.consolidated) / split).glob("*.jsonl"))
        else:
            sps = sorted(p.stem for p in (dirs[0] / split).glob("*.bed"))
        gdir = Path(args.genomes_dir) / split
        print(f"[{split}] {len(sps)} especies", flush=True)
        (jsonl / split).mkdir(parents=True, exist_ok=True)
        faltan = 0
        trabajos = []
        for sp in sps:
            bed = buscar_bed(split, sp)
            genome = gdir / f"{sp}.fna.gz"
            dst = jsonl / split / f"{sp}.jsonl"
            if bed is None or not genome.exists():
                faltan += 1
                continue
            trabajos.append((str(bed), tmp / f"{sp}.sizes.tsv", str(dst),
                             args.reino, split, args.window, str(genome)))
        if faltan:
            print(f"  AVISO: {faltan} sin BED o sin genoma en {split}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (ok, sp, err) in enumerate(ex.map(_una_especie, trabajos), 1):
                if not ok:
                    print("  FALLO", sp, err, flush=True)
                if i % 10 == 0:
                    print(f"  {i}/{len(trabajos)}", flush=True)

    print("generando bundle (shards paralelos) ...", flush=True)
    procs = []
    for k in range(args.workers):
        jd = out / f"shard{k}"
        for split in args.splits.split(","):
            (jd / split).mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sorted((jsonl / split).glob("*.jsonl"))):
                if i % args.workers != k:
                    continue
                dst = jd / split / f.name
                if not dst.exists():
                    os.symlink(f.resolve(), dst)
        procs.append(subprocess.Popen(
            [sys.executable, str(HERE.parent / "training" / "make_sequence_bundle.py"),
             "--jsonl_dir", str(jd), "--genomes_dir", args.genomes_dir,
             "--out_dir", str(bundle), "--chunks", str(args.chunks),
             "--window_size", str(args.window), "--splits", args.splits]))
    for pr in procs:
        pr.wait()
    print("listo:", bundle, flush=True)


if __name__ == "__main__":
    main()
