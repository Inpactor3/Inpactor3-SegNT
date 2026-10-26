"""Reporte de cobertura: especies con genoma vs especies con ventanas etiquetadas.

Sin acceso a los BED/GFF (disco externo) no se pueden generar ventanas nuevas;
este script deja la lista exacta de species_tag faltantes para relanzar el
pipeline downstream (BED -> dedup -> windows -> relabel -> consolidate).

Uso:
  python coverage_report.py --seq_dir seq_bundles --genomes_dir <genomes_449> \
      --out coverage_report.json
"""
import argparse
import json
from pathlib import Path


def species_in_bundles(seq_dir):
    found = set()
    for f in Path(seq_dir).rglob("*.npz"):
        found.add(f.stem)
    return found


def genomes_by_split(genomes_dir):
    out = {}
    for split in ("train", "val", "test"):
        d = Path(genomes_dir) / split
        if d.is_dir():
            out[split] = {f.name.replace(".fna.gz", "") for f in d.glob("*.fna.gz")}
    return out


def negative_types(seq_dir):
    """Cuenta tipos de negativas presentes en los JSONL/metadata del bundle."""
    import collections
    c = collections.Counter()
    for f in Path(seq_dir).rglob("*_meta.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if int(r.get("label", 1)) == 0:
                c[r.get("negative_type") or "sin_tipo"] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path, required=True)
    ap.add_argument("--genomes_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    have = species_in_bundles(args.seq_dir)
    gen = genomes_by_split(args.genomes_dir)
    rep = {"reino_dir": str(args.seq_dir), "n_especies_con_ventanas": len(have),
           "genomas_por_split": {k: len(v) for k, v in gen.items()},
           "faltantes": {}, "total_faltantes": 0}
    for split, gs in gen.items():
        miss = sorted(gs - have)
        rep["faltantes"][split] = miss
        rep["total_faltantes"] += len(miss)
    rep["genomas_sin_ventanas"] = rep["total_faltantes"]
    try:
        rep["negativas_por_tipo"] = negative_types(args.seq_dir)
    except Exception:
        rep["negativas_por_tipo"] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"especies con ventanas: {len(have)}")
    for split, miss in rep["faltantes"].items():
        print(f"  {split}: {len(miss)} sin ventanas")
    print("reporte:", args.out)


if __name__ == "__main__":
    main()
