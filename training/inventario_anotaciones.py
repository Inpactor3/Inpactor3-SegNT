#!/usr/bin/env python3
"""Inventario de genomas, BED/RepeatMasker, consolidados y ventanas por especie.

Sirve para saber, antes de pedir o correr RepeatMasker, qué especies ya tienen
anotación utilizable y cuáles no. Tambien detecta BED de un ensamblado distinto
al genoma disponible.

Uso:
  python inventario_anotaciones.py --root /ruta/InpactorRM --out inventario.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

ACC = re.compile(r"(GC[AF]_\d+\.\d+)")
KINGDOMS = ("Animales", "Plantas", "Hongos")


def accession(path):
    m = ACC.search(os.path.basename(path))
    return m.group(1) if m else None


def species_name(path):
    stem = os.path.basename(path)
    for suffix in (".fna.gz", ".bed", ".jsonl", ".npz", ".fa.gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = ACC.sub("", stem)
    return stem.strip().rstrip("_")


def collect(pattern, key="acc"):
    out = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        acc = accession(path)
        if not acc:
            continue
        out[acc if key == "acc" else os.path.basename(path)].append(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raiz con genomes_449/, annotations/, consolidated/, windows_ltr/")
    ap.add_argument("--out", default="inventario_anotaciones.csv")
    args = ap.parse_args()

    rows = []
    summary = []
    for kingdom in KINGDOMS:
        genomes = collect(os.path.join(args.root, "genomes_449", kingdom, "*", "*.fna.gz"))
        beds = collect(os.path.join(args.root, "annotations", kingdom, "*", "*.bed"))
        cons = collect(os.path.join(args.root, "consolidated", kingdom, "*", "*.jsonl"))
        wins = collect(os.path.join(args.root, "windows_ltr", kingdom, "*", "*"))

        genome_accs = set(genomes)
        for acc in sorted(genome_accs | set(beds) | set(cons) | set(wins)):
            g = genomes.get(acc, [""])[0]
            b = beds.get(acc, [""])[0]
            c = cons.get(acc, [""])[0]
            w = wins.get(acc, [""])[0]
            split = ""
            for p in (g, c, b, w):
                if p:
                    split = os.path.basename(os.path.dirname(p))
                    break
            rows.append({
                "kingdom": kingdom,
                "species": species_name(g or b or c or w),
                "accession": acc,
                "split": split,
                "genome": g,
                "bed": b,
                "consolidated": c,
                "windows_ltr": w,
                "assembly_match": "YES" if (b and acc in genome_accs) else ("NO" if b else ""),
                "bed_bytes": os.path.getsize(b) if b else 0,
                "n_bed_files": len(beds.get(acc, [])),
            })

        missing = genome_accs - set(cons)
        with_bed = {a for a in missing if a in beds and a in genome_accs}
        mismatch = {a for a in beds if a not in genome_accs}
        summary.append({
            "kingdom": kingdom,
            "genomes": len(genome_accs),
            "consolidated": len(cons),
            "missing": len(missing),
            "missing_with_bed": len(with_bed),
            "missing_without_bed": len(missing - with_bed),
            "bed_without_matching_genome": len(mismatch),
            "beds": len(beds),
            "windows_ltr": len(wins),
        })

        if mismatch:
            print(f"\n[{kingdom}] BED sin genoma del mismo ensamblado:")
            for acc in sorted(mismatch):
                p = beds[acc][0]
                sp = species_name(p).lower()
                same = sorted(a for a, paths in genomes.items()
                              if species_name(paths[0]).lower() == sp)
                print(f"  {os.path.basename(p)}  ({len(beds[acc])} archivo(s))")
                if same:
                    print("    genomas de la misma especie en disco: "
                          + ", ".join(f"{a} ({os.path.basename(genomes[a][0])})" for a in same))
                else:
                    print("    no hay genoma de esa especie en disco")

    fields = ["kingdom", "species", "accession", "split", "genome", "bed", "consolidated",
              "windows_ltr", "assembly_match", "bed_bytes", "n_bed_files"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nInventario escrito en: {args.out}  ({len(rows)} especies)\n")
    header = f"{'Reino':10s} {'Genomas':>7s} {'Consol.':>7s} {'Faltan':>6s} {'+BED':>5s} {'sin BED':>7s} {'BED mal':>7s}"
    print(header)
    print("-" * len(header))
    for s in summary:
        print(f"{s['kingdom']:10s} {s['genomes']:7d} {s['consolidated']:7d} {s['missing']:6d} "
              f"{s['missing_with_bed']:5d} {s['missing_without_bed']:7d} {s['bed_without_matching_genome']:7d}")
    print("\n'+BED' = especie sin consolidar pero con BED y genoma del mismo ensamblado (se puede procesar ya).")
    print("'BED mal' = BED cuyo ensamblado no coincide con ningún genoma disponible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
