"""Genera ventanas de 50 kb con etiquetas binarias desde el genoma + BED.

Reglas (base comun Simon):
- Ventana de 50 kb teselada (sin solape para train, solape 50% para dev/test)
- Positiva si existe TE con solape >= max(50 bp, 20% de la longitud del elemento)
- Para negativas usa el muestreo de las 5 categorias ya generado
- Guarda metadata: n_TE, fraccion cubierta, ordenes presentes

Salida: data/windows_pilot/<species>.jsonl
    Un JSON por ventana con todos los campos exigidos por Simon.

Uso:
    python scripts/generate_windows.py <species> [--split train|devtest]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "annotations"
OUT_DIR = ROOT / "data" / "windows_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENOME_PATHS = {
    "arabidopsis":  ROOT / "data" / "tair10_full"    / "TAIR10.fasta",
    "scerevisiae":  ROOT / "data" / "pilot_genomes"  / "scerevisiae.fasta",
    "celegans":     ROOT / "data" / "pilot_genomes"  / "celegans.fasta",
    "brachypodium": ROOT / "data" / "pilot_genomes"  / "brachypodium.fasta",
    "neurospora":   ROOT / "data" / "pilot_genomes"  / "neurospora.fasta",
    "drosophila":   ROOT / "data" / "pilot_genomes"  / "drosophila.fasta",
}

KINGDOM = {
    "arabidopsis":  "Plantae",
    "brachypodium": "Plantae",
    "scerevisiae":  "Fungi",
    "neurospora":   "Fungi",
    "celegans":     "Animalia",
    "drosophila":   "Animalia",
}

WINDOW_SIZE = 50_000
STRIDE_TRAIN = 50_000      # sin solape
STRIDE_DEVTEST = 25_000    # solape 50%
MIN_TE_OVERLAP_BP = 50     # umbral absoluto Simon
MIN_TE_OVERLAP_FRAC = 0.2  # umbral relativo Simon


def genome_sizes(fa: Path) -> dict:
    sizes = {}
    cur, acc = None, 0
    with fa.open() as f:
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
    return sizes


def read_bed(bed_file: Path):
    """Devuelve dict chrom -> list[(start, end, name_tokens)]
       name_tokens = ['PDB...', 'LTR', 'GYPSY']
    """
    hits = {}
    if not bed_file.exists():
        return hits
    with bed_file.open() as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 4:
                continue
            chrom = s[0]
            start = int(s[1])
            end = int(s[2])
            name_tokens = s[3].split("|")
            hits.setdefault(chrom, []).append((start, end, name_tokens))
    for k in hits:
        hits[k].sort()
    return hits


def window_positives(chrom: str, w_start: int, w_end: int, te_hits: dict):
    """Para una ventana, mira si contiene un TE que supere el umbral.

    Devuelve (is_positive, n_TE_total, bp_cubiertos, ordenes_presentes).
    """
    hits_in = [h for h in te_hits.get(chrom, [])
               if h[0] < w_end and h[1] > w_start]
    if not hits_in:
        return False, 0, 0, []

    is_positive = False
    bp_total = 0
    ordenes = set()
    for s, e, name_tokens in hits_in:
        overlap = min(e, w_end) - max(s, w_start)
        if overlap <= 0:
            continue
        elem_len = e - s
        bp_total += overlap
        if len(name_tokens) >= 2:
            ordenes.add(name_tokens[1])
        # Regla positiva: solape >= max(50 bp, 20% del elemento)
        threshold = max(MIN_TE_OVERLAP_BP, int(MIN_TE_OVERLAP_FRAC * elem_len))
        if overlap >= threshold:
            is_positive = True
    return is_positive, len(hits_in), bp_total, sorted(ordenes)


def tile_species(species: str, split: str = "train"):
    print(f"\n=== Teselado {species} ({split}) ===")

    genome = GENOME_PATHS[species]
    bed = ANN_DIR / f"{species}.bed"
    out_file = OUT_DIR / f"{species}.{split}.jsonl"

    if not genome.exists():
        print(f"  Falta genoma: {genome}")
        return
    if not bed.exists():
        print(f"  Falta BED: {bed}  (correr clean_rm_output.py)")
        return

    sizes = genome_sizes(genome)
    te_hits = read_bed(bed)

    stride = STRIDE_DEVTEST if split == "devtest" else STRIDE_TRAIN

    n_total = 0
    n_pos = 0
    categorias = Counter()
    ordenes_global = Counter()

    with out_file.open("w") as fo:
        for chrom, size in sizes.items():
            for start in range(0, size - WINDOW_SIZE + 1, stride):
                end = start + WINDOW_SIZE
                is_pos, n_te, bp_cov, ords = window_positives(
                    chrom, start, end, te_hits)
                label = 1 if is_pos else 0
                rec = {
                    "window_id": f"{species}_{chrom}_{start}",
                    "species": species,
                    "kingdom": KINGDOM.get(species, "?"),
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "label": label,
                    "n_TE": n_te,
                    "bp_TE": bp_cov,
                    "coverage_fraction": round(bp_cov / WINDOW_SIZE, 4),
                    "orders_present": ords,
                }
                fo.write(json.dumps(rec) + "\n")
                n_total += 1
                if is_pos:
                    n_pos += 1
                    categorias["positive"] += 1
                    for o in ords:
                        ordenes_global[o] += 1
                else:
                    categorias["negative"] += 1

    print(f"  ventanas totales: {n_total:,}")
    print(f"  positivas:        {n_pos:,}  ({100*n_pos/max(n_total,1):.1f}%)")
    print(f"  negativas:        {n_total - n_pos:,}")
    print(f"  ordenes en pos:   ", end="")
    for o, n in ordenes_global.most_common(5):
        print(f"{o}={n}  ", end="")
    print()
    print(f"  salida: {out_file.name}")

    # Summary
    out_sum = OUT_DIR / f"{species}.{split}.summary.json"
    out_sum.write_text(json.dumps({
        "species": species,
        "split": split,
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_total - n_pos,
        "pos_fraction": round(n_pos / max(n_total, 1), 4),
        "orders_in_positives": dict(ordenes_global.most_common()),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="?", default="all")
    ap.add_argument("--split", default="train", choices=["train", "devtest"])
    args = ap.parse_args()

    if args.species == "all":
        species = sorted(set(GENOME_PATHS) & set(
            p.stem for p in ANN_DIR.glob("*.bed")
            if not p.name.endswith(".excluded.bed")
        ))
    else:
        species = [args.species]

    for sp in species:
        tile_species(sp, split=args.split)


if __name__ == "__main__":
    main()
