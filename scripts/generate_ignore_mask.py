"""Genera BED de "ignorar" para el test (regla Simón punto 5).

Regla:
    "Regiones ambiguas (Unknown, satélites, baja complejidad) se marcan como
     IGNORAR en el test, no como negativo."

Para cada especie produce 3 BEDs:
    annotations/<sp>.bed              -> POSITIVO (TE anotado por RM)        (ya existe)
    annotations/<sp>.excluded.bed     -> AMBIGUO (simple_rep, low_complex,
                                                   satellite, Unknown)        (ya existe)
    annotations/<sp>.ignore.bed       -> IGNORAR TOTAL en test:
                                           = excluded.bed
                                           + buffer 1 kb alrededor de cada TE positivo
                                           + N-runs del genoma

Uso:
    python scripts/generate_ignore_mask.py <species>
    python scripts/generate_ignore_mask.py all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "annotations"

GENOME_PATHS = {
    "arabidopsis":  ROOT / "data" / "tair10_full"    / "TAIR10.fasta",
    "scerevisiae":  ROOT / "data" / "pilot_genomes"  / "scerevisiae.fasta",
    "celegans":     ROOT / "data" / "pilot_genomes"  / "celegans.fasta",
    "brachypodium": ROOT / "data" / "pilot_genomes"  / "brachypodium.fasta",
    "neurospora":   ROOT / "data" / "pilot_genomes"  / "neurospora.fasta",
    "drosophila":   ROOT / "data" / "pilot_genomes"  / "drosophila.fasta",
    "spombe":       ROOT / "data" / "pilot_genomes"  / "spombe.fasta",
    "aspergillus":  ROOT / "data" / "pilot_genomes"  / "aspergillus.fasta",
}

BUFFER_BP = 1000  # Simón: ≥1 kb


def read_bed(bed_file: Path):
    intervals = {}  # chrom -> list[(start, end)]
    if not bed_file.exists():
        return intervals
    with bed_file.open() as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 3:
                continue
            chrom = s[0]
            start = int(s[1])
            end = int(s[2])
            intervals.setdefault(chrom, []).append((start, end))
    for k in intervals:
        intervals[k].sort()
    return intervals


def merge_intervals(intervals: list) -> list:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def genome_n_runs(fa: Path, min_len: int = 100):
    """Detecta N-runs ≥ min_len bp."""
    runs = {}
    cur_chrom = None
    pos = 0
    nstart = None
    with fa.open() as f:
        for line in f:
            if line.startswith(">"):
                if cur_chrom and nstart is not None:
                    if pos - nstart >= min_len:
                        runs.setdefault(cur_chrom, []).append((nstart, pos))
                cur_chrom = line[1:].strip().split()[0]
                pos = 0
                nstart = None
            else:
                seq = line.strip().upper()
                for i, base in enumerate(seq):
                    gp = pos + i
                    if base == "N":
                        if nstart is None:
                            nstart = gp
                    else:
                        if nstart is not None and gp - nstart >= min_len:
                            runs.setdefault(cur_chrom, []).append((nstart, gp))
                        nstart = None
                pos += len(seq)
    if cur_chrom and nstart is not None and pos - nstart >= min_len:
        runs.setdefault(cur_chrom, []).append((nstart, pos))
    return runs


def generate_ignore(species: str):
    print(f"\n=== Ignore mask: {species} ===")

    bed_pos = ANN_DIR / f"{species}.bed"
    bed_exc = ANN_DIR / f"{species}.excluded.bed"
    out_bed = ANN_DIR / f"{species}.ignore.bed"
    out_sum = ANN_DIR / f"{species}.ignore.summary.json"

    if not bed_pos.exists():
        print(f"  Falta {bed_pos} (correr clean_rm_output.py primero)")
        return

    # 1. Positivos → generar buffer de 1kb a cada lado
    positives = read_bed(bed_pos)
    excluded = read_bed(bed_exc)

    # 2. N-runs del genoma
    n_runs = {}
    if species in GENOME_PATHS and GENOME_PATHS[species].exists():
        n_runs = genome_n_runs(GENOME_PATHS[species])

    # Combinar todo lo que va a "ignorar"
    ignore_by_chrom = {}
    counts = {"buffer_pos": 0, "excluded_rm": 0, "n_runs": 0}

    for chrom, lst in positives.items():
        for s, e in lst:
            # Buffer de BUFFER_BP a cada lado
            buf_s = max(0, s - BUFFER_BP)
            buf_e = e + BUFFER_BP
            ignore_by_chrom.setdefault(chrom, []).append((buf_s, buf_e))
            counts["buffer_pos"] += 1

    for chrom, lst in excluded.items():
        for s, e in lst:
            ignore_by_chrom.setdefault(chrom, []).append((s, e))
            counts["excluded_rm"] += 1

    for chrom, lst in n_runs.items():
        for s, e in lst:
            ignore_by_chrom.setdefault(chrom, []).append((s, e))
            counts["n_runs"] += 1

    # Fusionar intervalos solapados
    with out_bed.open("w") as fo:
        total_bp = 0
        total_elements = 0
        for chrom in sorted(ignore_by_chrom):
            merged = merge_intervals(ignore_by_chrom[chrom])
            for s, e in merged:
                fo.write(f"{chrom}\t{s}\t{e}\tignore\t0\t.\n")
                total_bp += e - s
                total_elements += 1

    summary = {
        "species": species,
        "ignore_elements": total_elements,
        "ignore_bp": total_bp,
        "source_counts": counts,
        "buffer_bp": BUFFER_BP,
    }
    out_sum.write_text(json.dumps(summary, indent=2))

    print(f"  {total_elements:,} intervalos → {total_bp:,} bp ignorados")
    print(f"  Fuentes: buffer={counts['buffer_pos']:,} TE  "
          f"excluded={counts['excluded_rm']:,}  N-runs={counts['n_runs']}")
    print(f"  Salida: annotations/{species}.ignore.bed")


def main():
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        species = sorted(set(GENOME_PATHS) & set(
            p.stem for p in ANN_DIR.glob("*.bed")
            if not p.stem.endswith(".excluded") and not p.stem.endswith(".ignore")
        ))
    else:
        species = args
    for sp in species:
        generate_ignore(sp)


if __name__ == "__main__":
    main()
