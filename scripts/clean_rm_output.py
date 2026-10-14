"""Limpia la salida de RepeatMasker + PanTEon y genera un BED estandarizado.

Entradas:
    results/rm_pilots/<species>/*.out

Salidas:
    annotations/<species>.bed              -> coordenadas limpias
    annotations/<species>.excluded.bed     -> buffer y clases excluidas
    annotations/<species>.summary.json     -> metricas de limpieza

Reglas (base comun de Simon):
    1. Excluir Simple_repeat, Low_complexity y Unknown.
    2. Excluir satelite (Satellite/*).
    3. Mapear cada clase PanTEon a un orden estandar:
        CLASSI/LTR/*       -> LTR
        CLASSI/LINE/*      -> LINE
        CLASSI/SINE/*      -> SINE
        CLASSI/*           -> ClassI_other
        CLASSII/HELITRON/* -> Helitron
        CLASSII/TIR/*      -> TIR
        CLASSII/*          -> ClassII_other
    4. Buffer >=1 kb alrededor de cada hit no se marca; se usa en el muestreo
       de negativas despues (no es tarea de limpieza).

Uso:
    python scripts/clean_rm_output.py <species>
    python scripts/clean_rm_output.py all
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RM_DIR = ROOT / "results" / "rm_pilots"
ANN_DIR = ROOT / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)


# Clases a EXCLUIR del BED (Simon: no son ni positivo ni negativo)
EXCLUDE_PREFIX = (
    "Simple_repeat",
    "Low_complexity",
    "Unknown",
    "Satellite",
    "rRNA",
    "tRNA",
    "snRNA",
    "scRNA",
)


def map_class(rm_class: str) -> tuple[str, str]:
    """Mapea la cadena class/family de PanTEon a (orden, superfamilia).

    Devuelve (orden, superfamilia). Si no se puede mapear devuelve ("Unknown", "Unknown").
    """
    if "/" not in rm_class:
        # p.ej. "DNA" o "LINE" solo (menos comun con PanTEon)
        return rm_class, rm_class

    parts = rm_class.split("/")
    # PanTEon: CLASSI/LTR/GYPSY  o  CLASSII/HELITRON/HELITRON
    if parts[0].upper() in {"CLASSI", "CLASS_I"}:
        if len(parts) >= 3:
            return parts[1].upper(), parts[2].upper()   # LTR, GYPSY
        return parts[1].upper(), parts[1].upper()
    if parts[0].upper() in {"CLASSII", "CLASS_II"}:
        if len(parts) >= 3:
            return parts[1].upper(), parts[2].upper()
        return parts[1].upper(), parts[1].upper()
    # Formato alternativo RepBase (LTR/Gypsy)
    return parts[0].upper(), parts[-1].upper()


def parse_out(out_file: Path):
    """Yield (chrom, start, end, strand, repeat, orden, superfam, pct_div)."""
    with out_file.open() as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 11:
                continue
            try:
                _ = int(s[5])
            except (ValueError, IndexError):
                continue
            try:
                pct_div = float(s[1])
                chrom = s[4]
                qstart = int(s[5])
                qend = int(s[6])
                strand = "+" if s[8] == "+" else "-"
                repeat = s[9]
                rm_class = s[10]
            except (ValueError, IndexError):
                continue
            orden, superfam = map_class(rm_class)
            yield (chrom, qstart, qend, strand, repeat, orden, superfam,
                   pct_div, rm_class)


def clean_species(species: str) -> dict:
    sp_dir = RM_DIR / species
    out_files = list(sp_dir.glob("*.out"))
    if not out_files:
        print(f"[clean] {species}: NO hay .out en {sp_dir}")
        return {}

    out_file = out_files[0]
    bed_path = ANN_DIR / f"{species}.bed"
    exc_path = ANN_DIR / f"{species}.excluded.bed"
    summary_path = ANN_DIR / f"{species}.summary.json"

    kept = 0
    excluded = 0
    by_order = Counter()
    by_superfam = Counter()
    by_exclude = Counter()
    total_bp_kept = 0

    with bed_path.open("w") as fo, exc_path.open("w") as fx:
        for chrom, s0, s1, strand, rep, orden, sf, div, raw_class in parse_out(out_file):
            start_bed = s0 - 1  # BED es 0-based
            end_bed = s1
            # Reglas de exclusion
            excluded_now = False
            for pref in EXCLUDE_PREFIX:
                if raw_class.startswith(pref):
                    excluded_now = True
                    by_exclude[pref] += 1
                    break
            if excluded_now:
                fx.write(f"{chrom}\t{start_bed}\t{end_bed}\t{rep}|{raw_class}\t"
                         f"{div:.2f}\t{strand}\n")
                excluded += 1
            else:
                name = f"{rep}|{orden}|{sf}"
                fo.write(f"{chrom}\t{start_bed}\t{end_bed}\t{name}\t"
                         f"{div:.2f}\t{strand}\n")
                kept += 1
                total_bp_kept += (end_bed - start_bed)
                by_order[orden] += 1
                by_superfam[sf] += 1

    summary = {
        "species": species,
        "out_file": str(out_file),
        "kept_elements": kept,
        "excluded_elements": excluded,
        "total_bp_kept": total_bp_kept,
        "by_order": dict(by_order.most_common()),
        "by_superfamily_top10": dict(by_superfam.most_common(10)),
        "by_excluded_reason": dict(by_exclude.most_common()),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def main():
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        species = sorted(p.name for p in RM_DIR.iterdir() if p.is_dir())
    else:
        species = args

    for sp in species:
        print(f"\n=== Limpiando {sp} ===")
        summary = clean_species(sp)
        if not summary:
            continue
        print(f"  elementos conservados: {summary['kept_elements']:,}")
        print(f"  elementos excluidos:   {summary['excluded_elements']:,}")
        print(f"  bp cubiertos (kept):   {summary['total_bp_kept']:,}")
        print(f"  por orden: ", end="")
        for k, v in list(summary["by_order"].items())[:6]:
            print(f"{k}={v:,}  ", end="")
        print()
        print(f"  BED limpio: annotations/{sp}.bed")


if __name__ == "__main__":
    main()
