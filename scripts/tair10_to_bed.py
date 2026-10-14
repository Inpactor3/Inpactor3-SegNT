"""Convierte TAIR10_Transposable_Elements.txt a BED estandar.

El formato de TAIR codifica el cromosoma en el nombre:
    AT1TE52125  = Chr1 TE
    AT2TE...    = Chr2
    etc.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "curated_te" / "tair10_te.txt"
OUT = ROOT / "data" / "curated_te" / "tair10_te.bed"

NAME_RE = re.compile(r"AT(\d|[MC])TE\d+", re.IGNORECASE)

n_ok = 0
n_bad = 0
with IN.open() as fi, OUT.open("w") as fo:
    for i, line in enumerate(fi):
        if i == 0:
            continue  # header
        s = line.strip().split("\t")
        if len(s) < 6:
            n_bad += 1
            continue
        name, orient, start, end, family, superfam = s[:6]
        m = NAME_RE.match(name)
        if not m:
            n_bad += 1
            continue
        chrom_tag = m.group(1).upper()
        if chrom_tag == "M":
            chrom = "ChrM"
        elif chrom_tag == "C":
            chrom = "ChrC"
        else:
            chrom = f"Chr{chrom_tag}"
        try:
            start_bed = int(start) - 1
            end_bed = int(end)
        except ValueError:
            n_bad += 1
            continue
        strand = "+" if orient.lower() == "true" else "-"
        label = f"{name}|{superfam}|{family}"
        fo.write(f"{chrom}\t{start_bed}\t{end_bed}\t{label}\t0\t{strand}\n")
        n_ok += 1

print(f"[bed] {OUT}")
print(f"TEs curados convertidos: {n_ok}")
print(f"Lineas descartadas:      {n_bad}")
