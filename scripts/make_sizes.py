"""Calcula tamaños de cromosoma por especie a partir de los .fna.gz del USB.
Salida: un TSV por especie con lineas 'chrom<TAB>length'.
Uso: python3 make_sizes.py <genomes_root> <bed_root> <out_root> <reino>
Solo procesa especies que tienen BED (para no leer genomas innecesarios).
"""
import gzip
import sys
from pathlib import Path

genomes_root, bed_root, out_root, reino = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]

def sizes_of(gz_path: Path):
    sizes = {}
    cur, acc = None, 0
    with gzip.open(gz_path, "rt", errors="ignore") as f:
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

count = 0
for split in ("train", "val", "test"):
    beds = list((bed_root / split).glob("*.bed"))
    for bed in beds:
        tag = bed.stem
        gz = genomes_root / split / f"{tag}.fna.gz"
        if not gz.exists():
            # buscar en otros splits
            hits = list(genomes_root.rglob(f"{tag}.fna.gz"))
            if not hits:
                print(f"SKIP sin genoma: {tag}")
                continue
            gz = hits[0]
        out = out_root / split / f"{tag}.tsv"
        if out.exists() and out.stat().st_size > 0:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        s = sizes_of(gz)
        with out.open("w") as fo:
            for chrom, L in s.items():
                fo.write(f"{chrom}\t{L}\n")
        count += 1
        print(f"OK {split}/{tag} ({len(s)} cromosomas)")

print(f"Total sizes generados: {count} ({reino})")
