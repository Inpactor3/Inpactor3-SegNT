"""Filtra la biblioteca PanTEon por especie.

Uso:
    python filter_panteon_by_species.py "Arabidopsis thaliana" out.fasta
"""
import sys
from pathlib import Path

PANTEON = Path("data/raw/PanTEon_Database_v1.6.2.fasta")


def filter_species(target_species: str, out_path: Path):
    kept = 0
    total = 0
    writing = False
    with PANTEON.open() as fi, out_path.open("w") as fo:
        for line in fi:
            if line.startswith(">"):
                total += 1
                # Header format: >PDB...#CLASSX/.../... @Species name
                sp = line.split("@", 1)[1].strip() if "@" in line else ""
                writing = sp.lower() == target_species.lower()
                if writing:
                    kept += 1
                    fo.write(line)
            else:
                if writing:
                    fo.write(line)
    return kept, total


if __name__ == "__main__":
    target = sys.argv[1]
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, total = filter_species(target, out)
    print(f"[filter] {target}: {kept} / {total} consensos extraidos -> {out}")
