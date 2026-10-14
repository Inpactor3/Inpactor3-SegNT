"""Crea bibliotecas PanTEon por reino para usar como fallback.

Para especies con <30 consensos propios, se usa la biblioteca completa del reino.

Uso:
    python filter_panteon_by_kingdom.py
"""
import csv
from pathlib import Path

PANTEON_FA = Path("data/raw/PanTEon_Database_v1.6.2.fasta")
METADATA   = Path("data/raw/PanTEon_Database_metadata_v1.6.2.csv")
OUT_DIR    = Path("data/panteon_libraries")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Mapear cada especie a su reino
    sp_to_kingdom = {}
    with METADATA.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sp_to_kingdom[row["Species"]] = row["Kingdom"]

    # 2. Abrir archivos de salida (uno por reino)
    outs = {
        k: (OUT_DIR / f"panteon_{k.lower()}.fasta").open("w")
        for k in ("Animalia", "Fungi", "Plantae")
    }
    counts = {k: 0 for k in outs}

    # 3. Recorrer el fasta y repartir
    current_kingdom = None
    writing = False
    with PANTEON_FA.open() as fi:
        for line in fi:
            if line.startswith(">"):
                if "@" in line:
                    sp = line.split("@", 1)[1].strip()
                    current_kingdom = sp_to_kingdom.get(sp)
                else:
                    current_kingdom = None
                writing = current_kingdom in outs
                if writing:
                    outs[current_kingdom].write(line)
                    counts[current_kingdom] += 1
            else:
                if writing:
                    outs[current_kingdom].write(line)

    for h in outs.values():
        h.close()

    print("=== Bibliotecas por reino ===")
    for k, n in counts.items():
        path = OUT_DIR / f"panteon_{k.lower()}.fasta"
        size = path.stat().st_size / 1e6
        print(f"  {k:10s}: {n:7d} consensos  ({size:6.1f} MB)  {path}")


if __name__ == "__main__":
    main()
