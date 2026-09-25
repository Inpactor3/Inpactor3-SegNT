"""
Como solo tenemos Chr1 con 16 anotaciones, dividimos ese cromosoma en
3 pseudo-scaffolds para permitir un split rigoroso train/val/test:

  Chr1_train : bp        0 - 14 500 000  (~14.5 Mb, contiene 10 elementos)
  Chr1_val   : bp 14 500 000 - 17 000 000 (~ 2.5 Mb, contiene  3 elementos)
  Chr1_test  : bp 17 000 000 - 20 000 000 (~ 3.0 Mb, contiene  3 elementos)

Las coordenadas de las anotaciones se re-mapean al inicio de cada nuevo
pseudo-scaffold (restando el offset).

Salidas:
    data/raw/TAIR10_pseudo.fasta          FASTA con 3 pseudo-scaffolds
    data/raw/Inpactor2_predictions_pseudo.tab  anotaciones re-mapeadas

Nota metodológica: esto es un piloto. Cuando se pueda correr Inpactor2
sobre TAIR10 completo (5 cromosomas), se descarta este preprocesamiento
y se usa la división por cromosoma real.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Definición del split (ajustado para balance ~10/3/3 anotaciones)
SPLITS = [
    ("Chr1_train", 0, 14_400_000),
    ("Chr1_val", 14_400_000, 17_000_000),
    ("Chr1_test", 17_000_000, 20_000_000),
]


def read_fasta_single(path: Path) -> tuple[str, str]:
    header, buf = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    return header, "".join(buf).upper()
                header = line[1:].split()[0]
            else:
                buf.append(line.strip())
    return header, "".join(buf).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=Path, default=Path("data/raw/TAIR10_chr1.fasta"))
    ap.add_argument("--tab", type=Path, default=Path("data/raw/Inpactor2_predictions.tab"))
    ap.add_argument("--out-fasta", type=Path, default=Path("data/raw/TAIR10_pseudo.fasta"))
    ap.add_argument("--out-tab", type=Path, default=Path("data/raw/Inpactor2_predictions_pseudo.tab"))
    args = ap.parse_args()

    scaffold, seq = read_fasta_single(args.genome)
    print(f"[read] {scaffold} · {len(seq):,} bp")

    # Escribir FASTA de pseudo-scaffolds
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_fasta, "w") as f:
        for name, s, e in SPLITS:
            chunk = seq[s:e]
            f.write(f">{name}\n")
            for k in range(0, len(chunk), 80):
                f.write(chunk[k : k + 80] + "\n")
            print(f"  {name:12s} bp {s:>10,} – {e:>10,} · {len(chunk):,} bp")
    print(f"[out ] {args.out_fasta}")

    # Re-mapear anotaciones
    per_split = {name: 0 for name, _, _ in SPLITS}
    with open(args.tab) as fin, open(args.out_tab, "w") as fout:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue
            _, start, end, length, lin, *rest = cols
            start, end = int(start), int(end)
            # Localizar en qué split cae
            placed = False
            for name, s, e in SPLITS:
                if start >= s and end <= e:
                    new_start = start - s
                    new_end = end - s
                    rest_s = "\t".join(rest) if rest else "-\t-\t-"
                    fout.write(
                        f"{name}\t{new_start}\t{new_end}\t{length}\t{lin}\t{rest_s}\n"
                    )
                    per_split[name] += 1
                    placed = True
                    break
            if not placed:
                # elemento que cruza límite de split → lo descartamos
                print(f"  [skip] {start}-{end} cruza límite entre pseudo-scaffolds")

    print(f"[out ] {args.out_tab}")
    for name, n in per_split.items():
        print(f"  {name:12s} anotaciones: {n}")


if __name__ == "__main__":
    raise SystemExit(main())
