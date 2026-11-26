#!/usr/bin/env python3
"""Filtra las especies rotas del CSV de taxonomía y lo guarda limpio.

Usa: python filtrar_especies_rotas.py --csv docs/animales_taxonomia.csv \
         --out docs/animales_taxonomia_limpio.csv --especies_a_eliminar

Las 3 especies a eliminar:
  - Ellobius_talpinus__GCA_001685095.1 (100% positiva)
  - Porites_divaricata__GCA_964035745.1 (100% positiva)
  - Dugesia_japonica__GCA_001938525.1 (100% negativa, vacía)

Resultado: 893.235 → 887.179 ventanas (6.056 menos, 0.68%)
"""
import argparse
import csv
import sys

ESPECIES_ROTAS = {
    "Ellobius_talpinus__GCA_001685095.1",
    "Porites_divaricata__GCA_964035745.1",
    "Dugesia_japonica__GCA_001938525.1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--especies-a-eliminar", action="store_true",
                    help="lista las especies a eliminar")
    args = ap.parse_args()

    if args.especies_a_eliminar:
        print("Especies a eliminar:")
        for sp in sorted(ESPECIES_ROTAS):
            print(f"  {sp}")
        return

    filas = list(csv.DictReader(open(args.csv)))
    filas_limpias = [f for f in filas if f["species"] not in ESPECIES_ROTAS]

    ventanas_antes = sum(int(f.get("n_windows", 0)) for f in filas)
    ventanas_despues = sum(int(f.get("n_windows", 0)) for f in filas_limpias)
    ventanas_quitadas = ventanas_antes - ventanas_despues

    print(f"Ventanas antes:   {ventanas_antes:,}")
    print(f"Ventanas después: {ventanas_despues:,}")
    print(f"Quitadas:         {ventanas_quitadas:,} ({ventanas_quitadas/ventanas_antes*100:.2f}%)")
    print(f"Especies antes:   {len(filas)}")
    print(f"Especies después: {len(filas_limpias)}")
    print()

    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas_limpias)

    print(f"CSV limpio guardado en: {args.out}")


if __name__ == "__main__":
    main()
