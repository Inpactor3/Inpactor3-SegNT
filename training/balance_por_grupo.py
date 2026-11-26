#!/usr/bin/env python3
"""OBSOLETO: este script solo calcula factores por grupo; no los aplica.

Para el experimento real usa train_group_weights.py, que aplica los pesos en la
loss (con cap y fusion de grupos pequenos) y reporta macro PR-AUC y lift.

Genera pesos por grupo para balance en entrenamiento.

Equilibra ventanas: cada grupo taxonómico tiene el mismo peso durante el
entrenamiento (1 / num_grupos).

Uso: python balance_por_grupo.py --csv docs/animales_taxonomia_limpio.csv \
         --out pesos_animales_balanceados.csv

Genera un CSV con peso de clase para cada ventana:
  - Insectos: multiplica por 2.4 (pasa de 8.7% a ~20% del lote)
  - Mamíferos: multiplica por 0.9 (pasa de 21% a ~20% del lote)
  - etc.

Integración:
  - El script train_mlp.py debe leer este CSV
  - En DataLoader, multiplica sample_weight por el peso del grupo
"""
import argparse
import csv
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="animales_taxonomia_limpio.csv")
    ap.add_argument("--out", default="pesos_animales_balanceados.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    filas = list(csv.DictReader(open(args.csv)))

    # Acumular ventanas por grupo
    por_grupo = defaultdict(int)
    for f in filas:
        grupo = f.get("grupo", "Sin clasificar")
        n_win = int(f.get("n_windows", 0))
        por_grupo[grupo] += n_win

    total_ventanas = sum(por_grupo.values())
    num_grupos = len(por_grupo)
    peso_uniforme = 1.0 / num_grupos

    # Calcular factores de peso
    pesos_grupo = {}
    for grupo, ventanas in por_grupo.items():
        pct_actual = ventanas / total_ventanas
        factor = peso_uniforme / pct_actual
        pesos_grupo[grupo] = factor

    if args.verbose:
        print("Distribución de pesos por grupo:")
        print(f"{'Grupo':30s} {'Ventanas':>9s} {'% Actual':>8s} {'% Target':>8s} {'Factor':>7s}")
        print("-" * 70)
        for grupo in sorted(por_grupo.keys(), key=lambda g: -por_grupo[g]):
            pct_actual = por_grupo[grupo] / total_ventanas * 100
            pct_target = peso_uniforme * 100
            factor = pesos_grupo[grupo]
            print(f"{grupo:30s} {por_grupo[grupo]:9d} {pct_actual:8.1f}% {pct_target:8.1f}% {factor:7.2f}x")
        print()

    # Generar CSV de pesos
    salida = []
    for f in filas:
        sp = f["species"]
        grupo = f.get("grupo", "Sin clasificar")
        peso = pesos_grupo.get(grupo, 1.0)
        salida.append({
            "species": sp,
            "grupo": grupo,
            "peso_grupo": round(peso, 3),
        })

    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["species", "grupo", "peso_grupo"])
        w.writeheader()
        w.writerows(salida)

    print(f"CSV de pesos guardado en: {args.out}")
    print(f"Úsalo en train_mlp.py con: --group_weights {args.out}")


if __name__ == "__main__":
    main()
