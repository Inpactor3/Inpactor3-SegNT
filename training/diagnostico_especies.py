#!/usr/bin/env python3
"""Diagnóstico por especie individual: qué está roto, qué desbalancea.

Uso:
  python diagnostico_especies.py --csv docs/animales_taxonomia.csv \
      --out diagnostico_animales.csv

Genera un CSV con:
  - Especies 100% positivas (inútiles para entrenar)
  - Especies 100% negativas (inútiles para entrenar)
  - Desbalance por grupo
  - Distribución train/val/test
"""
import argparse
import csv
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="animales_taxonomia.csv")
    ap.add_argument("--out", default=None, help="output CSV con diagnóstico")
    args = ap.parse_args()

    filas = list(csv.DictReader(open(args.csv)))

    # Por grupo: acumular stats
    por_grupo = defaultdict(lambda: {
        "especies": 0,
        "ventanas": 0,
        "pos": 0,
        "neg": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "especies_rotas": [],
        "especies_lista": []
    })

    diagnostico = []

    for f in filas:
        sp = f["species"]
        grupo = f.get("grupo", "Sin clasificar")
        n_win = int(f.get("n_windows", 0))
        n_pos = int(f.get("n_pos", 0))
        n_neg = int(f.get("n_neg", 0))
        pct_pos = float(f.get("pct_pos", 0))
        split = f.get("split", "unknown")

        # Flags de diagnóstico
        roto = False
        razon = ""

        if pct_pos == 0.0:
            roto = True
            razon = "100% negativa (no aprende TE)"
        elif pct_pos == 100.0:
            roto = True
            razon = "100% positiva (no aprende negativa)"

        por_grupo[grupo]["especies"] += 1
        por_grupo[grupo]["ventanas"] += n_win
        por_grupo[grupo]["pos"] += n_pos
        por_grupo[grupo]["neg"] += n_neg

        split_map = {"train": "train", "validation": "val", "test": "test"}
        split_col = split_map.get(split.lower(), split.lower())
        por_grupo[grupo][split_col] = por_grupo[grupo].get(split_col, 0) + n_win

        if roto:
            por_grupo[grupo]["especies_rotas"].append((sp, razon, n_win))
        else:
            por_grupo[grupo]["especies_lista"].append(sp)

        diagnostico.append({
            "especie": sp,
            "grupo": grupo,
            "split": split,
            "ventanas": n_win,
            "positivas": n_pos,
            "negativas": n_neg,
            "pct_pos": round(pct_pos, 1),
            "roto": "SÍ" if roto else "NO",
            "razon": razon,
        })

    # Reporte por grupo
    print("\n" + "="*90)
    print("DIAGNÓSTICO POR GRUPO TAXONÓMICO")
    print("="*90 + "\n")
    print(f"{'Grupo':30s} {'Esp':>4s} {'Ventanas':>9s} {'% Pos':>6s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Rotas':>5s}")
    print("-"*90)

    grupos_ordenados = sorted(por_grupo.items(),
                             key=lambda kv: -kv[1]["ventanas"])

    for grupo, stats in grupos_ordenados:
        pct_pos_grupo = (stats["pos"] / stats["ventanas"] * 100) if stats["ventanas"] > 0 else 0
        print(f"{grupo:30s} {stats['especies']:4d} {stats['ventanas']:9d} "
              f"{pct_pos_grupo:6.1f} {stats.get('train', 0):7d} {stats.get('val', 0):7d} "
              f"{stats.get('test', 0):7d} {len(stats['especies_rotas']):5d}")

    # Resumen total
    total_esp = sum(s["especies"] for s in por_grupo.values())
    total_win = sum(s["ventanas"] for s in por_grupo.values())
    total_pos = sum(s["pos"] for s in por_grupo.values())
    pct_total = (total_pos / total_win * 100) if total_win > 0 else 0
    total_train = sum(s.get("train", 0) for s in por_grupo.values())
    total_val = sum(s.get("val", 0) for s in por_grupo.values())
    total_test = sum(s.get("test", 0) for s in por_grupo.values())
    total_rotas = sum(len(s["especies_rotas"]) for s in por_grupo.values())

    print("-"*90)
    print(f"{'TOTAL':30s} {total_esp:4d} {total_win:9d} {pct_total:6.1f} "
          f"{total_train:7d} {total_val:7d} {total_test:7d} {total_rotas:5d}")

    # Especies rotas por grupo
    print("\n" + "="*90)
    print("ESPECIES ROTAS (100% positiva/negativa)")
    print("="*90 + "\n")

    hay_rotas = False
    for grupo, stats in grupos_ordenados:
        if stats["especies_rotas"]:
            hay_rotas = True
            print(f"\n{grupo} ({len(stats['especies_rotas'])} rotas):")
            for sp, razon, n_win in sorted(stats["especies_rotas"], key=lambda x: -x[2])[:10]:
                print(f"  {sp:50s} {n_win:7d} ventanas · {razon}")
            if len(stats["especies_rotas"]) > 10:
                print(f"  ... y {len(stats['especies_rotas']) - 10} más")

    if not hay_rotas:
        print("✓ No hay especies 100% positivas o negativas.\n")

    # Desbalance por grupo
    print("\n" + "="*90)
    print("DESBALANCE POR GRUPO (% de ventanas totales)")
    print("="*90 + "\n")

    distribuciones = [(g, s["ventanas"] / total_win * 100)
                      for g, s in grupos_ordenados]
    for grupo, pct in distribuciones:
        barra = "█" * int(pct / 2)
        print(f"{grupo:30s} {pct:6.1f}%  {barra}")

    # Recomendaciones
    print("\n" + "="*90)
    print("RECOMENDACIONES")
    print("="*90 + "\n")

    max_grupo_pct = max(pct for _, pct in distribuciones)
    grupo_dominante = [g for g, pct in distribuciones if pct == max_grupo_pct][0]

    print(f"1. {grupo_dominante} domina con {max_grupo_pct:.1f}% de ventanas.")
    print(f"   → Rebalancear por grupo ayudará a Insectos/Moluscos.\n")

    if total_rotas > 0:
        print(f"2. {total_rotas} especies rotas → SACAR ANTES de entrenar.\n")
    else:
        print(f"2. No hay especies rotas → Seguir adelante.\n")

    # Escribir CSV si se solicita
    if args.out:
        diagnostico_sorted = sorted(diagnostico,
                                   key=lambda r: (-int(r["ventanas"]), r["grupo"]))
        with open(args.out, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(diagnostico_sorted[0].keys()))
            w.writeheader()
            w.writerows(diagnostico_sorted)
        print(f"CSV detallado guardado en: {args.out}\n")


if __name__ == "__main__":
    main()
