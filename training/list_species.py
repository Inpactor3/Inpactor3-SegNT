"""Lista las especies de un reino con sus ventanas y etiquetas.

Sirve para revisar la composicion del dataset antes de cambiar de estrategia
(por ejemplo, ver que especies aportan negativas y cuales no).

Uso:
  python list_species.py --seq_dir ../../seq_bundles4/Animales \
      --out animales_especies.csv

  # opcional: comparar con lo que realmente se entreno (train submuestreado)
  python list_species.py --seq_dir ../../seq_bundles4/Animales \
      --sub_dir ../../trabajo/sub/Animales --out animales_especies.csv
"""
import argparse
import csv
import glob
import os

import numpy as np


def leer_split(seq_dir, split):
    filas = {}
    for f in sorted(glob.glob(os.path.join(seq_dir, split, "*.npz"))):
        tag = os.path.basename(f)[:-4]
        z = np.load(f, allow_pickle=True)
        y = z["label"]
        n = int(len(y))
        pos = int(y.sum())
        filas[tag] = {
            "species": tag,
            "split": split,
            "n_windows": n,
            "n_pos": pos,
            "n_neg": n - pos,
            "pct_pos": round(100.0 * pos / n, 1) if n else 0.0,
            "region_bp": int(z["seq"].shape[1] * int(z["chunk_bp"])) if n else 0,
            "n_1a1": (n - pos) + min(pos, n - pos) if n and (n - pos) > 0
                     else min(pos, 2000),
            "n_train_used": "",
        }
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True,
                    help="directorio con train/ val/ test/ de npz")
    ap.add_argument("--sub_dir", default=None,
                    help="opcional: dir del train submuestreado (trabajo/sub/<Reino>)")
    ap.add_argument("--out", required=True, help="CSV de salida")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    filas = {}
    for split in args.splits.split(","):
        filas.update(leer_split(args.seq_dir, split))

    if args.sub_dir and os.path.isdir(os.path.join(args.sub_dir, "train")):
        for f in sorted(glob.glob(os.path.join(args.sub_dir, "train", "*.npz"))):
            tag = os.path.basename(f)[:-4]
            if tag in filas:
                filas[tag]["n_train_used"] = int(len(np.load(f, allow_pickle=True)["label"]))

    campos = ["species", "split", "n_windows", "n_pos", "n_neg", "pct_pos",
              "region_bp", "n_1a1", "n_train_used"]
    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for tag in sorted(filas):
            w.writerow(filas[tag])

    # --- resumen por split ---
    print(f"{'split':6s} {'especies':>9s} {'ventanas':>10s} {'pos':>9s} {'neg':>9s} {'% pos':>7s}")
    for split in args.splits.split(","):
        d = [v for v in filas.values() if v["split"] == split]
        if not d:
            continue
        tw = sum(v["n_windows"] for v in d)
        tp = sum(v["n_pos"] for v in d)
        tn = sum(v["n_neg"] for v in d)
        pct = 100.0 * tp / tw if tw else 0
        print(f"{split:6s} {len(d):9d} {tw:10d} {tp:9d} {tn:9d} {pct:6.1f}%")

    # --- especies vacias / sin negativas ---
    vacias = [v for v in filas.values() if v["n_windows"] == 0]
    usables = [v for v in filas.values() if v["n_windows"] > 0]
    sin_neg = [v for v in usables if v["n_neg"] == 0]
    con_neg = [v for v in usables if v["n_neg"] > 0]
    if vacias:
        print(f"\nEspecies con 0 ventanas (npz vacio): {len(vacias)}")
        for v in vacias:
            print(f"    {v['split']:5s} {v['species']}")
    print(f"\nEspecies SIN negativas: {len(sin_neg)}  "
          f"(aportan {sum(v['n_pos'] for v in sin_neg)} positivas)")
    print(f"Especies CON negativas: {len(con_neg)}  "
          f"(aportan {sum(v['n_neg'] for v in con_neg)} negativas)")
    if sin_neg:
        print("\n  Sin negativas:")
        for v in sorted(sin_neg, key=lambda x: -x["n_pos"])[:20]:
            print(f"    {v['split']:5s} {v['species'][:52]:52s} pos={v['n_pos']:6d}")

    # --- cuanto aporta cada especie al train 1:1 ---
    tr_todos = [v for v in filas.values() if v["split"] == "train" and v["n_windows"] > 0]
    if tr_todos:
        total_1a1 = sum(v["n_1a1"] for v in tr_todos)
        print(f"\nTrain tras balance 1:1 por especie: {total_1a1} ventanas "
              f"de {sum(v['n_windows'] for v in tr_todos)}")
        print("  Aporte por especie (top 10):")
        for v in sorted(tr_todos, key=lambda x: -x["n_1a1"])[:10]:
            print(f"    {v['species'][:50]:50s} usado={v['n_1a1']:6d} "
                  f"de {v['n_windows']:6d}  ({100*v['n_1a1']/max(v['n_windows'],1):4.0f}%)")

    tr = tr_todos
    if tr:
        print("\n  Train — 8 especies MÁS positivas:")
        for v in sorted(tr, key=lambda x: -x["pct_pos"])[:8]:
            print(f"    {v['species'][:52]:52s} {v['pct_pos']:5.1f}%  "
                  f"pos={v['n_pos']:5d} neg={v['n_neg']:5d}")
        print("\n  Train — 8 especies MÁS balanceadas:")
        for v in sorted(tr, key=lambda x: abs(x["pct_pos"] - 50))[:8]:
            print(f"    {v['species'][:52]:52s} {v['pct_pos']:5.1f}%  "
                  f"pos={v['n_pos']:5d} neg={v['n_neg']:5d}")

    print(f"\nCSV escrito en: {args.out}")


if __name__ == "__main__":
    main()
