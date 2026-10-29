"""Reasigna las especies de un reino a train/val/test estratificando por grupo.

Motivo: el split actual se hizo por especie pero sin cuidar la taxonomía. En
animales quedó 74 % del entrenamiento en vertebrados y 0 % de vertebrados en
validación y prueba. Este script genera un reparto nuevo que mantiene la
disjunción por especie y garantiza representación de cada grupo.

No mueve nada por sí solo: escribe
  - nuevo_split_<reino>.csv      (especie, grupo, split actual, split nuevo)
  - aplicar_split_<reino>.sh     (script con los mv/rm/reentrenos, para revisar)

Uso:
  python rehacer_split_animales.py --tax_csv docs/animales_taxonomia.csv \
      --reino Animales --seed 0
"""
import argparse
import collections
import csv
import random

VERTEBRADOS = {"Mamíferos", "Aves", "Reptiles", "Peces"}


def plan_grupo(n, rng):
    """Cuantas especies van a val/test segun cuantas haya en el grupo."""
    if n >= 6:
        n_test = max(1, round(0.18 * n))
        n_val = max(1, round(0.18 * n))
    elif n >= 4:
        n_test, n_val = 1, 1
    elif n == 3:
        n_test, n_val = 1, 0
    else:
        n_test, n_val = 0, 0
    while n - n_test - n_val < 2:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tax_csv", required=True)
    ap.add_argument("--reino", default="Animales")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_windows", type=int, default=8000,
                    help="tamano ideal de las especies que van a val/test (abarata "
                         "el recalculo de embeddings)")
    ap.add_argument("--bundle", default="datos/bundle4_Animales/Animales")
    ap.add_argument("--emb", default="trabajo/emb/Animales")
    ap.add_argument("--sub", default="trabajo/sub/Animales")
    args = ap.parse_args()

    filas = [r for r in csv.DictReader(open(args.tax_csv))]
    filas = [r for r in filas if int(r.get("n_windows", 0)) > 0]

    por_grupo = collections.defaultdict(list)
    for r in filas:
        por_grupo[r["grupo"]].append(r)

    rng = random.Random(args.seed)
    for grupo, sp in sorted(por_grupo.items()):
        # se eligen las especies mas cercanas al tamano objetivo, para no mandar
        # a val/test especies de 50.000 ventanas (encarece el recalculo)
        sp.sort(key=lambda r: abs(int(r["n_windows"]) - args.target_windows))
        n_val, n_test = plan_grupo(len(sp), rng)
        for i, r in enumerate(sp):
            if i < n_test + n_val:
                # se alternan val y test, respetando los cupos
                quiere_test = (i % 2 == 0)
                if quiere_test and sum(1 for x in sp[:i] if x.get("split_nuevo") == "test") >= n_test:
                    quiere_test = False
                if not quiere_test and sum(1 for x in sp[:i] if x.get("split_nuevo") == "val") >= n_val:
                    quiere_test = True
                r["split_nuevo"] = "test" if quiere_test else "val"
            else:
                r["split_nuevo"] = "train"

    # --- informe ---
    def resumen(split):
        d = [r for r in filas if r["split_nuevo"] == split]
        w = sum(int(r["n_windows"]) for r in d)
        wv = sum(int(r["n_windows"]) for r in d if r["grupo"] in VERTEBRADOS)
        pct = 100.0 * wv / w if w else 0
        grupos = collections.Counter(r["grupo"] for r in d)
        return d, w, pct, grupos

    print(f"Reparto propuesto para {args.reino} (semilla {args.seed})\n")
    for split in ("train", "val", "test"):
        d, w, pct, grupos = resumen(split)
        print(f"== {split.upper()}: {len(d)} especies · {w:,} ventanas · "
              f"{pct:.1f} % vertebrados")
        for g, n in grupos.most_common():
            print(f"     {g:26s} {n}")

    movidas = [r for r in filas if r["split"] != r["split_nuevo"]]
    print(f"\nEspecies que cambian de split: {len(movidas)}")
    for r in sorted(movidas, key=lambda r: (r["split"], r["split_nuevo"])):
        print(f"   {r['split']:5s} -> {r['split_nuevo']:5s}  "
              f"{r['nombre_cientifico'][:34]:34s} {r['grupo']:22s} "
              f"{int(r['n_windows']):6,}")

    csv_out = f"nuevo_split_{args.reino.lower()}.csv"
    with open(csv_out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["species", "nombre_cientifico", "grupo",
                                           "split", "split_nuevo", "n_windows",
                                           "pct_pos"])
        w.writeheader()
        for r in sorted(filas, key=lambda r: (r["split_nuevo"], r["grupo"])):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # --- script para aplicar ---
    sh = f"aplicar_split_{args.reino.lower()}.sh"
    with open(sh, "w") as fo:
        fo.write("#!/bin/bash\n# Generado por rehacer_split_animales.py — REVISAR ANTES DE EJECUTAR\n")
        fo.write("set -e\ncd \"$HOME/Escritorio/Inpactor3\"\n")
        fo.write(". .venv/bin/activate\n\n")
        fo.write(f'B={args.bundle}\nE={args.emb}\nS={args.sub}\n\n')
        fo.write('echo "== 1/5 moviendo npz ==="\n')
        for r in movidas:
            sp = r["species"]
            fo.write(f'mv "$B/{r["split"]}/{sp}.npz" "$B/{r["split_nuevo"]}/"\n')
        fo.write('\necho "== 2/5 regenerando train submuestreado (semilla por especie) ==="\n')
        fo.write('rm -rf "$S"\n')
        fo.write('python code/subsample_train.py --seq_dir "$B" --out_dir "$S" --ratio 1\n')
        fo.write('\necho "== 3/5 borrando embeddings de las especies movidas ==="\n')
        for r in movidas:
            sp = r["species"]
            fo.write(f'rm -f "$E"/*/{sp}_*.npy\n')
        fo.write('\necho "== 4/5 recalculando embeddings (salta los que existen) ==="\n')
        fo.write('python code/precompute_embeddings.py --seq_dir "$S" --out_dir "$E" '
                 '--splits val,test --batch_size 32 --token_batch 128\n')
        fo.write('python code/precompute_embeddings.py --seq_dir "$S" --out_dir "$E" '
                 '--splits train --batch_size 32 --token_batch 128\n')
        fo.write('\necho "== 5/5 reentrenando ==="\n')
        fo.write('python code/train_mlp.py --emb_dir "$E" --out_dir trabajo/ckpt/Animales_split2 '
                 '--epochs 100 --patience 10 --batch_size 4096 --balance global '
                 '--class_weight --calibrate --select balanced_accuracy\n')
        fo.write('\necho "== LISTO =="\n')

    print(f"\nCSV: {csv_out}")
    print(f"Script a ejecutar en el servidor: {sh}")
    print("Revísalo con:  less " + sh)


if __name__ == "__main__":
    main()
