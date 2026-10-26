"""Reduce el split de train al tamano que realmente usa el entrenador 1:1.

El MLP se entrena con `--balance global` (1:1), asi que calcular embeddings para
los positivos sobrantes es tirar GPU: en Plantas era el 92% del trabajo.

Este script construye un directorio nuevo con:
  train/  -> npz submuestreados: TODAS las negativas + hasta <ratio> x negativas
             de positivas por especie (o <= orphan_cap si la especie no tiene negativas)
  val/    -> enlaces simbolicos a los npz originales (SIN tocar)
  test/   -> idem

Uso:
  python subsample_train.py --seq_dir <bundle4>/<Reino> --out_dir /content/sub/Plantas \
      --ratio 1
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True,
                    help="directorio con train/ val/ test/ de npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--orphan_cap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits_to_link", default="val,test")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dst = os.path.join(args.out_dir, "train")
    os.makedirs(dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.seq_dir, "train", "*.npz")))
    if not files:
        raise SystemExit(f"sin npz en {args.seq_dir}/train")

    tot_p = tot_n = 0
    for f in files:
        out = os.path.join(dst, os.path.basename(f))
        if os.path.exists(out):
            z = np.load(out, allow_pickle=True)
            y = z["label"]
            tot_p += int((y == 1).sum())
            tot_n += int((y == 0).sum())
            print("SKIP", os.path.basename(f), flush=True)
            continue
        z = np.load(f, allow_pickle=True)
        y = z["label"]
        neg = np.flatnonzero(y == 0)
        pos = np.flatnonzero(y == 1)
        if len(neg) == 0:
            keep = pos[: args.orphan_cap]
        else:
            k = min(len(pos), int(round(args.ratio * len(neg))))
            keep = np.concatenate([neg, rng.choice(pos, k, replace=False)])
        keep.sort()
        np.savez_compressed(out, seq=z["seq"][keep], label=y[keep],
                            window_id=z["window_id"][keep],
                            offsets=z["offsets"], chunk_bp=z["chunk_bp"])
        tot_p += int((y[keep] == 1).sum())
        tot_n += int((y[keep] == 0).sum())
        print(f"OK {os.path.basename(f)}: {len(keep)}/{len(y)}", flush=True)

    for split in args.splits_to_link.split(","):
        d = os.path.join(args.out_dir, split)
        os.makedirs(d, exist_ok=True)
        n = 0
        for f in glob.glob(os.path.join(args.seq_dir, split, "*.npz")):
            link = os.path.join(d, os.path.basename(f))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(f), link)
            n += 1
        print(f"{split}: {n} npz enlazados (intactos)", flush=True)

    print(f"TOTAL train submuestreado: {tot_p} pos + {tot_n} neg = {tot_p + tot_n}")


if __name__ == "__main__":
    main()
