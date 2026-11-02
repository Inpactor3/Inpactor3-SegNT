"""Muestra el compromiso precision/recall al mover el umbral.

Sirve para elegir el punto de operacion segun el uso: si no se pueden perder
transposones, se busca recall alto aunque baje la precision.

Uso:
  python barrido_umbral.py --emb_dir trabajo/emb/Animales \
      --head trabajo/ckpt/Animales_split2/best.pt --split test
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import Head, metrics  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--targets", default="0.70,0.80,0.90")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=device)
    head = Head(int(ck.get("hidden", 1280))).to(device)
    head.load_state_dict(ck["head"])
    head.eval()

    ps, ys = [], []
    for f in sorted(glob.glob(os.path.join(args.emb_dir, args.split, "*_emb.npy"))):
        lp = f.replace("_emb.npy", "_labels.npy")
        if not os.path.exists(lp):
            continue
        x = np.load(f).astype(np.float32)
        y = np.load(lp).astype(np.float32)
        if len(x) == 0:
            continue
        with torch.no_grad():
            lg = head(torch.from_numpy(x).to(device)).float().cpu().numpy()
        ps.append(sigmoid(lg))
        ys.append(y)

    p = np.concatenate(ps)
    y = np.concatenate(ys)
    lg = logit(p)

    print(f"{args.split}: {len(y)} ventanas · prevalencia {y.mean():.3f}\n")
    print(f"{'umbral':>7s} {'P':>6s} {'R':>6s} {'F1':>6s} {'BA':>6s}")
    for t in np.arange(0.10, 0.96, 0.05):
        m = metrics(lg, y, float(t))
        print(f"{t:7.2f} {m['precision']:6.3f} {m['recall']:6.3f} "
              f"{m['f1']:6.3f} {m['balanced_accuracy']:6.3f}")

    print("\nUmbral para alcanzar cada recall objetivo:")
    print(f"{'objetivo':>9s} {'umbral':>7s} {'P':>6s} {'F1':>6s} {'BA':>6s}")
    for tgt in [float(x) for x in args.targets.split(",")]:
        mejor = None
        for t in np.arange(0.02, 0.99, 0.01):
            m = metrics(lg, y, float(t))
            if m["recall"] >= tgt:
                mejor = (float(t), m)
        if mejor:
            t, m = mejor
            print(f"{tgt:9.2f} {t:7.2f} {m['precision']:6.3f} {m['f1']:6.3f} "
                  f"{m['balanced_accuracy']:6.3f}")
        else:
            print(f"{tgt:9.2f}   no alcanzable")


if __name__ == "__main__":
    main()
