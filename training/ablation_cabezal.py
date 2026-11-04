"""Ablacion del cabezal: apaga cada dimension del embedding y mide cuanto cae el acierto.

Sirve para saber si la decision esta concentrada en unas pocas dimensiones (defecto
localizado, lo que una edicion de pesos podria arreglar) o repartida por igual
(no hay nada que arreglar).

Uso:
  python ablation_cabezal.py --emb_dir trabajo/emb/Animales \
      --head trabajo/ckpt/Animales_split2/best.pt --split test
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import Head, load_split, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    # cargar siempre en CPU: el checkpoint es diminuto y asi no falla si la GPU
    # esta ocupada por otro proceso
    ck = torch.load(args.head, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = int(ck.get("hidden", 1280))
    head = Head(hidden).to(dev)
    head.load_state_dict(ck["head"])
    head.eval()

    x, y = load_split(args.emb_dir, args.split)
    if ck.get("mu") is not None:
        mu = np.asarray(ck["mu"], dtype=np.float32)
        sd = np.asarray(ck["sd"], dtype=np.float32)
        x = ((x - mu) / sd).astype(np.float32)
    try:
        xt = torch.from_numpy(x).to(dev)
        yt = torch.from_numpy(y).to(dev)
    except RuntimeError as e:
        print("GPU sin memoria, se usa CPU:", str(e)[:80], flush=True)
        dev = "cpu"
        head = head.cpu()
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
    print(f"{args.split}: {tuple(xt.shape)} · device {dev}", flush=True)

    with torch.no_grad():
        pre = head.pool_proj(xt)                  # (N, 512)
        base_lg = head.cls_head(torch.relu(pre)).squeeze(-1)
        base = metrics(base_lg.float().cpu().numpy(), y, 0.5)
    print(f"base: BA {base['balanced_accuracy']:.4f} · F1 {base['f1']:.4f}", flush=True)

    W1 = head.pool_proj.weight.detach()           # (512, hidden)
    ba_base = base["balanced_accuracy"]
    drops = np.zeros(hidden, dtype=np.float32)
    with torch.no_grad():
        for d in range(hidden):
            h = torch.relu(pre - xt[:, d:d + 1] * W1[:, d])
            lg = head.cls_head(h).squeeze(-1)
            ba = metrics(lg.float().cpu().numpy(), y, 0.5)["balanced_accuracy"]
            drops[d] = ba_base - ba
            if d % 200 == 0:
                print(f"  {d}/{hidden}", flush=True)

    orden = np.argsort(-drops)
    top = orden[:args.top]

    # control: apagar las top-k juntas vs k al azar
    def con_dim_apagadas(dims):
        h = pre.clone()
        for d in dims:
            h = h - xt[:, d:d + 1] * W1[:, d]
        with torch.no_grad():
            lg = head.cls_head(torch.relu(h)).squeeze(-1)
        return metrics(lg.float().cpu().numpy(), y, 0.5)["balanced_accuracy"]

    ba_top = con_dim_apagadas(top)
    rng = np.random.default_rng(0)
    controles = []
    for _ in range(20):
        dims = rng.choice(hidden, args.top, replace=False)
        controles.append(con_dim_apagadas(dims))
    ba_ctrl = float(np.mean(controles))

    print(f"\nCaida al apagar 1 dimension: mediana {np.median(drops):.5f} "
          f"· maximo {drops.max():.5f} · minimo {drops.min():.5f}")
    print(f"Caida al apagar las {args.top} mejores juntas: {ba_base - ba_top:.4f}")
    print(f"Caida al apagar {args.top} al azar:             {ba_base - ba_ctrl:.4f}")
    print(f"Top {args.top} dimensiones: {top.tolist()[:10]}...")

    res = {"split": args.split, "hidden": hidden,
           "base_ba": ba_base, "base_f1": base["f1"],
           "drop_mediana": float(np.median(drops)),
           "drop_maximo": float(drops.max()),
           "drop_minimo": float(drops.min()),
           "top_dim": top.tolist(),
           "drop_top_dim": float(drops[top[0]]),
           "caida_top_k": float(ba_base - ba_top),
           "caida_k_azar": float(ba_base - ba_ctrl),
           "ratio_top_vs_azar": float((ba_base - ba_top) / max(ba_base - ba_ctrl, 1e-6))}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("guardado en", args.out)


if __name__ == "__main__":
    main()
