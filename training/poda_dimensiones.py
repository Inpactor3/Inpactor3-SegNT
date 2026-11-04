"""Busca dimensiones del embedding cuyo apagado MEJORA el acierto.

Selecciona en validacion (nunca en test) y luego mide en test. Es una version
simple de "editar los pesos": no se reentrena nada, solo se dejan de usar las
dimensiones que perjudican.

Uso:
  python poda_dimensiones.py --emb_dir trabajo/emb/Animales \
      --head trabajo/ckpt/Animales_split2/best.pt --k 5,10,20,40
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import Head, load_split, metrics  # noqa: E402


def elegir_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        libre, _ = torch.cuda.mem_get_info()
        return "cuda" if libre > 1.5e9 else "cpu"
    except Exception:
        return "cpu"


def preparar(split, emb_dir, head, ck, dev):
    x, y = load_split(Path(emb_dir), split)
    if ck.get("mu") is not None:
        x = ((x - np.asarray(ck["mu"], dtype=np.float32)) /
             np.asarray(ck["sd"], dtype=np.float32)).astype(np.float32)
    xt = torch.from_numpy(x).to(dev)
    with torch.no_grad():
        pre = head.pool_proj(xt)
    return xt, y, pre


def metricas_con(xt, pre, W1, y, head, dims=None):
    h = pre if not dims else pre - sum(
        xt[:, d:d + 1] * W1[:, d] for d in dims)
    with torch.no_grad():
        lg = head.cls_head(torch.relu(h)).squeeze(-1)
    return metrics(lg.float().cpu().numpy(), y, 0.5)


def ba_con(xt, pre, W1, y, head, dims=None):
    return metricas_con(xt, pre, W1, y, head, dims)["balanced_accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--k", default="5,10,20,40")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = elegir_device()
    ck = torch.load(args.head, map_location="cpu")
    head = Head(int(ck.get("hidden", 1280))).to(dev)
    head.load_state_dict(ck["head"])
    head.eval()
    W1 = head.pool_proj.weight.detach()
    print("device:", dev, flush=True)

    xv, yv, pv = preparar("val", args.emb_dir, head, ck, dev)
    xt, yt, pt = preparar("test", args.emb_dir, head, ck, dev)
    ba_v0 = ba_con(xv, pv, W1, yv, head)
    ba_t0 = ba_con(xt, pt, W1, yt, head)
    print(f"base: val BA {ba_v0:.4f} · test BA {ba_t0:.4f}", flush=True)

    # cuanto mejora la validacion al apagar cada dimension (se cachea)
    cache = (args.out or "poda") + ".mejora.npy"
    if os.path.exists(cache):
        mejora = np.load(cache)
        print("mejoras cargadas de", cache, flush=True)
    else:
        mejora = np.zeros(W1.shape[1], dtype=np.float32)
        with torch.no_grad():
            for d in range(W1.shape[1]):
                mejora[d] = ba_con(xv, pv, W1, yv, head, [d]) - ba_v0
                if d % 200 == 0:
                    print(f"  {d}/{W1.shape[1]}", flush=True)
        np.save(cache, mejora)
    orden = np.argsort(-mejora)

    res = {"base_val": ba_v0, "base_test": ba_t0, "k": {}}
    mv0 = metricas_con(xv, pv, W1, yv, head)
    mt0 = metricas_con(xt, pt, W1, yt, head)
    print(f"\nbase test: BA {mt0['balanced_accuracy']:.4f} · PR-AUC {mt0['pr_auc']:.4f}")
    print(f"{'k':>4s} {'val BA':>8s} {'test BA':>8s} {'d BA':>8s} "
          f"{'test PR-AUC':>12s} {'d PR-AUC':>9s} {'R':>6s} {'Spec':>6s}")
    for k in [int(x) for x in args.k.split(",")]:
        dims = orden[:k].tolist()
        mv = metricas_con(xv, pv, W1, yv, head, dims)
        mt = metricas_con(xt, pt, W1, yt, head, dims)
        res["k"][str(k)] = {
            "dims": dims,
            "val": mv["balanced_accuracy"], "test": mt["balanced_accuracy"],
            "d_ba": mt["balanced_accuracy"] - mt0["balanced_accuracy"],
            "test_pr_auc": mt["pr_auc"],
            "d_pr_auc": mt["pr_auc"] - mt0["pr_auc"],
            "recall": mt["recall"], "specificity": mt["specificity"],
        }
        print(f"{k:4d} {mv['balanced_accuracy']:8.4f} {mt['balanced_accuracy']:8.4f} "
              f"{mt['balanced_accuracy']-mt0['balanced_accuracy']:+8.4f} "
              f"{mt['pr_auc']:12.4f} {mt['pr_auc']-mt0['pr_auc']:+9.4f} "
              f"{mt['recall']:6.3f} {mt['specificity']:6.3f}")

    print(f"\nmejoran la validacion al apagarlas: "
          f"{int((mejora > 0).sum())} de {len(mejora)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("guardado en", args.out)


if __name__ == "__main__":
    main()
