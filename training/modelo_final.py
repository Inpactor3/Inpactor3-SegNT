"""Modelo final: negativas dificiles + ensamble de semillas + umbral elegido en validacion.

Flujo:
  1. Entrena un modelo "explorador" con negativas al azar.
  2. Puntua todas las negativas de train y selecciona las mas dificiles.
  3. Reentrena K semillas sobre ese conjunto (reemplazando negativas faciles).
  4. Promedia las probabilidades (ensamble).
  5. Elige el umbral en VALIDACION (max balanced accuracy) y reporta test con el.

Uso:
  python modelo_final.py --emb_dir trabajo/emb/Animales \
      --out_dir trabajo/final/Animales --seeds 3
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mlp import Head, load_split, metrics  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def entrenar(xtr, ytr, xva, yva, epochs, patience, batch_size, seed,
             select="balanced_accuracy", device="cpu"):
    torch.manual_seed(seed)
    model = Head(hidden=xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(xva).to(device)
    yv = torch.from_numpy(yva).to(device)

    mejor, mejor_ep, malas, estado = -1.0, -1, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            m = metrics(model(xv).float().cpu().numpy(), yva, 0.5)
        if m[select] > mejor:
            mejor, mejor_ep, malas = m[select], ep, 0
            estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            malas += 1
            if malas >= patience:
                break
    model.load_state_dict(estado)
    model.eval()
    return model, mejor_ep, mejor


def predecir(model, x, device):
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 20000):
            lg = model(torch.from_numpy(x[i:i + 20000]).to(device)).float().cpu().numpy()
            out[i:i + 20000] = sigmoid(lg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hard_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--select", choices=("f1", "balanced_accuracy"),
                    default="balanced_accuracy")
    ap.add_argument("--target_recall", type=float, default=None,
                    help="si se da, el umbral se elige para alcanzar ese recall en val")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    emb = Path(args.emb_dir)
    xtr, ytr = load_split(emb, "train")
    xva, yva = load_split(emb, "val")
    xte, yte = load_split(emb, "test")
    print(f"train {xtr.shape} · val {xva.shape} · test {xte.shape} · device {device}",
          flush=True)

    rng = np.random.default_rng(0)
    pos = np.flatnonzero(ytr == 1)
    neg = np.flatnonzero(ytr == 0)
    k = min(len(pos), len(neg))
    print(f"pos={len(pos)} neg={len(neg)} -> {k} de cada clase", flush=True)

    # 1) explorador con negativas al azar
    pos_sel = rng.choice(pos, k, replace=False)
    neg_rand = rng.choice(neg, k, replace=False)
    base_idx = np.sort(np.concatenate([pos_sel, neg_rand]))
    print("entrenando explorador...", flush=True)
    expl, ep_e, ba_e = entrenar(xtr[base_idx], ytr[base_idx], xva, yva,
                                args.epochs, args.patience, args.batch_size, 0,
                                args.select, device)
    print(f"  explorador: ep {ep_e} val {ba_e:.4f}", flush=True)

    # 2) negativas dificiles
    p_neg = predecir(expl, xtr[neg], device)
    n_hard = int(args.hard_frac * k)
    hard_all = neg[np.argsort(-p_neg)[:n_hard]]
    resto_pool = np.setdiff1d(neg, hard_all, assume_unique=False)
    faltan = k - n_hard
    resto = rng.choice(resto_pool, min(faltan, len(resto_pool)), replace=False)
    neg_hard = np.concatenate([hard_all, resto])
    idx = np.sort(np.concatenate([pos_sel, neg_hard]))
    print(f"negativas dificiles: {len(hard_all)} de {k}", flush=True)

    # 3) ensamble sobre ese conjunto
    pv_seeds, pt_seeds = [], []
    for s in range(args.seeds):
        seed = 100 + s
        print(f"entrenando semilla {seed}...", flush=True)
        m, ep, ba = entrenar(xtr[idx], ytr[idx], xva, yva, args.epochs,
                             args.patience, args.batch_size, seed,
                             args.select, device)
        print(f"  ep {ep} val {ba:.4f}", flush=True)
        pv_seeds.append(predecir(m, xva, device))
        pt_seeds.append(predecir(m, xte, device))
        torch.save({"head": m.state_dict(), "hidden": xtr.shape[1], "mu": None,
                    "sd": None, "chunks": 4, "threshold": 0.5},
                   os.path.join(args.out_dir, f"seed{seed}.pt"))

    pv = np.mean(pv_seeds, axis=0)
    pt = np.mean(pt_seeds, axis=0)

    # 4) umbral en validacion
    if args.target_recall is not None:
        thr = None
        for t in np.arange(0.02, 0.99, 0.01):
            if metrics(logit(pv), yva, float(t))["recall"] >= args.target_recall:
                thr = float(t)
        thr = thr if thr is not None else 0.5
        criterio = f"recall>={args.target_recall}"
    else:
        mejor = (-1.0, 0.5)
        for t in np.arange(0.02, 0.99, 0.01):
            m = metrics(logit(pv), yva, float(t))
            if m["balanced_accuracy"] > mejor[0]:
                mejor = (m["balanced_accuracy"], float(t))
        thr = mejor[1]
        criterio = "max balanced accuracy"

    mval = metrics(logit(pv), yva, thr)
    mtest = metrics(logit(pt), yte, thr)
    mtest50 = metrics(logit(pt), yte, 0.5)

    print(f"\nUmbral elegido en validacion: {thr:.2f} ({criterio})")
    print(f"{'':16s} {'BA':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} {'Spec':>7s} {'PR-AUC':>7s}")
    for nombre, m in (("validacion", mval), ("test", mtest), ("test @0.5", mtest50)):
        print(f"{nombre:16s} {m['balanced_accuracy']:7.3f} {m['f1']:7.3f} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} {m['specificity']:7.3f} "
              f"{m['pr_auc']:7.3f}")

    res = {"reino": os.path.basename(os.path.normpath(args.emb_dir)),
           "seeds": args.seeds, "hard_frac": args.hard_frac,
           "threshold": thr, "criterio": criterio,
           "val": mval, "test": mtest, "test_at_050": mtest50}
    with open(os.path.join(args.out_dir, "final.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nGuardado en {os.path.join(args.out_dir, 'final.json')}")


if __name__ == "__main__":
    main()
