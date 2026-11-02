"""Negativas dificiles: reentrena dando mas peso a las negativas que el modelo confunde.

Idea: hoy las negativas del entrenamiento son faciles (el modelo las descarta sin
esfuerzo). Si seleccionamos las negativas que el modelo puntua mas alto y les
damos mas peso, el cabezal aprende a separar los casos dificiles y deberia subir
el recall.

Uso:
  python minar_negativas.py --emb_dir trabajo/emb/Animales --out_dir trabajo/hard/Animales \
      --seed 100 --hard_frac 0.2 --hard_weight 3

Compara dos entrenamientos con los mismos datos, la misma semilla y las mismas
epocas: uno con negativas al azar (base) y otro con negativas dificiles.
"""
import argparse
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


def entrenar(xtr, ytr, xva, yva, weights, pos_weight, epochs, patience,
             batch_size, seed, select="balanced_accuracy"):
    torch.manual_seed(seed)
    model = Head(hidden=xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    wt = torch.from_numpy(weights)
    xv = torch.from_numpy(xva).to(DEVICE)
    yv = torch.from_numpy(yva).to(DEVICE)

    mejor, mejor_ep, malas = -1.0, -1, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch_size):
            b = perm[i:i + batch_size]
            xb, yb, wb = xt[b].to(DEVICE), yt[b].to(DEVICE), wt[b].to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            pw = torch.where(yb > 0.5, pos_weight, torch.ones_like(yb))
            loss = (loss_fn(logits, yb) * wb * pw).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(xv).float().cpu().numpy()
        m = metrics(lg, yva, 0.5)
        if m[select] > mejor:
            mejor, mejor_ep, malas = m[select], ep, 0
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            malas += 1
            if malas >= patience:
                break
    model.load_state_dict(mejor_estado)
    model.eval()
    return model, mejor_ep, mejor


def evaluar(model, xte, yte):
    with torch.no_grad():
        lg = model(torch.from_numpy(xte).to(DEVICE)).float().cpu().numpy()
    p = sigmoid(lg)
    m = metrics(lg, yte, 0.5)
    return p, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--hard_frac", type=float, default=0.2,
                    help="fraccion de negativas tratadas como dificiles")
    ap.add_argument("--hard_weight", type=float, default=3.0,
                    help="solo en modo weight")
    ap.add_argument("--mode", choices=("replace", "weight"), default="replace",
                    help="replace: cambia negativas faciles por dificiles sin "
                         "sobreponderar · weight: les da mas peso")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--select", choices=("f1", "balanced_accuracy"),
                    default="balanced_accuracy")
    args = ap.parse_args()

    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    emb = Path(args.emb_dir)
    xtr, ytr = load_split(emb, "train")
    xva, yva = load_split(emb, "val")
    xte, yte = load_split(emb, "test")
    print(f"train {xtr.shape} · val {xva.shape} · test {xte.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    pos = np.flatnonzero(ytr == 1)
    neg = np.flatnonzero(ytr == 0)
    k = min(len(pos), len(neg))
    print(f"pos={len(pos)} neg={len(neg)} -> se usan {k} de cada clase")

    # --- modelo base para puntuar las negativas ---
    print("entrenando modelo base (negativas al azar)...", flush=True)
    pos_sel = rng.choice(pos, min(k, len(pos)), replace=False)
    neg_sel = rng.choice(neg, min(k, len(neg)), replace=False)
    idx_base = np.sort(np.concatenate([pos_sel, neg_sel]))
    w_base = np.ones(len(idx_base), dtype=np.float32)
    modelo_base, ep_b, ba_b = entrenar(
        xtr[idx_base], ytr[idx_base], xva, yva, w_base,
        torch.tensor([float(len(neg_sel)) / max(len(pos_sel), 1)], device=DEVICE),
        args.epochs, args.patience, args.batch_size, args.seed, args.select)
    _, m_base = evaluar(modelo_base, xte, yte)
    print(f"  base: ep {ep_b} · val {ba_b:.4f} · test BA {m_base['balanced_accuracy']:.4f} "
          f"R {m_base['recall']:.4f}", flush=True)

    # --- puntuar negativas y elegir las dificiles ---
    with torch.no_grad():
        p_neg = np.empty(len(neg), dtype=np.float32)
        for i in range(0, len(neg), 20000):
            sl = neg[i:i + 20000]
            lg = modelo_base(torch.from_numpy(xtr[sl]).to(DEVICE)).float().cpu().numpy()
            p_neg[i:i + 20000] = sigmoid(lg)
    n_hard = int(args.hard_frac * len(neg_sel))
    orden = np.argsort(-p_neg)
    hard_all = neg[orden[:n_hard]]
    resto = np.setdiff1d(neg_sel, hard_all, assume_unique=False)
    faltan = max(len(neg_sel) - len(hard_all), 0)
    if len(resto) > faltan:
        resto = rng.choice(resto, faltan, replace=False)
    neg_hard_sel = np.concatenate([hard_all, resto])

    idx_hard = np.sort(np.concatenate([pos_sel, neg_hard_sel]))
    if args.mode == "weight":
        es_hard = np.isin(idx_hard, hard_all)
        w_hard = np.where(es_hard, args.hard_weight, 1.0).astype(np.float32)
        masa_neg = float(w_hard[~np.isin(idx_hard, pos_sel)].sum())
    else:
        w_hard = np.ones(len(idx_hard), dtype=np.float32)
        masa_neg = float(len(neg_hard_sel))
    pw = torch.tensor([masa_neg / max(len(pos_sel), 1)], device=DEVICE)
    print(f"negativas dificiles: {len(hard_all)} de {len(neg_sel)} "
          f"· modo {args.mode} · pos_weight {float(pw):.3f}", flush=True)

    print("entrenando con negativas dificiles...", flush=True)
    modelo_hard, ep_h, ba_h = entrenar(
        xtr[idx_hard], ytr[idx_hard], xva, yva, w_hard, pw,
        args.epochs, args.patience, args.batch_size, args.seed, args.select)
    _, m_hard = evaluar(modelo_hard, xte, yte)
    print(f"  hard: ep {ep_h} · val {ba_h:.4f} · test BA {m_hard['balanced_accuracy']:.4f} "
          f"R {m_hard['recall']:.4f}", flush=True)

    res = {
        "reino": os.path.basename(os.path.normpath(args.emb_dir)),
        "seed": args.seed, "hard_frac": args.hard_frac,
        "hard_weight": args.hard_weight, "mode": args.mode,
        "base": m_base, "hard": m_hard,
        "delta_balanced_accuracy": m_hard["balanced_accuracy"] - m_base["balanced_accuracy"],
        "delta_recall": m_hard["recall"] - m_base["recall"],
        "delta_f1": m_hard["f1"] - m_base["f1"],
    }
    with open(os.path.join(args.out_dir, "hard_neg.json"), "w") as f:
        json.dump(res, f, indent=2)
    torch.save({"head": modelo_hard.state_dict(), "hidden": xtr.shape[1],
                "chunks": 4, "mu": None, "sd": None,
                "threshold": 0.5}, os.path.join(args.out_dir, "best_hard.pt"))

    print("\nComparación en test (umbral 0,5)")
    print(f"{'':16s} {'BA':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} {'Spec':>7s}")
    for nombre, m in (("base (azar)", m_base), ("negativas duras", m_hard)):
        print(f"{nombre:16s} {m['balanced_accuracy']:7.3f} {m['f1']:7.3f} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} {m['specificity']:7.3f}")
    print(f"\ndelta BA {res['delta_balanced_accuracy']:+.4f} · "
          f"delta recall {res['delta_recall']:+.4f} · delta F1 {res['delta_f1']:+.4f}")


if __name__ == "__main__":
    main()
