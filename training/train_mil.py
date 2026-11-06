"""MIL por bloque: puntuar cada bloque y quedarse con el maximo de logits.

Diferencia con train_mlp.py: la cache guarda un embedding por bloque (N,C,H) y el
cabezal se aplica a cada bloque. La ventana se predice con max(logits por bloque),
que es lo que permite detectar elementos pequenos enterrados en un solo bloque.

Uso:
  python train_mil.py --emb_dir /emb/HongosMIL --out_dir /emb/Modelos/HongosMIL \
      --epochs 15 --class_weight --balance global
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA = 1e-4


def metrics(logits, labels, thr=0.5):
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    out = {"precision": prec, "recall": rec, "specificity": spec, "f1": f1,
           "accuracy": acc, "balanced_accuracy": (rec + spec) / 2,
           "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(labels)),
           "prevalence": float(labels.mean()) if len(labels) else 0.0}
    order = np.argsort(-p)
    y = labels[order]
    tpc = np.cumsum(y)
    fpc = np.cumsum(1 - y)
    pc = tpc / np.maximum(tpc + fpc, 1)
    rc = tpc / max(int(labels.sum()), 1)
    out["pr_auc"] = float(np.sum(np.diff(np.concatenate([[0], rc])) * pc))
    return out


def calibrate_threshold(logits, labels):
    best = (-1.0, 0.5)
    for t in np.unique(np.round(np.arange(0.02, 0.99, 0.01), 4)):
        m = metrics(logits, labels, t)
        if m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], float(t))
    t = best[1]
    return round(t, 3)


def cargar(d, split):
    files = sorted((d / split).glob("*_emb.npy"))
    if not files:
        raise FileNotFoundError(f"sin embeddings en {d}/{split}")
    embs, labs = [], []
    for f in files:
        e = np.load(f)
        if e.ndim == 3 and e.shape[0] == 0:
            continue
        embs.append(e)
        lab = np.load(f.with_name(f.name.replace("_emb.npy", "_labels.npy")))
        labs.append(lab.astype(np.float32))
    x = np.concatenate(embs).astype(np.float16)
    y = np.concatenate(labs).astype(np.float32)
    print(f"{split}: {len(files)} archivos · {x.shape[0]} ventanas · "
          f"{x.shape[1]} bloques · {x.shape[2]} dims", flush=True)
    return x, y


class MILHead(nn.Module):
    def __init__(self, hidden=512, dropout=0.2):
        super().__init__()
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(512, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        B, C, H = x.shape
        f = self.dropout(torch.relu(self.pool_proj(x.reshape(B * C, H))))
        lg = self.cls(f).view(B, C)
        return lg.max(dim=1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--balance", choices=("none", "global"), default="global")
    ap.add_argument("--select", default="balanced_accuracy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xtr_all, ytr_all = cargar(args.emb_dir, "train")
    xva, yva = cargar(args.emb_dir, "val")
    xte, yte = cargar(args.emb_dir, "test")

    if args.balance == "global":
        pos = np.flatnonzero(ytr_all == 1)
        neg = np.flatnonzero(ytr_all == 0)
        k = min(len(pos), len(neg))
        idx = np.sort(np.concatenate([rng.choice(pos, k, replace=False),
                                      rng.choice(neg, k, replace=False)]))
        xtr_all, ytr_all = xtr_all[idx], ytr_all[idx]
        print(f"balance 1:1 -> {len(idx)} ventanas", flush=True)

    muestra = xtr_all.reshape(-1, xtr_all.shape[-1])
    if len(muestra) > 200_000:
        sel = rng.choice(len(muestra), 200_000, replace=False)
        muestra = muestra[sel]
    mu = muestra.mean(axis=0).astype(np.float32)
    sd = muestra.std(axis=0).astype(np.float32) + SIGMA

    model = MILHead(xtr_all.shape[-1], args.dropout).to(device)
    pos_weight = None
    if args.class_weight:
        neg = int((ytr_all == 0).sum())
        pos = int((ytr_all == 1).sum())
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        print(f"class_weight pos_weight={float(pos_weight):.4f}", flush=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)

    def a_device(x):
        return (torch.from_numpy(x).to(device).float() - mu_t) / sd_t

    yt = torch.from_numpy(ytr_all).to(device)
    xv = a_device(xva)
    xte_t = a_device(xte)

    def predecir(x):
        salida = []
        for i in range(0, len(x), 8192):
            with torch.no_grad():
                salida.append(model(x[i:i + 8192]).float().cpu().numpy())
        return np.concatenate(salida)

    mejor, mejor_ep = -1.0, -1
    log = open(args.out_dir / "metrics.csv", "w")
    log.write("epoch,loss,val_loss,val_precision,val_recall,val_f1,val_acc,val_bal_acc\n")

    for ep in range(1, args.epochs + 1):
        model.train()
        orden = torch.randperm(len(xtr_all))
        tot, n = 0.0, 0
        for i in range(0, len(orden), args.batch_size):
            b = orden[i:i + args.batch_size]
            xb = a_device(xtr_all[b.numpy()])
            yb = yt[b]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)

        model.eval()
        lgv = predecir(xv)
        vl = F.binary_cross_entropy_with_logits(
            torch.from_numpy(lgv), torch.from_numpy(yva)).item()
        m = metrics(lgv, yva)
        log.write(f"{ep},{tot/n:.4f},{vl:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                  f"{m['f1']:.4f},{m['accuracy']:.4f},{m['balanced_accuracy']:.4f}\n")
        log.flush()
        print(f"ep {ep:3d} loss={tot/n:.4f} val_f1={m['f1']:.4f} "
              f"val_bal_acc={m['balanced_accuracy']:.4f} rec={m['recall']:.4f}",
              flush=True)

        if m[args.select] > mejor:
            mejor, mejor_ep = m[args.select], ep
            torch.save({"head": model.state_dict(), "epoch": ep, "hidden": int(xtr_all.shape[-1]),
                        "mu": mu, "sd": sd, "val_f1": m["f1"],
                        "val_selected": mejor, "select": args.select,
                        "mode": "mil_max", "blocks": int(xtr_all.shape[1])},
                       args.out_dir / "best.pt")

    ck = torch.load(args.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["head"])
    model.eval()
    lgv = predecir(xv)
    lgte = predecir(xte_t)
    thr = calibrate_threshold(lgv, yva)
    ck["threshold"] = thr
    torch.save(ck, args.out_dir / "best.pt")

    res = {"threshold": thr, "epoch": ck["epoch"], "blocks": ck["blocks"],
           "mode": "mil_max", "balance": args.balance, "class_weight": args.class_weight,
           "at_calibrated_threshold": metrics(lgte, yte, thr),
           "at_050": metrics(lgte, yte, 0.5),
           "always_positive": metrics(lgte, np.ones_like(lgte), 0.5),
           "best_f1_threshold": float(max(np.arange(0.05, 0.96, 0.05),
                                          key=lambda t: metrics(lgte, yte, t)["f1"]))}
    with open(args.out_dir / "test_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"threshold": thr, "epoch": ck["epoch"],
                      **{k: round(v, 4) for k, v in
                         res["at_calibrated_threshold"].items()
                         if isinstance(v, float)}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
