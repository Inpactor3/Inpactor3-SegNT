"""Compara variantes de cabezal sobre los embeddings ya calculados (sin tocar el backbone).

Cada variante se entrena con el mismo procedimiento (balance 1:1, early stopping por
balanced accuracy en validacion) y se evalua en test con umbral 0,5. Se repite con
varias semillas para no confundir capacidad con ruido.

Uso:
  python probar_cabezales.py --emb_dir trabajo/emb/Animales --out_dir trabajo/cabezales
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
from train_mlp import load_split, metrics, balance_indices  # noqa: E402

VARIANTES = {
    "base":   {"dims": [None, 512, 256, 1], "dropout": 0.2},
    "wide":   {"dims": [None, 1024, 512, 1], "dropout": 0.2},
    "deep":   {"dims": [None, 512, 256, 128, 1], "dropout": 0.2},
    "linear": {"dims": [None, 1], "dropout": 0.0},
    "ancho_profundo": {"dims": [None, 1024, 512, 256, 1], "dropout": 0.3},
}


class MLP(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        capas = []
        for i in range(len(dims) - 1):
            capas.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                capas += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*capas)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def entrenar(xtr, ytr, xva, yva, dims, dropout, epochs, patience, bs, seed, device):
    torch.manual_seed(seed)
    m = MLP(dims, dropout).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    L = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(xva).to(device), torch.from_numpy(yva).to(device)
    best, bep, bad, est = -1.0, -1, 0, None
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = L(m(xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            mm = metrics(m(xv).float().cpu().numpy(), yva, 0.5)
        if mm["balanced_accuracy"] > best:
            best, bep, bad = mm["balanced_accuracy"], ep, 0
            est = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(est)
    m.eval()
    return m, bep, best


def evaluar(m, x, y, device, bs=8192):
    out = np.empty(len(x), dtype=np.float32)
    m.eval()
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out[i:i + bs] = m(torch.from_numpy(x[i:i + bs]).to(device)).float().cpu().numpy()
    return metrics(out, y, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--variants", default="base,wide,deep,linear,ancho_profundo")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = Path(args.emb_dir)
    xtr, ytr = load_split(emb, "train")
    xva, yva = load_split(emb, "val")
    xte, yte = load_split(emb, "test")
    print(f"train {xtr.shape} · val {xva.shape} · test {xte.shape}", flush=True)

    res = {}
    for nombre in args.variants.split(","):
        cfg = VARIANTES[nombre]
        dims = [xtr.shape[1]] + cfg["dims"][1:]
        print(f"\n== {nombre}: {dims} dropout {cfg['dropout']}", flush=True)
        bas, f1s, prs = [], [], []
        for s in range(args.seeds):
            seed = 500 + s
            idx = balance_indices(ytr, "global", seed)
            m, ep, vba = entrenar(xtr[idx], ytr[idx], xva, yva, dims,
                                  cfg["dropout"], args.epochs, args.patience,
                                  args.batch_size, seed, device)
            mt = evaluar(m, xte, yte, device)
            bas.append(mt["balanced_accuracy"]); f1s.append(mt["f1"]); prs.append(mt["pr_auc"])
            print(f"   semilla {seed}: ep {ep} test BA {mt['balanced_accuracy']:.4f} "
                  f"F1 {mt['f1']:.4f} PR-AUC {mt['pr_auc']:.4f}", flush=True)
        res[nombre] = {
            "dims": dims, "dropout": cfg["dropout"],
            "ba_media": float(np.mean(bas)), "ba_std": float(np.std(bas)),
            "f1_media": float(np.mean(f1s)), "pr_auc_media": float(np.mean(prs)),
            "pr_auc_std": float(np.std(prs)),
        }

    print("\nResumen (media de", args.seeds, "semillas)")
    print(f"{'variante':18s} {'BA':>7s} {'±':>6s} {'F1':>7s} {'PR-AUC':>8s} {'±':>6s}")
    for n, r in res.items():
        print(f"{n:18s} {r['ba_media']:7.3f} {r['ba_std']:6.3f} {r['f1_media']:7.3f} "
              f"{r['pr_auc_media']:8.3f} {r['pr_auc_std']:6.3f}")
    with open(os.path.join(args.out_dir, "cabezales.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nGuardado en {os.path.join(args.out_dir, 'cabezales.json')}")


if __name__ == "__main__":
    main()
