"""Entrena solo el cabezal MLP sobre embeddings precalculados (backbone congelado).

Uso:
  python train_mlp.py --emb_dir <out> --out_dir <ckpt> --epochs 100

Requiere los archivos generados por precompute_embeddings.py.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, hidden=1280, dropout=0.2):
        super().__init__()
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.dropout(torch.relu(self.pool_proj(x)))
        return self.cls_head(f).squeeze(-1)


def metrics(logits, labels, thr=0.5):
    p = 1 / (1 + np.exp(-logits))
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def load_split(d, split):
    emb = np.load(d / f"{split}_emb.npy").astype(np.float32)
    lab = np.load(d / f"{split}_labels.npy").astype(np.float32)
    return emb, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--class_weight", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr = load_split(args.emb_dir, "train")
    xva, yva = load_split(args.emb_dir, "val")
    print("train", xtr.shape, "val", xva.shape, "pos frac", ytr.mean().round(4))

    model = Head(hidden=xtr.shape[1], dropout=args.dropout).to(device)
    pos_weight = None
    if args.class_weight:
        pos = float(ytr.sum()); neg = float(len(ytr) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xt = torch.from_numpy(xtr); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(xva).to(device); yv = torch.from_numpy(yva).to(device)
    print("val on", device)

    best_f1, best_ep, bad = -1.0, -1, 0
    log = open(args.out_dir / "metrics.csv", "w")
    log.write("epoch,loss,val_loss,val_precision,val_recall,val_f1,val_acc\n")

    for ep in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(len(xt))
        tot, n = 0.0, 0
        for i in range(0, len(xt), args.batch_size):
            b = idx[i:i + args.batch_size]
            xb = xt[b].to(device); yb = yt[b].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(b); n += len(b)

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xv), yv).item()
            logits = model(xv).float().cpu().numpy()
        m = metrics(logits, yva)
        log.write(f"{ep},{tot/n:.4f},{vl:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                  f"{m['f1']:.4f},{m['accuracy']:.4f}\n")
        log.flush()
        print(f"ep {ep:3d} loss={tot/n:.4f} val_f1={m['f1']:.4f} rec={m['recall']:.4f} prec={m['precision']:.4f}", flush=True)

        if m["f1"] > best_f1:
            best_f1, best_ep, bad = m["f1"], ep, 0
            torch.save({"head": model.state_dict(), "epoch": ep, "val_f1": m["f1"]},
                       args.out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping en", ep)
                break

    log.close()
    ck = torch.load(args.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["head"]); model.eval()
    print("mejor epoch", ck["epoch"], "val_f1", round(ck["val_f1"], 4))

    if (args.emb_dir / "test_emb.npy").exists():
        xte, yte = load_split(args.emb_dir, "test")
        with torch.no_grad():
            lg = model(torch.from_numpy(xte).to(device)).float().cpu().numpy()
        mt = metrics(lg, yte)
        print("TEST", json.dumps({k: round(v, 4) for k, v in mt.items()}))
        with open(args.out_dir / "test_metrics.json", "w") as f:
            json.dump(mt, f, indent=2)


if __name__ == "__main__":
    main()
