"""Compara entrenar con 4 vs 8 bloques, evaluando ambos sobre los mismos embeddings (8 bloques)."""
import os, sys, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import Head, load_split, metrics

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def entrenar(xtr, ytr, xva, yva, seed=100, epochs=100, patience=10, bs=4096):
    torch.manual_seed(seed)
    m = Head(hidden=xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    L = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(xva).to(DEV), torch.from_numpy(yva).to(DEV)
    best, bep, bad, est = -1, -1, 0, None
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); loss = L(m(xt[b].to(DEV)), yt[b].to(DEV)); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mm = metrics(m(xv).float().cpu().numpy(), yva, 0.5)
        if mm["balanced_accuracy"] > best:
            best, bep, bad = mm["balanced_accuracy"], ep, 0
            est = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(est); m.eval()
    return m, bep

def probs(m, x):
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 20000):
            out[i:i+20000] = torch.sigmoid(m(torch.from_numpy(x[i:i+20000]).to(DEV)).float()).cpu().numpy()
    return out

def juntar(d, split):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(d, split, "*_emb.npy"))):
        l = f.replace("_emb.npy", "_labels.npy")
        if os.path.exists(l):
            X.append(np.load(f)); Y.append(np.load(l))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

print("== cargando embeddings de 48 kb (evaluación común)")
xe_tr, ye_tr = juntar("trabajo/piloto48/emb", "train")
xe_va, ye_va = juntar("trabajo/piloto48/emb", "val")
xe_te, ye_te = juntar("trabajo/piloto48/emb", "test")
print(f"   train48 {xe_tr.shape} val48 {xe_va.shape} test48 {xe_te.shape}")

print("== entrenando con 48 kb")
m8, ep8 = entrenar(xe_tr, ye_tr, xe_va, ye_va)
p8 = probs(m8, xe_te)
r8 = metrics(np.log(np.clip(p8,1e-6,1-1e-6)/np.clip(1-p8,1e-6,1)), ye_te, 0.5)

print("== cargando embeddings de 24 kb para entrenar")
xt_tr, yt_tr = juntar("trabajo/piloto24/emb", "train")
xt_va, yt_va = juntar("trabajo/piloto24/emb", "val")
print(f"   train24 {xt_tr.shape} val24 {xt_va.shape}")

print("== entrenando con 24 kb (evaluado sobre 48 kb)")
m4, ep4 = entrenar(xt_tr, yt_tr, xt_va, yt_va)
p4 = probs(m4, xe_te)
r4 = metrics(np.log(np.clip(p4,1e-6,1-1e-6)/np.clip(1-p4,1e-6,1)), ye_te, 0.5)

print("\nComparación sobre el MISMO test (8 bloques)")
print(f"{'entrenado con':16s} {'BA':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} {'PR-AUC':>7s}")
for n, r in (("24 kb (4 blq)", r4), ("48 kb (8 blq)", r8)):
    print(f"{n:16s} {r['balanced_accuracy']:7.3f} {r['f1']:7.3f} {r['precision']:7.3f} "
          f"{r['recall']:7.3f} {r['pr_auc']:7.3f}")
print(f"\ndelta BA {r8['balanced_accuracy']-r4['balanced_accuracy']:+.4f} · "
      f"delta F1 {r8['f1']-r4['f1']:+.4f}")
