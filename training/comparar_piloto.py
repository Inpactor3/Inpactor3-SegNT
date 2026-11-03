"""Compara entrenar con 4 vs 8 bloques, evaluando ambos sobre el MISMO test de 8 bloques."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import Head, metrics

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def juntar(d, split):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(d, split, "*_emb.npy"))):
        l = f.replace("_emb.npy", "_labels.npy")
        if os.path.exists(l):
            X.append(np.load(f)); Y.append(np.load(l))
    if not X:
        return None, None
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

def entrenar(xtr, ytr, xva, yva, seed=100, epochs=100, patience=10, bs=4096):
    torch.manual_seed(seed)
    m = Head(hidden=xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    L = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(xva).to(DEV), torch.from_numpy(yva).to(DEV)
    best, bep, bad, est = -1.0, -1, 0, None
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); loss = L(m(xt[b].to(DEV)), yt[b].to(DEV))
            loss.backward(); opt.step()
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
    m.load_state_dict(est); m.eval()
    return m, bep

def probs(m, x):
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 20000):
            out[i:i+20000] = torch.sigmoid(
                m(torch.from_numpy(x[i:i+20000]).to(DEV)).float()).cpu().numpy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb48", required=True)
    ap.add_argument("--emb24", required=True)
    a = ap.parse_args()

    xte, yte = juntar(a.emb48, "test")
    if xte is None:
        raise SystemExit("sin embeddings de test en " + a.emb48)
    print(f"test comun (8 bloques): {xte.shape}", flush=True)

    r = {}
    for nombre, d in (("48 kb (8 blq)", a.emb48), ("24 kb (4 blq)", a.emb24)):
        xtr, ytr = juntar(d, "train")
        xva, yva = juntar(d, "val")
        if xtr is None or xva is None:
            raise SystemExit(f"faltan embeddings en {d}")
        print(f"entrenando {nombre}: train {xtr.shape} val {xva.shape}", flush=True)
        m, ep = entrenar(xtr, ytr, xva, yva)
        p = probs(m, xte)
        lg = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))
        r[nombre] = metrics(lg, yte, 0.5)
        print(f"   {nombre}: ep {ep} test BA {r[nombre]['balanced_accuracy']:.4f}", flush=True)

    print("\nComparación sobre el MISMO test (8 bloques)")
    print(f"{'entrenado con':16s} {'BA':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} {'PR-AUC':>7s}")
    for n, m in r.items():
        print(f"{n:16s} {m['balanced_accuracy']:7.3f} {m['f1']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['pr_auc']:7.3f}")
    d8, d4 = r["48 kb (8 blq)"], r["24 kb (4 blq)"]
    print(f"\ndelta (48k - 24k): BA {d8['balanced_accuracy']-d4['balanced_accuracy']:+.4f} · "
          f"F1 {d8['f1']-d4['f1']:+.4f} · recall {d8['recall']-d4['recall']:+.4f}")

if __name__ == "__main__":
    main()
