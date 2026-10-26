"""Entrena solo el cabezal MLP sobre embeddings precalculados (backbone congelado).

Cambios v2 (entrada 24 kb con max pooling):
  * `--balance 1` muestrea el train a relacion 1:1 (topicos balanceados 1:1) y
    penaliza lo remanente con `--pos_weight` (class_weight) sobre la relacion real.
  * `--balance_species` hace el 1:1 por especie (evita que una especie sin negativas
    empuje el balance global; ver README).
  * `--calibrate` elige el umbral por **balanced accuracy** en validacion: maximiza
    Youden J y luego refina por tramos; se guarda en `calibration.json` y viaja
    dentro de `best.pt` para que la evaluacion gond use exactamente el mismo umbral.
  * Reporta PR-AUC y balanced accuracy en test, no solo F1.
  * Guarda stats de estandarizacion en `best.pt` (necesario para evaluacion gond).
"""
import argparse
import json
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


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def metrics(logits, labels, thr=0.5):
    p = sigmoid(logits)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = max(len(labels), 1)
    out = {"precision": prec, "recall": rec, "specificity": spec, "f1": f1,
           "accuracy": (tp + tn) / n, "balanced_accuracy": (rec + spec) / 2,
           "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(labels)),
           "prevalence": float(labels.mean()) if len(labels) else 0.0}
    if labels.sum() > 0:
        order = np.argsort(-p)
        y = labels[order]
        tpc = np.cumsum(y)
        fpc = np.cumsum(1 - y)
        pc = tpc / np.maximum(tpc + fpc, 1)
        rc = tpc / max(int(labels.sum()), 1)
        out["pr_auc"] = float(np.sum(np.diff(np.concatenate([[0], rc])) * pc))
    else:
        out["pr_auc"] = 0.0
    return out


def calibrate_threshold(logits, labels):
    """Umbral que maximiza balanced accuracy (Youden) con refinamiento local."""
    p = sigmoid(logits)
    grid = np.unique(np.round(np.arange(0.02, 0.99, 0.01), 4))
    best = (-1.0, 0.5)
    for t in grid:
        m = metrics(logits, labels, t)
        if m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], float(t))
    lo = max(0.01, best[1] - 0.02)
    hi = min(0.995, best[1] + 0.02)
    for t in np.arange(lo, hi, 0.001):
        m = metrics(logits, labels, t)
        if m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], float(t))
    t = best[1]
    m = metrics(logits, labels, t)
    return {"threshold": round(t, 3),
            "objective": "balanced_accuracy",
            "val_balanced_accuracy": round(m["balanced_accuracy"], 4),
            "val_f1": round(m["f1"], 4),
            "val_precision": round(m["precision"], 4),
            "val_recall": round(m["recall"], 4),
            "val_specificity": round(m["specificity"], 4)}


def load_split(d, split):
    single = d / f"{split}_emb.npy"
    if single.exists():
        emb = np.load(single).astype(np.float32)
        lab = np.load(d / f"{split}_labels.npy").astype(np.float32)
        return emb, lab
    files = sorted((d / split).glob("*_emb.npy"))
    if not files:
        raise FileNotFoundError(f"sin embeddings para {split} en {d}")
    embs, labs = [], []
    for f in files:
        embs.append(np.load(f))
        labs.append(np.load(f.with_name(f.name.replace("_emb.npy", "_labels.npy"))))
    print(f"{split}: {len(files)} archivos por especie, {sum(len(x) for x in embs)} ventanas")
    return np.concatenate(embs).astype(np.float32), np.concatenate(labs).astype(np.float32)


def balance_indices(y, mode, seed):
    """Devuelve indices de entrenamiento 1:1 (global o por especie)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if mode == "global":
        k = min(len(pos), len(neg))
        return np.sort(np.concatenate([rng.choice(pos, k, replace=False),
                                       rng.choice(neg, k, replace=False)]))
    return None


def species_groups(d, split, n):
    """Agrupa indices por archivo de especie (mismo orden que load_split)."""
    groups = []
    off = 0
    for f in sorted((d / split).glob("*_emb.npy")):
        m = len(np.load(f, mmap_mode="r"))
        groups.append(np.arange(off, off + m))
        off += m
    if off != n:
        return [np.arange(n)]
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--select", choices=("f1", "balanced_accuracy"), default="f1",
                    help="metrica de validacion para elegir la mejor epoca. Con "
                         "validacion muy desbalanceada usa balanced_accuracy: el F1 "
                         "premia al modelo que predice casi todo positivo.")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--class_weight", action="store_true",
                    help="pos_weight = n_neg/n_pos sobre el train usado")
    ap.add_argument("--balance", choices=("none", "global", "species"), default="none",
                    help="submuestreo 1:1 del train (global o por especie)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true",
                    help="calibrar umbral en validacion (balanced accuracy)")
    ap.add_argument("--standardize", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr = load_split(args.emb_dir, "train")
    xva, yva = load_split(args.emb_dir, "val")
    print("train", xtr.shape, "val", xva.shape, "pos frac train", float(ytr.mean()))

    if args.balance != "none":
        if args.balance == "global":
            idx = balance_indices(ytr, "global", args.seed)
        else:
            groups = species_groups(args.emb_dir, "train", len(ytr))
            picks = []
            for g in groups:
                yg = ytr[g]
                gp = g[yg == 1]
                gn = g[yg == 0]
                if len(gp) == 0 or len(gn) == 0:
                    picks.append(g)
                    continue
                k = min(len(gp), len(gn))
                rng = np.random.default_rng(args.seed + len(picks))
                picks.append(np.sort(np.concatenate([
                    rng.choice(gp, k, replace=False), rng.choice(gn, k, replace=False)])))
            idx = np.sort(np.concatenate(picks))
        before = len(ytr)
        xtr, ytr = xtr[idx], ytr[idx]
        print(f"balance {args.balance}: {before} -> {len(ytr)} ventanas "
              f"(pos frac {float(ytr.mean()):.3f})")

    mu = sd = None
    if args.standardize:
        mu_np = xtr.mean(axis=0, keepdims=True)
        sd_np = xtr.std(axis=0, keepdims=True) + 1e-6
        xtr = ((xtr - mu_np) / sd_np).astype(np.float32)
        xva = ((xva - mu_np) / sd_np).astype(np.float32)
        mu = mu_np.ravel().astype(float).tolist()
        sd = sd_np.ravel().astype(float).tolist()
        print("estandarizado: mu/sd calculados en train (50 mil ventanas)")

    model = Head(hidden=xtr.shape[1], dropout=args.dropout).to(device)
    pos_weight = None
    if args.class_weight:
        pos = float(ytr.sum())
        neg = float(len(ytr) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        print(f"class_weight pos_weight={float(pos_weight):.4f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(xva).to(device)
    yv = torch.from_numpy(yva).to(device)

    best_sel, best_ep, bad = -1.0, -1, 0
    log = open(args.out_dir / "metrics.csv", "w")
    log.write("epoch,loss,val_loss,val_precision,val_recall,val_f1,val_acc,val_bal_acc\n")

    for ep in range(1, args.epochs + 1):
        model.train()
        idx_ep = torch.randperm(len(xt))
        tot, n = 0.0, 0
        for i in range(0, len(xt), args.batch_size):
            b = idx_ep[i:i + args.batch_size]
            xb = xt[b].to(device)
            yb = yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)

        model.eval()
        with torch.no_grad():
            lg = model(xv)
            vl = nn.functional.binary_cross_entropy_with_logits(lg, yv).item()
            logits = lg.float().cpu().numpy()
        m = metrics(logits, yva)
        log.write(f"{ep},{tot/n:.4f},{vl:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                  f"{m['f1']:.4f},{m['accuracy']:.4f},{m['balanced_accuracy']:.4f}\n")
        log.flush()
        print(f"ep {ep:3d} loss={tot/n:.4f} val_f1={m['f1']:.4f} "
              f"val_bal_acc={m['balanced_accuracy']:.4f} rec={m['recall']:.4f} "
              f"spec={m['specificity']:.4f}", flush=True)

        sel = m[args.select]
        if sel > best_sel:
            best_sel, best_ep, bad = sel, ep, 0
            torch.save({"head": model.state_dict(), "epoch": ep, "val_f1": m["f1"],
                        "val_selected": sel, "select": args.select,
                        "hidden": xtr.shape[1], "mu": mu, "sd": sd,
                        "balance": args.balance, "class_weight": bool(args.class_weight),
                        "chunks": 4}, args.out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping en", ep)
                break

    log.close()
    ck = torch.load(args.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["head"])
    model.eval()
    print("mejor epoch", ck["epoch"], "| criterio", ck.get("select", "f1"),
          "| val_f1", round(ck["val_f1"], 4),
          "| val", round(ck.get("val_selected", ck["val_f1"]), 4), flush=True)

    if args.calibrate:
        model.eval()
        with torch.no_grad():
            lgv = model(xv).float().cpu().numpy()
        cal = calibrate_threshold(lgv, yva)
        vm = metrics(lgv, yva, cal["threshold"])
        cal["val_metrics_at_threshold"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in vm.items()}
        cal["val_metrics_at_050"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in metrics(lgv, yva, 0.5).items()}

        # Umbral alternativo: el que maximiza F1 en validacion. Con clases muy
        # desbalanceadas el de balanced accuracy suele ser demasiado alto.
        best_f1, best_t = -1.0, 0.5
        for t in np.unique(np.round(np.arange(0.02, 0.99, 0.01), 4)):
            m = metrics(lgv, yva, float(t))
            if m["f1"] > best_f1:
                best_f1, best_t = m["f1"], float(t)
        cal["threshold_max_f1"] = round(best_t, 3)
        cal["val_f1_at_max_f1"] = round(best_f1, 4)

        with open(args.out_dir / "calibration.json", "w") as f:
            json.dump(cal, f, indent=2)
        ck["threshold"] = cal["threshold"]
        ck["threshold_max_f1"] = best_t
        ck["calibration"] = cal
        torch.save(ck, args.out_dir / "best.pt")
        print("calibracion", json.dumps(cal, ensure_ascii=False), flush=True)
    else:
        ck["threshold"] = 0.5
        torch.save(ck, args.out_dir / "best.pt")

    thr = float(ck.get("threshold", 0.5))
    if (args.emb_dir / "test_emb.npy").exists() or (args.emb_dir / "test").is_dir():
        xte, yte = load_split(args.emb_dir, "test")
        if args.standardize and mu is not None:
            xte = ((xte - np.asarray(mu, dtype=np.float32)) /
                   np.asarray(sd, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            lg = model(torch.from_numpy(xte).to(device)).float().cpu().numpy()
        res = {"threshold": thr,
               "at_calibrated_threshold": metrics(lg, yte, thr),
               "at_050": metrics(lg, yte, 0.5),
               "test_prevalence": float(yte.mean())}

        # Con prevalencia muy alta (Plantas: 92% positivo) F1 y accuracy enganan:
        # un clasificador trivial supera al modelo. Se reporta ademas un test
        # balanceado 1:1 (todas las negativas + igual numero de positivas).
        rng = np.random.default_rng(args.seed)
        neg_i = np.flatnonzero(yte == 0)
        pos_i = np.flatnonzero(yte == 1)
        k = int(min(len(neg_i), len(pos_i)))
        if k > 0:
            bal = np.sort(np.concatenate([rng.choice(neg_i, k, replace=False),
                                          rng.choice(pos_i, k, replace=False)]))
            res["balanced_test"] = {
                "n": int(len(bal)),
                "at_calibrated_threshold": metrics(lg[bal], yte[bal], thr),
                "at_050": metrics(lg[bal], yte[bal], 0.5),
            }
        if "threshold_max_f1" in ck:
            t_f1 = float(ck["threshold_max_f1"])
            res["threshold_max_f1"] = t_f1
            res["at_max_f1_threshold"] = metrics(lg, yte, t_f1)
            if k > 0:
                res["balanced_test"]["at_max_f1_threshold"] = metrics(lg[bal], yte[bal], t_f1)

        print("TEST @", thr, json.dumps({k2: (round(v, 4) if isinstance(v, float) else v)
                                         for k2, v in res["at_calibrated_threshold"].items()}))
        if "balanced_test" in res:
            bt = res["balanced_test"]["at_050"]
            print("TEST balanceado (n=%d) @0.5: F1=%.4f bal_acc=%.4f prec=%.4f rec=%.4f"
                  % (res["balanced_test"]["n"], bt["f1"], bt["balanced_accuracy"],
                     bt["precision"], bt["recall"]), flush=True)
        with open(args.out_dir / "test_metrics.json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
