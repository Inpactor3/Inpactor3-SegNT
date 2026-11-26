#!/usr/bin/env python3
"""Entrena el cabezal MIL con pesos por grupo taxonomico.

Diferencia con train_mil.py --balance:
  * No recorta ventanas ni elimina negativas. Cada ventana recibe un peso
    w = 1 / ventanas_del_grupo, normalizado a media 1 y capado (--cap).
  * Los grupos con menos de --min_group_windows se fusionan en "Otros" para que
    una especie con 200 ventanas no reciba un peso 250x.
  * Reporta metrica global, macro PR-AUC por grupo y lift por grupo.

Uso (en la maquina con la cache y GPU):
  python train_group_weights.py --emb_dir trabajo/emb/Animales \
      --out_dir trabajo/ckpt/Animales_balance \
      --group_csv docs/animales_taxonomia_limpio.csv \
      --epochs 30 --pool logits --cap 3 --min_group_windows 2000 --class_weight
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA = 1e-4


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


def lift(m):
    if m["prevalence"] >= 1.0:
        return 0.0
    return (m["pr_auc"] - m["prevalence"]) / (1.0 - m["prevalence"])


def calibrate_threshold(logits, labels):
    p = sigmoid(logits)
    grid = np.unique(np.round(np.arange(0.02, 0.99, 0.01), 4))
    best = (-1.0, 0.5)
    for t in grid:
        m = metrics(logits, labels, t)
        if m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], float(t))
    lo = max(0.01, best[1] - 0.02)
    hi = min(0.99, best[1] + 0.02)
    for t in np.arange(lo, hi + 1e-9, 0.005):
        m = metrics(logits, labels, float(t))
        if m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], float(round(t, 3)))
    return best[1]


def norm_tag(s):
    return re.sub(r"_+", "_", s.lower())


def load_groups(path):
    exact, norm, by_acc = {}, {}, {}
    acc_re = re.compile(r"(GC[AF]_\d+\.\d+)")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sp = (row.get("species") or row.get("especie") or "").strip()
            grp = (row.get("grupo") or row.get("group") or "sin_grupo").strip()
            if not sp:
                continue
            exact[sp] = grp
            norm[norm_tag(sp)] = grp
            m = acc_re.search(sp)
            if m:
                by_acc[m.group(1)] = grp
    return exact, norm, by_acc


def group_for(stem, maps):
    exact, norm, by_acc = maps
    if stem in exact:
        return exact[stem]
    if norm_tag(stem) in norm:
        return norm[norm_tag(stem)]
    m = re.search(r"(GC[AF]_\d+\.\d+)", stem)
    if m and m.group(1) in by_acc:
        return by_acc[m.group(1)]
    return "sin_grupo"


class MILHead(nn.Module):
    """pool='logits' -> MIL por bloque; pool='embeddings' -> max antes del cabezal."""

    def __init__(self, hidden=512, dropout=0.2, pool="logits"):
        super().__init__()
        self.pool = pool
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(512, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x, keep=None):
        B, C, H = x.shape
        if self.pool == "embeddings":
            if keep is not None:
                x = x.masked_fill(~keep.unsqueeze(-1), -1e9)
            e = x.max(dim=1).values
            f = self.dropout(torch.relu(self.pool_proj(e)))
            return self.cls(f).squeeze(-1)
        f = self.dropout(torch.relu(self.pool_proj(x.reshape(B * C, H))))
        lg = self.cls(f).view(B, C)
        if keep is not None:
            lg = lg.masked_fill(~keep, -1e9)
        return lg.max(dim=1).values


def load_split(emb_dir, split, maps, drop_unmapped=True):
    files = sorted((emb_dir / split).glob("*_emb.npy"))
    if not files:
        raise FileNotFoundError(f"sin embeddings en {emb_dir}/{split}")
    xs, ys, gs, dropped = [], [], [], []
    for f in files:
        g = group_for(f.name[:-8], maps)
        if g == "sin_grupo" and drop_unmapped:
            dropped.append(f.name[:-8])
            continue
        e = np.load(f)
        if e.ndim == 3 and e.shape[0] == 0:
            continue
        if e.ndim == 2:
            e = e[:, None, :]
        if e.ndim != 3:
            raise ValueError(f"{f.name}: forma inesperada {e.shape}")
        lab = np.load(f.with_name(f.name.replace("_emb.npy", "_labels.npy")))
        xs.append(e.astype(np.float16))
        ys.append(lab.astype(np.float32))
        gs.extend([g] * len(lab))
    if dropped:
        print(f"{split}: {len(dropped)} especies sin grupo omitidas: "
              f"{', '.join(dropped[:4])}{' ...' if len(dropped) > 4 else ''}", flush=True)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    g = np.array(gs, dtype=object)
    print(f"{split}: {len(files) - len(dropped)} archivos · {x.shape[0]} ventanas · "
          f"{len(set(g))} grupos", flush=True)
    return x, y, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--group_csv", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pool", choices=("logits", "embeddings"), default="logits")
    ap.add_argument("--cap", type=float, default=3.0,
                    help="peso maximo por grupo antes de normalizar (3 = 3x)")
    ap.add_argument("--min_group_windows", type=int, default=2000,
                    help="grupos con menos ventanas de train se fusionan en Otros")
    ap.add_argument("--class_weight", action="store_true",
                    help="pos_weight global dentro de cada grupo")
    ap.add_argument("--select", choices=("balanced_accuracy", "f1"), default="balanced_accuracy")
    ap.add_argument("--keep_unmapped", action="store_true",
                    help="no descartar especies que no estan en el CSV de grupos")
    ap.add_argument("--no_group_weights", action="store_true",
                    help="control: todos los pesos = 1 (sin rebalanceo)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    maps = load_groups(args.group_csv)
    keep = not args.keep_unmapped
    xtr, ytr, gtr = load_split(args.emb_dir, "train", maps, keep)
    xva, yva, gva = load_split(args.emb_dir, "val", maps, keep)
    xte, yte, gte = load_split(args.emb_dir, "test", maps, keep)

    counts = Counter(gtr)
    merged = {g for g, n in counts.items() if n < args.min_group_windows}
    if merged:
        print(f"fusionando {len(merged)} grupos pequenos en 'Otros': "
              f"{sorted(merged)[:6]}{' ...' if len(merged) > 6 else ''}", flush=True)
        gtr = np.array(["Otros" if g in merged else g for g in gtr], dtype=object)
        gva = np.array(["Otros" if g in merged else g for g in gva], dtype=object)
        gte = np.array(["Otros" if g in merged else g for g in gte], dtype=object)

    counts = Counter(gtr)
    total = len(gtr)
    groups = sorted(counts)
    G = len(groups)
    if args.no_group_weights:
        factor = {g: 1.0 for g in groups}
        print("control: sin pesos por grupo", flush=True)
    else:
        raw = {g: total / (G * counts[g]) for g in groups}
        factor = {g: min(raw[g], args.cap) for g in groups}
        mean_w = sum(factor[g] * counts[g] for g in groups) / total
        factor = {g: factor[g] / mean_w for g in groups}
        factor = {g: min(factor[g], args.cap) for g in groups}
        mean_w = sum(factor[g] * counts[g] for g in groups) / total
        factor = {g: min(factor[g] / mean_w, args.cap) for g in groups}

    with open(args.out_dir / "weights_by_group.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "ventanas_train", "share", "factor_capado"])
        for g in sorted(groups, key=lambda x: -counts[x]):
            w.writerow([g, counts[g], round(counts[g] / total, 4), round(factor[g], 4)])
    print("\nPesos por grupo (media 1, cap %.1f):" % args.cap, flush=True)
    for g in sorted(groups, key=lambda x: -counts[x]):
        print(f"  {g:28s} n={counts[g]:7d}  {counts[g]/total*100:5.2f}%  peso={factor[g]:6.2f}x",
              flush=True)
    print(flush=True)

    wtr = torch.tensor([factor[g] for g in gtr], dtype=torch.float32)

    muestra = xtr.reshape(-1, xtr.shape[-1]).astype(np.float32)
    if len(muestra) > 200_000:
        sel = rng.choice(len(muestra), 200_000, replace=False)
        muestra = muestra[sel]
    mu = muestra.mean(axis=0).astype(np.float32)
    sd = muestra.std(axis=0).astype(np.float32) + SIGMA

    model = MILHead(xtr.shape[-1], args.dropout, args.pool).to(device)
    pos_weight = None
    if args.class_weight:
        neg = int((ytr == 0).sum())
        pos = int((ytr == 1).sum())
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        print(f"class_weight pos_weight={float(pos_weight):.4f}", flush=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)

    def a_device(x):
        return (torch.from_numpy(x).to(device).float() - mu_t) / sd_t

    yt = torch.from_numpy(ytr).to(device)
    wt = wtr.to(device)
    xv = a_device(xva)
    xte_t = a_device(xte)

    def predecir(x):
        salida = []
        for i in range(0, len(x), 8192):
            with torch.no_grad():
                salida.append(model(x[i:i + 8192]).float().cpu().numpy())
        return np.concatenate(salida)

    def per_group(lg, yy, gg, thr):
        rows = []
        for g in sorted(set(gg)):
            mask = gg == g
            if mask.sum() == 0:
                continue
            m = metrics(lg[mask], yy[mask], thr)
            m["group"] = g
            m["lift"] = round(lift(m), 4)
            m["evaluable"] = bool(mask.sum() >= 50 and 0 < m["prevalence"] < 1)
            rows.append(m)
        ev = [r for r in rows if r["evaluable"]]
        macro = float(np.mean([r["pr_auc"] for r in ev])) if ev else 0.0
        macro_ba = float(np.mean([r["balanced_accuracy"] for r in ev])) if ev else 0.0
        return rows, macro, macro_ba

    mejor, mejor_ep, sin_mejora = -1.0, -1, 0
    log = open(args.out_dir / "metrics.csv", "w")
    log.write("epoch,loss,val_loss,val_precision,val_recall,val_f1,val_acc,val_bal_acc,val_macro_pr_auc\n")
    for ep in range(1, args.epochs + 1):
        model.train()
        orden = torch.randperm(len(xtr))
        tot, n = 0.0, 0
        for i in range(0, len(orden), args.batch_size):
            b = orden[i:i + args.batch_size].numpy()
            xb = a_device(xtr[b])
            yb = yt[b]
            wb = wt[b]
            opt.zero_grad()
            loss = (loss_fn(model(xb), yb) * wb).sum() / wb.sum().clamp_min(1)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)

        model.eval()
        lgv = predecir(xv)
        vl = F.binary_cross_entropy_with_logits(
            torch.from_numpy(lgv), torch.from_numpy(yva)).item()
        m = metrics(lgv, yva)
        _, macro, _ = per_group(lgv, yva, gva, 0.5)
        log.write(f"{ep},{tot/n:.4f},{vl:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                  f"{m['f1']:.4f},{m['accuracy']:.4f},{m['balanced_accuracy']:.4f},{macro:.4f}\n")
        log.flush()
        print(f"ep {ep:3d} loss={tot/n:.4f} val_bal_acc={m['balanced_accuracy']:.4f} "
              f"val_macro_pr={macro:.4f}", flush=True)

        if m[args.select] > mejor:
            mejor, mejor_ep, sin_mejora = m[args.select], ep, 0
            torch.save({"head": model.state_dict(), "epoch": ep, "hidden": int(xtr.shape[-1]),
                        "mu": mu, "sd": sd, "val_selected": mejor, "select": args.select,
                        "mode": f"{args.pool}_max", "blocks": int(xtr.shape[1]),
                        "group_cap": args.cap, "min_group_windows": args.min_group_windows},
                       args.out_dir / "best.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= args.patience:
                print(f"sin mejora en {args.patience} epocas; paro en {ep}", flush=True)
                break

    ck = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["head"])
    model.eval()
    lgv = predecir(xv)
    lgte = predecir(xte_t)
    thr = calibrate_threshold(lgv, yva)
    ck["threshold"] = thr
    torch.save(ck, args.out_dir / "best.pt")

    rows_te, macro_te, macro_ba_te = per_group(lgte, yte, gte, thr)
    res = {"threshold": thr, "epoch": ck["epoch"], "blocks": ck["blocks"],
           "mode": f"{args.pool}_max", "group_cap": args.cap,
           "min_group_windows": args.min_group_windows,
           "class_weight": args.class_weight,
           "at_calibrated_threshold": metrics(lgte, yte, thr),
           "at_050": metrics(lgte, yte, 0.5),
           "macro_pr_auc": macro_te, "macro_balanced_accuracy": macro_ba_te,
           "n_groups_evaluables": sum(1 for r in rows_te if r["evaluable"])}
    with open(args.out_dir / "test_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(args.out_dir / "per_group_test.csv", "w", newline="") as f:
        cols = ["group", "n", "prevalence", "pr_auc", "lift", "balanced_accuracy",
                "f1", "precision", "recall", "evaluable"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows_te, key=lambda x: -x["n"]):
            w.writerow(r)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in res["at_calibrated_threshold"].items()}, indent=2), flush=True)
    print(f"macro PR-AUC={macro_te:.4f} en {res['n_groups_evaluables']} grupos", flush=True)


if __name__ == "__main__":
    main()
