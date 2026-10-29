"""Métricas por grupo taxonómico sobre un split (normalmente test).

Sirve para saber si el modelo falla en todos los grupos o solo en algunos, antes
de decidir cualquier rebalanceo.

Uso:
  python eval_por_grupo.py --emb_dir trabajo/emb/Animales --split test \
      --head trabajo/ckpt/Animales_ba/best.pt \
      --group_csv docs/animales_taxonomia.csv --out trabajo/por_grupo_test.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import Head, metrics  # noqa: E402


def cargar_grupos(path):
    """species_tag -> grupo. Acepta el CSV de taxonomía o uno simple."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sp = row.get("species") or row.get("especie") or ""
            g = row.get("grupo") or row.get("group") or "sin_grupo"
            if sp:
                out[sp] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--head", required=True, help="best.pt con el cabezal entrenado")
    ap.add_argument("--group_csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="si se omite, usa el del checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=device)
    hidden = int(ck.get("hidden", 1280))
    mu, sd = ck.get("mu"), ck.get("sd")
    thr = args.threshold if args.threshold is not None else float(ck.get("threshold", 0.5))

    head = Head(hidden).to(device)
    head.load_state_dict(ck["head"])
    head.eval()

    grupos = cargar_grupos(args.group_csv)
    files = sorted(glob.glob(os.path.join(args.emb_dir, args.split, "*_emb.npy")))
    if not files:
        raise SystemExit(f"sin embeddings en {args.emb_dir}/{args.split}")

    por_grupo = {}
    sin_grupo = []
    no_evaluables = []
    total_probs, total_labs = [], []

    for f in files:
        sp = os.path.basename(f)[:-8]
        lab_path = f.replace("_emb.npy", "_labels.npy")
        if not os.path.exists(lab_path):
            print("AVISO sin labels:", sp)
            continue
        x = np.load(f).astype(np.float32)
        y = np.load(lab_path).astype(np.float32)
        if len(x) == 0:
            continue
        if mu is not None:
            m = np.asarray(mu, dtype=np.float32)
            s = np.asarray(sd, dtype=np.float32)
            x = ((x - m) / s).astype(np.float32)
        with torch.no_grad():
            lg = head(torch.from_numpy(x).to(device)).float().cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-np.clip(lg, -60, 60)))

        # una especie con una sola clase no permite medir balanced accuracy
        if y.mean() in (0.0, 1.0):
            no_evaluables.append((sp, float(y.mean()), int(len(y))))

        g = grupos.get(sp, "SIN GRUPO")
        if g == "SIN GRUPO":
            sin_grupo.append(sp)
        por_grupo.setdefault(g, {"p": [], "y": []})
        por_grupo[g]["p"].append(p)
        por_grupo[g]["y"].append(y)
        if y.mean() not in (0.0, 1.0):
            total_probs.append(p)
            total_labs.append(y)

    filas = []
    for g, d in por_grupo.items():
        p = np.concatenate(d["p"])
        y = np.concatenate(d["y"])
        # metrics() espera logits: convertimos las probabilidades a log-odds
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        m = metrics(lg, y, thr)
        n_sp = sum(1 for f in files
                   if grupos.get(os.path.basename(f)[:-8]) == g)
        filas.append({
            "grupo": g, "especies": n_sp, "ventanas": int(len(y)),
            "prevalencia": round(float(y.mean()), 3),
            "bal_acc": round(m["balanced_accuracy"], 3),
            "f1": round(m["f1"], 3),
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "specificity": round(m["specificity"], 3),
            "pr_auc": round(m["pr_auc"], 3),
        })

    if not total_probs:
        raise SystemExit("no hay especies evaluables (todas con una sola clase)")

    P = np.concatenate(total_probs)
    Y = np.concatenate(total_labs)
    LG = np.log(np.clip(P, 1e-6, 1 - 1e-6) / np.clip(1 - P, 1e-6, 1))
    M = metrics(LG, Y, thr)
    filas.append({
        "grupo": "== TOTAL evaluable ==", "especies": len(total_probs),
        "ventanas": int(len(Y)),
        "prevalencia": round(float(Y.mean()), 3),
        "bal_acc": round(M["balanced_accuracy"], 3), "f1": round(M["f1"], 3),
        "precision": round(M["precision"], 3), "recall": round(M["recall"], 3),
        "specificity": round(M["specificity"], 3), "pr_auc": round(M["pr_auc"], 3),
    })

    filas.sort(key=lambda r: (not r["grupo"].startswith("== TOTAL"), -r["ventanas"]))

    print(f"Umbral usado: {thr}  |  split: {args.split}  |  {len(files)} especies\n")
    print(f"{'grupo':26s} {'esp':>4s} {'vent':>7s} {'prev':>5s} {'BA':>6s} "
          f"{'F1':>6s} {'P':>6s} {'R':>6s} {'Spec':>6s} {'PR-AUC':>7s}")
    for r in filas:
        print(f"{r['grupo']:26s} {r['especies']:4d} {r['ventanas']:7d} "
              f"{r['prevalencia']:5.2f} {r['bal_acc']:6.3f} {r['f1']:6.3f} "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['specificity']:6.3f} "
              f"{r['pr_auc']:7.3f}")

    if no_evaluables:
        print("\nExcluidas del total (una sola clase, balanced accuracy no aplica):")
        for sp, prev, n in no_evaluables:
            etiqueta = "100 % positiva" if prev == 1.0 else "100 % negativa"
            print(f"   {sp[:48]:48s} {n:6d} ventanas · {etiqueta}")

    if sin_grupo:
        print(f"\nSin grupo asignado ({len(sin_grupo)}): {sin_grupo[:5]}")

    if args.out:
        with open(args.out, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
