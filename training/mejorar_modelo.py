"""Mejora el modelo sin recalcular embeddings: ensamble de semillas y umbral por grupo.

Hace tres cosas y las compara:

  1. Entrena K cabezales (K semillas distintas) sobre los embeddings ya calculados.
  2. Evalúa cada semilla por separado y el ensamble (promedio de probabilidades).
  3. Calibra el umbral de dos formas: uno global y uno por grupo taxonómico.

Uso (animales, con grupos):
  python mejorar_modelo.py --emb_dir trabajo/emb/Animales --reino Animales \
      --out_dir trabajo/mejora/Animales --group_csv docs/animales_taxonomia.csv --seeds 3

Uso (sin grupos, solo ensamble):
  python mejorar_modelo.py --emb_dir trabajo/emb/Hongos --reino Hongos \
      --out_dir trabajo/mejora/Hongos --seeds 3
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mlp import Head, metrics  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cargar_grupos(path):
    out = {}
    if not path:
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            sp = row.get("species")
            if sp:
                out[sp] = row.get("grupo", "sin_grupo")
    return out


def predecir_split(head, emb_dir, split, device):
    """species -> (probs, labels) para un split, usando un cabezal."""
    out = {}
    for f in sorted(glob.glob(os.path.join(emb_dir, split, "*_emb.npy"))):
        sp = os.path.basename(f)[:-8]
        lp = f.replace("_emb.npy", "_labels.npy")
        if not os.path.exists(lp):
            continue
        x = np.load(f).astype(np.float32)
        y = np.load(lp).astype(np.float32)
        if len(x) == 0:
            continue
        with torch.no_grad():
            lg = head(torch.from_numpy(x).to(device)).float().cpu().numpy()
        out[sp] = (sigmoid(lg), y)
    return out


def cargar_head(path, device):
    ck = torch.load(path, map_location=device)
    head = Head(int(ck.get("hidden", 1280))).to(device)
    head.load_state_dict(ck["head"])
    head.eval()
    return head, ck


def juntar(preds, especies):
    ps, ys = [], []
    for sp in especies:
        if sp in preds:
            ps.append(preds[sp][0]); ys.append(preds[sp][1])
    if not ps:
        return np.array([]), np.array([])
    return np.concatenate(ps), np.concatenate(ys)


def mejor_umbral(probs, labels, lo=0.02, hi=0.98, step=0.01):
    """Umbral que maximiza balanced accuracy."""
    if len(labels) == 0 or labels.min() == labels.max():
        return None
    lg = logit(probs)
    grid = np.round(np.arange(lo, hi, step), 4)
    mejor = (-1.0, 0.5)
    for t in grid:
        m = metrics(lg, labels, float(t))
        if m["balanced_accuracy"] > mejor[0]:
            mejor = (m["balanced_accuracy"], float(t))
    return mejor[1]


def por_especie_probs(preds):
    """Lista (sp, prob, label) para aplicar umbrales por especie."""
    sps, ps, ys = [], [], []
    for sp, (p, y) in preds.items():
        sps.extend([sp] * len(p)); ps.append(p); ys.append(y)
    return np.concatenate(ps), np.concatenate(ys), sps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--reino", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_csv", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--select", choices=("f1", "balanced_accuracy"),
                    default="balanced_accuracy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    grupos = cargar_grupos(args.group_csv)
    resultados = {"reino": args.reino, "seeds": args.seeds, "semillas": []}

    # 1) entrenar una semilla por vez
    for i in range(args.seeds):
        seed = 100 + i
        d = os.path.join(args.out_dir, f"seed{seed}")
        if not os.path.exists(os.path.join(d, "best.pt")):
            cmd = [sys.executable, os.path.join(HERE, "train_mlp.py"),
                   "--emb_dir", args.emb_dir, "--out_dir", d,
                   "--epochs", str(args.epochs), "--patience", str(args.patience),
                   "--batch_size", str(args.batch_size), "--balance", "global",
                   "--class_weight", "--calibrate", "--select", args.select,
                   "--seed", str(seed)]
            print("entrenando semilla", seed, flush=True)
            subprocess.run(cmd, check=True)
        head, ck = cargar_head(os.path.join(d, "best.pt"), device)
        val = predecir_split(head, args.emb_dir, "val", device)
        test = predecir_split(head, args.emb_dir, "test", device)
        resultados["semillas"].append({"seed": seed, "dir": d, "val": val, "test": test})

    # 2) ensamble: promedio de probabilidades
    def promedio(split):
        acc = {}
        for r in resultados["semillas"]:
            for sp, (p, y) in r[split].items():
                if sp not in acc:
                    acc[sp] = [np.zeros_like(p), y]
                acc[sp][0] += p
        return {sp: (v[0] / args.seeds, v[1]) for sp, v in acc.items()}

    val_ens, test_ens = promedio("val"), promedio("test")

    # 3) umbral global (sobre ensamble)
    pv, yv, spv = por_especie_probs(val_ens)
    thr_global = mejor_umbral(pv, yv) or 0.5

    # 4) umbrales por grupo
    umbrales = {}
    if grupos:
        grupos_val = {}
        for sp, (p, y) in val_ens.items():
            g = grupos.get(sp)
            if g:
                grupos_val.setdefault(g, []).append((p, y))
        for g, lst in grupos_val.items():
            p = np.concatenate([x[0] for x in lst])
            y = np.concatenate([x[1] for x in lst])
            t = mejor_umbral(p, y)
            if t is not None:
                umbrales[g] = t
                print(f"  umbral {g:26s} = {t}")

    # 5) evaluar en test
    pt, yt, spt = por_especie_probs(test_ens)
    mapa = {sp: grupos.get(sp, "sin_grupo") for sp in test_ens} if grupos else {}

    def reporte(nombre, probs, labels, thr_vec):
        pred = (probs >= thr_vec).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ba = (rec + spec) / 2
        print(f"{nombre:30s} {ba:7.3f} {f1:7.3f} {prec:7.3f} {rec:7.3f} {spec:7.3f}")
        return {"balanced_accuracy": ba, "f1": f1, "precision": prec,
                "recall": rec, "specificity": spec}

    print("\nComparación en test")
    print(f"{'':30s} {'BA':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} {'Spec':>7s}")
    bas = []
    for r in resultados["semillas"]:
        p, y, _ = por_especie_probs(r["test"])
        bas.append(reporte(f"semilla {r['seed']}", p, y,
                           np.full(len(p), thr_global)))
    ens_glob = reporte(f"ensamble · thr {thr_global:.3f}", pt, yt,
                       np.full(len(pt), thr_global))

    ens_grupo = None
    if grupos and umbrales:
        thr_vec = np.array([umbrales.get(mapa.get(sp, "sin_grupo"), thr_global)
                            for sp in spt])
        ens_grupo = reporte("ensamble · umbral por grupo", pt, yt, thr_vec)

    resumen = {
        "reino": args.reino,
        "seeds": args.seeds,
        "thr_global": round(thr_global, 4),
        "umbrales_por_grupo": {k: round(v, 3) for k, v in umbrales.items()},
        "test_ba_media_semillas": round(float(np.mean([b["balanced_accuracy"] for b in bas])), 4),
        "test_ba_ensamble_global": round(ens_glob["balanced_accuracy"], 4),
        "test_ensamble_global": ens_glob,
    }
    if ens_grupo:
        resumen["test_ba_ensamble_por_grupo"] = round(ens_grupo["balanced_accuracy"], 4)
        resumen["test_ensamble_por_grupo"] = ens_grupo

    print("\nResumen guardado en", os.path.join(args.out_dir, "resumen.json"))
    with open(os.path.join(args.out_dir, "resumen.json"), "w") as f:
        json.dump(resumen, f, indent=2)


if __name__ == "__main__":
    main()
