"""Visor de resultados, solo lectura.

Lee los JSON de resultados y los CSV por epoca. No ejecuta modelos, no sube archivos y no
expone pesos. Pensado para la demo y para integrarse en el pipeline de Inpactor2.

Uso:
  python3 visor/app.py --port 8000
"""
import argparse
import csv
import json
from pathlib import Path

from flask import Flask, abort, render_template

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RES_VENTANAS = DOCS / "ventanas"
RES_NTV3 = DOCS / "ntv3_resultados.json"
EPOCAS = DOCS / "ntv3_epocas"

app = Flask(__name__)
CACHE = {}


@app.template_filter("f4")
def f4(v):
    return "—" if v is None else f"{v:.4f}".replace(".", ",")


@app.template_filter("pct")
def pct(v):
    return "—" if v is None else f"{v*100:.1f} %".replace(".", ",")


def _json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def cargar_ventanas():
    if "ventanas" in CACHE:
        return CACHE["ventanas"]
    runs = []
    if RES_VENTANAS.exists():
        for d in sorted(RES_VENTANAS.iterdir()):
            f = d / "test_metrics.json"
            if not f.exists():
                continue
            t = _json(f)
            if not t or "at_calibrated_threshold" not in t:
                continue
            m = t["at_calibrated_threshold"]
            try:
                reino, ventana, agreg = d.name.split("_")[:3]
                runs.append({
                    "id": d.name,
                    "grupo": "ventanas",
                    "reino": reino,
                    "ventana": ventana.replace("w", "") + " kb",
                    "agregacion": "MIL" if agreg == "logits" else "pooled",
                    "lector": "v2",
                    "train": None,
                    "bal_acc": m["balanced_accuracy"],
                    "f1": m["f1"],
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "pr_auc": m["pr_auc"],
                    "n": m["n"],
                    "prev": m["prevalence"],
                    "epoch": t.get("epoch"),
                    "threshold": t.get("threshold"),
                })
            except Exception:
                continue
    CACHE["ventanas"] = runs
    return runs


def cargar_ntv3():
    if "ntv3" in CACHE:
        return CACHE["ntv3"]
    d = _json(RES_NTV3) or {}
    runs = []
    for r in d.get("runs", []):
        try:
            r["grupo"] = "ntv3"
            r.setdefault("threshold", None)
            r.setdefault("epoch", None)
            runs.append(r)
        except Exception:
            continue
    CACHE["ntv3"] = runs
    return runs


def epocas(reino):
    clave = f"epocas_{reino}"
    if clave in CACHE:
        return CACHE[clave]
    f = EPOCAS / f"{reino.lower()}.csv"
    rows = None
    if f.exists():
        try:
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            rows = None
    CACHE[clave] = rows
    return rows


def mejor_epoca(rows):
    if not rows:
        return None
    try:
        m = max(rows, key=lambda r: float(r["val_bal_acc"]))
        return {"epoch": m["epoch"], "val_bal_acc": float(m["val_bal_acc"]),
                "val_f1": float(m["val_f1"]), "val_loss": float(m["val_loss"]),
                "loss": float(m["loss"])}
    except Exception:
        return None


@app.route("/")
def index():
    runs = cargar_ventanas() + cargar_ntv3()
    runs_ok = [r for r in runs if r.get("bal_acc") is not None and r.get("pr_auc") is not None]
    vistos, unicos = set(), []
    for r in runs_ok:
        k = (r["reino"], r["lector"], r["ventana"], r["agregacion"],
             round(r["bal_acc"], 4), round(r["pr_auc"], 4), r["n"])
        if k in vistos:
            continue
        vistos.add(k)
        unicos.append(r)
    runs_ok = unicos
    mejor = max(runs_ok, key=lambda r: r["pr_auc"]) if runs_ok else None
    mayor = max(runs_ok, key=lambda r: r["n"]) if runs_ok else None
    resumen = {}
    for r in runs_ok:
        resumen.setdefault(r["reino"], []).append(r)
    return render_template("index.html", runs=runs_ok, mejor=mejor, mayor=mayor,
                           resumen=resumen, n=len(runs_ok))


@app.route("/corrida/<rid>")
def corrida(rid):
    runs = {r["id"]: r for r in cargar_ventanas() + cargar_ntv3()}
    if rid not in runs:
        abort(404)
    r = runs[rid]
    rows = epocas(r["reino"]) if r["reino"] in ("Hongos", "Plantas", "Animales") else None
    return render_template("corrida.html", r=r, epocas=rows, mejor=mejor_epoca(rows))


@app.route("/epocas/<reino>")
def ver_epocas(reino):
    if reino not in ("Hongos", "Plantas", "Animales"):
        abort(404)
    rows = epocas(reino)
    if rows is None:
        abort(404)
    return render_template("epocas.html", reino=reino, epocas=rows, mejor=mejor_epoca(rows))


@app.route("/comparar")
def comparar():
    runs = [r for r in cargar_ntv3() if r.get("lector") in ("v2", "NTv3")]
    pares = {}
    for r in runs:
        pares.setdefault((r["reino"], r["ventana"], r["agregacion"]), {})[r["lector"]] = r
    filas = []
    for k, v in sorted(pares.items()):
        if "v2" in v and "NTv3" in v:
            a, b = v["v2"], v["NTv3"]
            try:
                filas.append({"reino": k[0], "ventana": k[1], "v2": a, "ntv3": b,
                              "d_bal": b["bal_acc"] - a["bal_acc"],
                              "d_pr": b["pr_auc"] - a["pr_auc"]})
            except Exception:
                continue
    return render_template("comparar.html", filas=filas)


@app.route("/salud")
def salud():
    return {"ok": True,
            "ventanas": len(cargar_ventanas()),
            "ntv3": len(cargar_ntv3())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"visor en http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
