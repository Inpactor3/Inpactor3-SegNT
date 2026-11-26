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

from flask import (Flask, abort, render_template, request, redirect,
                   url_for, send_file, jsonify, flash)
from werkzeug.utils import secure_filename

from charts import bar_chart, heatmap, line_svg, confusion_table, color_grupo
import jobs as jobmod

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RES_VENTANAS = DOCS / "ventanas"
RES_NTV3 = DOCS / "ntv3_resultados.json"
EPOCAS = DOCS / "ntv3_epocas"
DIAG = ROOT / "diagnostico_animales.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(jobmod.MAX_GB * 1e9)
app.secret_key = "visor-inpactor3"  # solo mensajes flash locales
CACHE = {}
jobmod.iniciar_worker()


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


def _csv(path):
    clave = str(path)
    if clave in CACHE:
        return CACHE[clave]
    rows = None
    if path.exists():
        try:
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            rows = None
    CACHE[clave] = rows
    return rows


def epocas_run(r):
    """Cada corrida usa su propio metrics.csv; NTv3 tiene los suyos por reino."""
    if r.get("grupo") == "ventanas":
        return _csv(RES_VENTANAS / r["id"] / "metrics.csv")
    return _csv(EPOCAS / f"{r['reino'].lower()}.csv")


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


def cargar_diagnostico():
    if "diag" in CACHE:
        return CACHE["diag"]
    rows = []
    if DIAG.exists():
        with open(DIAG) as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append({
                        "especie": r["especie"],
                        "nombre": r["especie"].split("__")[0].replace("_", " "),
                        "grupo": r["grupo"],
                        "split": r["split"],
                        "ventanas": int(r["ventanas"]),
                        "positivas": int(r["positivas"]),
                        "negativas": int(r["negativas"]),
                        "pct_pos": float(r["pct_pos"]),
                        "roto": r.get("roto", "NO") == "SÍ",
                    })
                except Exception:
                    continue
    CACHE["diag"] = rows
    return rows


def config_label(r):
    return f"{r['lector']} · {r['ventana']} · {r['agregacion']}"


def valor_metrica(r, metrica):
    v = r.get(metrica)
    if v is None and metrica == "bal_acc":
        v = r.get("balanced_accuracy")
    return v


def todos_los_runs():
    return cargar_ventanas() + cargar_ntv3()


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
    rows = epocas_run(r)
    conf = None
    try:
        n = float(r["n"]); prev = float(r["prev"])
        rec = float(r["recall"]); prec = float(r["precision"])
        pos = n * prev
        tp = rec * pos
        fn = pos - tp
        fp = tp * (1 - prec) / prec if prec > 0 else None
        if fp is not None:
            tn = n - pos - fp
            if tn >= 0:
                conf = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "tabla": confusion_table(tp, fp, fn, tn)}
    except Exception:
        conf = None
    return render_template("corrida.html", r=r, epocas=rows, mejor=mejor_epoca(rows),
                           confusion=conf)


@app.route("/epocas/<reino>")
def ver_epocas(reino):
    if reino not in ("Hongos", "Plantas", "Animales"):
        abort(404)
    rows = _csv(EPOCAS / f"{reino.lower()}.csv")
    if rows is None:
        abort(404)
    labels = [r["epoch"] for r in rows]
    graficos_ep = [
        line_svg("Val loss", labels, [float(r["val_loss"]) for r in rows], "#9f1239"),
        line_svg("Val F1", labels, [float(r["val_f1"]) for r in rows], "#0f766e"),
        line_svg("Val balanced accuracy", labels,
                 [float(r["val_bal_acc"]) for r in rows], "#4c1d95"),
    ]
    return render_template("epocas.html", reino=reino, epocas=rows,
                           mejor=mejor_epoca(rows), graficos_ep=graficos_ep)


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


@app.route("/graficos")
def graficos():
    metrica = request.args.get("metrica", "pr_auc")
    if metrica not in ("pr_auc", "f1", "bal_acc", "precision", "recall"):
        metrica = "pr_auc"
    runs = [r for r in todos_los_runs() if valor_metrica(r, metrica) is not None]
    vistos, unicos = set(), []
    for r in runs:
        k = (r.get("id"), r.get("reino"), config_label(r))
        if k in vistos:
            continue
        vistos.add(k)
        unicos.append(r)
    ordenados = sorted(unicos, key=lambda r: valor_metrica(r, metrica), reverse=True)
    reino_color = {r: color_grupo(r) for r in sorted({x["reino"] for x in ordenados})}
    items = [{"label": f'{r["reino"]} · {r["lector"]} · {r["ventana"]} · {r["agregacion"]}',
              "value": valor_metrica(r, metrica),
              "color": reino_color.get(r["reino"])} for r in ordenados[:20]]
    barras = bar_chart(items)
    reinos = sorted({r["reino"] for r in unicos})
    cols = sorted({config_label(r) for r in unicos})
    valores = {}
    for r in unicos:
        k = (r["reino"], config_label(r))
        v = valor_metrica(r, metrica)
        if k not in valores or v > valores[k]:
            valores[k] = v
    hm = heatmap(reinos, cols, valores, fmt=".3f",
                 cell_title=lambda a, b, v: f"{a} · {b}: {v:.4f}")
    return render_template("graficos.html", barras=barras, hm=hm, metrica=metrica,
                           n=len(unicos), reinos=reinos, cols=cols)


@app.route("/especies")
def especies():
    grupo = request.args.get("grupo", "")
    split = request.args.get("split", "")
    rows = cargar_diagnostico()
    grupos = sorted({r["grupo"] for r in rows})
    if grupo:
        rows = [r for r in rows if r["grupo"] == grupo]
    if split:
        rows = [r for r in rows if r["split"] == split]
    rows_orden = sorted(rows, key=lambda r: r["ventanas"], reverse=True)
    items = [{"label": r["nombre"], "sub": f'{r["grupo"]} · {r["split"]}',
              "value": r["ventanas"], "color": color_grupo(r["grupo"], grupos)}
             for r in rows_orden[:25]]
    barras = bar_chart(items, value_fmt="{:,.0f}".replace(",", " "))
    matrix = {}
    for r in rows:
        k = (r["grupo"], r["split"])
        matrix[k] = matrix.get(k, 0) + r["ventanas"]
    splits = ["train", "val", "test"]
    hm = heatmap(grupos, splits, matrix, fmt=".0f",
                 cell_title=lambda a, b, v: f"{a} · {b}: {int(v):,} ventanas".replace(",", "."))
    total = sum(r["ventanas"] for r in rows)
    return render_template("especies.html", barras=barras, hm=hm, rows=rows_orden,
                           grupos=grupos, grupo=grupo, split=split,
                           total=total, n=len(rows))


@app.route("/analizar", methods=["GET", "POST"])
def analizar():
    modelos = jobmod.modelos()
    if request.method == "POST":
        archivo = request.files.get("genome")
        if not archivo or not archivo.filename:
            flash("Selecciona un genoma en FASTA.")
            return redirect(url_for("analizar"))
        nombre = secure_filename(archivo.filename)
        if not nombre.lower().endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")):
            flash("El genoma debe ser .fa, .fasta, .fna o .gz.")
            return redirect(url_for("analizar"))
        modelo = request.form.get("modelo", "")
        if modelo not in {m["path"] for m in modelos}:
            flash("Checkpoint no válido.")
            return redirect(url_for("analizar"))
        try:
            ventana = int(request.form.get("ventana", 50000))
            chunks = int(request.form.get("chunks", 4))
            if not (1000 <= ventana <= 1_000_000) or not (1 <= chunks <= 16):
                raise ValueError
            thr = request.form.get("threshold", "").strip()
            thr = float(thr) if thr else None
            if thr is not None and not (0.0 < thr < 1.0):
                raise ValueError
        except ValueError:
            flash("Revisa ventana, chunks o umbral.")
            return redirect(url_for("analizar"))
        backbone = request.form.get("backbone", "auto")
        if backbone not in ("auto", "v2", "ntv3"):
            backbone = "auto"
        try:
            job = jobmod.crear_job(archivo, nombre, modelo, ventana, chunks, thr, backbone)
        except ValueError as e:
            flash(str(e))
            return redirect(url_for("analizar"))
        return redirect(url_for("trabajo", job_id=job["id"]))
    return render_template("analizar.html", modelos=modelos,
                           jobs=jobmod.listar_jobs(), max_gb=jobmod.MAX_GB,
                           deps_ok=jobmod.deps_ok(), python=jobmod.PYTHON,
                           ntv3_disponible=any(m.get("reader") == "NTv3" for m in modelos))


@app.route("/trabajo/<job_id>")
def trabajo(job_id):
    job = jobmod.leer_job(job_id)
    if not job:
        abort(404)
    return render_template("trabajo.html", job=job)


@app.route("/trabajo/<job_id>.json")
def trabajo_json(job_id):
    job = jobmod.leer_job(job_id)
    if not job:
        return jsonify({"error": "no existe"}), 404
    return jsonify(job)


@app.route("/trabajo/<job_id>/<nombre>")
def descargar(job_id, nombre):
    job = jobmod.leer_job(job_id)
    if not job:
        abort(404)
    permitidos = set(job.get("archivos") or []) | {"log.txt"}
    if nombre not in permitidos or nombre.startswith("."):
        abort(404)
    d = jobmod._dir(job_id)
    p = d / nombre
    if not p.exists():
        abort(404)
    return send_file(p, as_attachment=True, download_name=f"{job_id}_{nombre}")


@app.errorhandler(413)
def demasiado_grande(_):
    return (f"El archivo supera el límite de {jobmod.MAX_GB:g} GB. "
            "Sube el genoma comprimido o aumenta INPACTOR3_MAX_GB.", 413)


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
