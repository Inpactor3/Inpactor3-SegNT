"""Cola de trabajos del visor: sube un genoma y corre inpactor3.detect.

Un solo worker para no competir por CPU/GPU. El estado se guarda en
trabajo/visor_jobs/<id>/status.json para sobrevivir recargas del servidor.
"""
import gzip
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "trabajo" / "visor_jobs"
PYTHON = os.environ.get("INPACTOR3_PYTHON", sys.executable)
MAX_GB = float(os.environ.get("INPACTOR3_MAX_GB", "6"))

_lock = threading.Lock()
_cola = queue.Queue()
_jobs = {}
_worker = None
_deps = None


def deps_ok():
    global _deps
    if _deps is None:
        try:
            r = subprocess.run([PYTHON, "-c", "import torch, transformers"],
                               capture_output=True, timeout=40)
            _deps = (r.returncode == 0)
        except Exception:
            _deps = False
    return _deps


def _dir(job_id):
    return JOBS / job_id


def _status_path(job_id):
    return _dir(job_id) / "status.json"


def guardar(job):
    with _lock:
        _jobs[job["id"]] = job
        d = _dir(job["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = _status_path(job["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        tmp.replace(_status_path(job["id"]))


def leer_job(job_id):
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    p = _status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def listar_jobs():
    out = {}
    if JOBS.exists():
        for d in JOBS.iterdir():
            p = d / "status.json"
            if not p.exists():
                continue
            try:
                j = json.loads(p.read_text())
                out[j["id"]] = j
            except Exception:
                continue
    with _lock:
        out.update(_jobs)
    return sorted(out.values(), key=lambda j: j.get("creado", ""), reverse=True)


def modelos():
    """Checkpoints compatibles con inpactor3.detect.

    v2: docs/ventanas/<Reino>_w<ventana>_<agregacion>/best.pt
    NTv3: models/ntv3/**/*.pt (se copian ahi; no se versionan)
    """
    out = []
    for p in sorted((ROOT / "docs" / "ventanas").glob("*/best.pt")):
        nombre = p.parent.name
        m = re.match(r"(.+?)_w(\d+)_(.+)", nombre)
        if m:
            reino, ventana, agreg = m.group(1), m.group(2), m.group(3)
            out.append({"path": str(p),
                        "label": f"v2 · {reino} · {ventana} kb · {agreg}",
                        "reino": reino, "ventana": ventana, "agregacion": agreg,
                        "reader": "v2", "backbone": "v2"})
    for p in sorted((ROOT / "models" / "ntv3").rglob("*.pt")):
        partes = [x for x in p.relative_to(ROOT / "models" / "ntv3").parts if x != p.name]
        reino = p.stem.replace("_best", "").replace("-best", "")
        etiqueta = " · ".join(partes + [reino]) if partes else reino
        out.append({"path": str(p), "label": f"NTv3 · {etiqueta}",
                    "reino": reino, "ventana": None, "agregacion": "",
                    "reader": "NTv3", "backbone": "ntv3"})
    return out


def _progreso(linea, job):
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*$", linea)
    if m:
        hechas, total = int(m.group(1)), int(m.group(2))
        if total:
            job["hechas"] = hechas
            job["total"] = total
            job["progreso"] = max(1, min(99, int(hechas * 100 / total)))
        return True
    return False


def _worker_loop():
    while True:
        job_id = _cola.get()
        job = leer_job(job_id)
        if not job:
            _cola.task_done()
            continue
        d = _dir(job_id)
        genoma = d / job.get("genoma", "genome.fa")
        log = d / "log.txt"
        bed = d / "out.bed"
        gff = d / "out.gff"
        job.update({"estado": "ejecutando", "progreso": 1,
                    "inicio": time.strftime("%Y-%m-%dT%H:%M:%S")})
        guardar(job)
        cmd = [PYTHON, "-m", "inpactor3.detect",
               "--genome", str(genoma), "--head", job["modelo"],
               "--out", str(bed), "--gff", str(gff),
               "--window", str(job["ventana"]), "--chunks", str(job["chunks"]),
               "--batch", "2", "--allow-empty"]
        if job.get("threshold") not in (None, ""):
            cmd += ["--threshold", str(job["threshold"])]
        if job.get("backbone") and job["backbone"] != "auto":
            cmd += ["--backbone", job["backbone"]]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            with open(log, "w") as fh:
                fh.write("$ " + " ".join(cmd) + "\n\n")
                fh.flush()
                p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env)
                for linea in p.stdout:
                    fh.write(linea)
                    fh.flush()
                    if _progreso(linea, job):
                        guardar(job)
                p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"detect termino con codigo {p.returncode}")
            job.update({"estado": "listo", "progreso": 100,
                        "fin": time.strftime("%Y-%m-%dT%H:%M:%S")})
            # Contar regiones del BED
            job["regiones"] = max(0, sum(1 for _ in open(bed))) if bed.exists() else 0
            job["archivos"] = [n for n in ("out.bed", "out.gff", "log.txt")
                               if (d / n).exists()]
        except Exception as e:
            job.update({"estado": "error", "progreso": 100,
                        "error": str(e)[:500],
                        "fin": time.strftime("%Y-%m-%dT%H:%M:%S")})
        guardar(job)
        _cola.task_done()


def iniciar_worker():
    global _worker
    if _worker and _worker.is_alive():
        return
    JOBS.mkdir(parents=True, exist_ok=True)
    # Los trabajos que estaban corriendo cuando se cayo el servidor se marcan.
    for j in listar_jobs():
        if j.get("estado") in ("en_cola", "ejecutando"):
            j["estado"] = "interrumpido"
            j["error"] = "El servidor se reinicio antes de terminar."
            guardar(j)
    _worker = threading.Thread(target=_worker_loop, daemon=True)
    _worker.start()


def crear_job(archivo, nombre_original, modelo, ventana, chunks, threshold,
              backbone):
    job_id = uuid.uuid4().hex[:12]
    d = _dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    sufijo = ".gz" if str(nombre_original).lower().endswith(".gz") else ""
    nombre_disco = "genome.fa" + sufijo
    destino = d / nombre_disco
    archivo.save(destino)
    if not destino.exists() or destino.stat().st_size == 0:
        raise ValueError("el archivo llego vacio")
    # Validar FASTA (primera linea), admitiendo gzip
    try:
        op = gzip.open if destino.suffix == ".gz" else open
        with op(destino, "rt", errors="ignore") as fh:
            primera = fh.readline().strip()
        if not primera.startswith(">"):
            raise ValueError("el archivo no parece FASTA: la primera linea no empieza por '>'")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"no pude leer el FASTA: {e}")
    info = next((m for m in modelos() if m["path"] == modelo), {})
    if backbone == "auto":
        backbone = info.get("backbone", "v2")
    job = {
        "id": job_id,
        "nombre": nombre_original,
        "estado": "en_cola",
        "progreso": 0,
        "creado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modelo": modelo,
        "modelo_label": info.get("label", modelo),
        "reader": info.get("reader", ""),
        "ventana": int(ventana),
        "chunks": int(chunks),
        "threshold": threshold,
        "backbone": backbone,
        "tamano_mb": round(destino.stat().st_size / 1e6, 1),
        "genoma": nombre_disco,
        "archivos": [],
    }
    guardar(job)
    _cola.put(job_id)
    return job


def borrar_job(job_id):
    with _lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(_dir(job_id), ignore_errors=True)
