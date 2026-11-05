"""Precalcula embeddings con el cuerpo v2 en Modal, una tarea por especie.

Cada contenedor baja los npz de UNA especie desde Kaggle, (si es train) aplica el
submuestreo 1:1 con la misma semilla por especie que en el servidor, y calcula sus
embeddings. Los resultados se guardan en un Volume de Modal.

Uso:
  modal secret create kaggle-token KAGGLE_API_TOKEN=KGAT_...
  modal run training/precompute_modal.py --reino Animales

Requisitos:
  modal token new          (una vez, abre el navegador)
"""
import modal

app = modal.App("inpactor-v2-emb")

GH_SHA = "30f4947"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{GH_SHA}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "transformers==4.46.3", "numpy", "kaggle")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/precompute_embeddings.py -o /code/precompute_embeddings.py",
        f"curl -sL {GH}/training/subsample_train.py -o /code/subsample_train.py",
        f"curl -sL {GH}/training/block_layout.py -o /code/block_layout.py",
        f"curl -sL {GH}/colab_bridge/src/model.py -o /code/model.py",
        f"curl -sL {GH}/colab_bridge/src/dataset.py -o /code/dataset.py",
    )
)

vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)

NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={"/emb": vol},
    secrets=[modal.Secret.from_name("kaggle-token")],
    retries=2,
)
def procesar(reino: str, split: str, archivo: str,
             batch: int = 16, token_batch: int = 64):
    import glob
    import os
    import shutil
    import subprocess
    import sys

    tok = os.environ.get("KAGGLE_API_TOKEN", "")
    if tok:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        ruta_tok = os.path.expanduser("~/.kaggle/access_token")
        with open(ruta_tok, "w") as f:
            f.write(tok.strip())
        os.chmod(ruta_tok, 0o600)

    ds = f"cristianosoriog/inpactor2-{reino.lower()}"
    tag = archivo[:-4] if archivo.endswith(".npz") else archivo
    ruta_kaggle = f"bundle4_{reino}/{reino}/{split}/{archivo}"

    tmp = "/tmp/trabajo"
    shutil.rmtree(tmp, ignore_errors=True)
    seq = f"{tmp}/seq/{split}"
    os.makedirs(seq, exist_ok=True)

    r = subprocess.run(["kaggle", "datasets", "download", "-d", ds,
                        "-f", ruta_kaggle, "-p", f"{tmp}/dl", "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"FALLO descarga {tag}: {r.stderr.strip()[:150]}"
    hallados = glob.glob(f"{tmp}/dl/**/*.npz", recursive=True)
    if not hallados:
        return f"FALLO sin npz {tag}"
    shutil.copy2(hallados[0], os.path.join(seq, os.path.basename(hallados[0])))

    entrada = f"{tmp}/seq"
    if split == "train":
        sub = f"{tmp}/sub"
        r = subprocess.run([sys.executable, "/code/subsample_train.py",
                            "--seq_dir", entrada, "--out_dir", sub, "--ratio", "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return f"FALLO submuestreo {tag}: {r.stderr.strip()[:150]}"
        entrada = sub

    r = subprocess.run([sys.executable, "/code/precompute_embeddings.py",
                        "--seq_dir", entrada, "--out_dir", f"/emb/{reino}",
                        "--splits", split, "--trust_remote_code",
                        "--nt_model", NT, "--max_tokens", "1024",
                        "--batch_size", str(batch), "--token_batch", str(token_batch)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"FALLO precalculo {tag}: {r.stderr.strip()[-200:]}"

    vol.commit()
    import numpy as np
    n = int(np.load(f"/emb/{reino}/{split}/{tag}_labels.npy").shape[0])
    return f"OK {split}/{tag}: {n} ventanas"


@app.local_entrypoint()
def main(reino: str = "Animales", batch: int = 16, token_batch: int = 64):
    import json
    import subprocess

    ds = f"cristianosoriog/inpactor2-{reino.lower()}"
    out = subprocess.run(["kaggle", "datasets", "files", "-d", ds,
                          "--page-size", "200", "--format", "json"],
                         capture_output=True, text=True, check=True).stdout
    tareas = []
    for it in json.loads(out):
        nombre = it["name"]
        if not nombre.endswith(".npz"):
            continue
        partes = nombre.split("/")
        if len(partes) != 4:
            continue
        tareas.append((reino, partes[2], partes[3]))

    print(f"{reino}: {len(tareas)} tareas")
    ok = fallos = 0
    n = len(tareas)
    for res in procesar.map([t[0] for t in tareas], [t[1] for t in tareas],
                            [t[2] for t in tareas],
                            [batch] * n, [token_batch] * n):
        print(res)
        if res.startswith("OK"):
            ok += 1
        else:
            fallos += 1
    print(f"\n{reino}: {ok} ok · {fallos} fallos")
