"""Completa el val/test de Animales v2: recalcula completas las 11 especies que
estaban en el train viejo (sus embeddings actuales estan submuestreados)."""
import modal

app = modal.App("inpactor-fix-animales2")
SHA = "30f4947"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"
NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
FALTA = {"Cariama_cristata__GCF_000690535.1": "test", "Chrysoperla_carnea__GCF_905475395.1": "test", "Ellobius_talpinus__GCA_001685095.1": "test", "Gigantopelta_aegis__GCF_016097555.1": "test", "Acanthopagrus_latus__GCF_904848185.1": "test", "Gopherus_agassizii__GCA_002896415.1": "test", "Charadrius_alexandrinus__GCA_053477275.1": "val", "Acropora_hyacinthus__GCA_964291705.1": "val", "Capromys_pilorides__GCA_004027915.1": "val", "Amphilophus_citrinellus__GCA_013435755.1": "val", "Emydura_subglobosa__GCA_054346155.1": "val"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "transformers==4.46.3", "numpy", "kaggle", "six")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/precompute_embeddings.py -o /code/precompute_embeddings.py",
        f"curl -sL {GH}/training/train_mlp.py -o /code/train_mlp.py",
        f"curl -sL {GH}/training/block_layout.py -o /code/block_layout.py",
        f"curl -sL {GH}/colab_bridge/src/model.py -o /code/model.py",
        f"curl -sL {GH}/colab_bridge/src/dataset.py -o /code/dataset.py",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=4 * 3600,
              secrets=[modal.Secret.from_name("kaggle-token")], retries=2)
def completar(sp: str, split: str):
    import glob
    import os
    import shutil
    import subprocess
    import sys

    tok = os.environ.get("KAGGLE_API_TOKEN", "")
    if tok:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        p = os.path.expanduser("~/.kaggle/access_token")
        open(p, "w").write(tok.strip())
        os.chmod(p, 0o600)

    tmp = "/tmp/c"
    shutil.rmtree(tmp, ignore_errors=True)
    seq = f"{tmp}/seq/{split}"
    os.makedirs(seq, exist_ok=True)
    ruta = f"bundle4_Animales/Animales/train/{sp}.npz"
    r = subprocess.run(["kaggle", "datasets", "download",
                        "-d", "cristianosoriog/inpactor2-animales",
                        "-f", ruta, "-p", f"{tmp}/dl", "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"FALLO descarga {sp}: {r.stderr.strip()[-200:]}"
    hall = glob.glob(f"{tmp}/dl/**/*.npz", recursive=True)
    if not hall:
        return f"FALLO sin npz {sp}"
    shutil.copy2(hall[0], os.path.join(seq, f"{sp}.npz"))

    import numpy as np
    n_real = int(len(np.load(hall[0], allow_pickle=True)["label"]))
    ya = f"/emb/AnimalesNuevo/{split}/{sp}_labels.npy"
    if os.path.exists(ya) and int(len(np.load(ya))) == n_real:
        return f"SKIP {split}/{sp}: ya completo ({n_real})"

    # el precalculo se salta la especie si ya hay embeddings: borrarlos primero
    import glob as _g
    for viejo in _g.glob(f"/emb/AnimalesNuevo/{split}/{sp}_*"):
        os.remove(viejo)

    r = subprocess.run([sys.executable, "/code/precompute_embeddings.py",
                        "--seq_dir", f"{tmp}/seq", "--out_dir", "/emb/AnimalesNuevo",
                        "--splits", split, "--trust_remote_code", "--nt_model", NT,
                        "--max_tokens", "1024", "--batch_size", "8",
                        "--token_batch", "32"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"FALLO precalculo {sp}: {r.stderr.strip()[-200:]}"
    vol.commit()
    n = int(np.load(ya).shape[0])
    return f"OK {split}/{sp}: {n} ventanas (real {n_real})"


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=3600)
def entrenar():
    import json
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "/code/train_mlp.py",
                        "--emb_dir", "/emb/AnimalesNuevo",
                        "--out_dir", "/emb/Modelos/Animales_v2fix2",
                        "--epochs", "100", "--patience", "10", "--batch_size", "4096",
                        "--balance", "global", "--class_weight", "--calibrate",
                        "--select", "balanced_accuracy"], capture_output=True, text=True)
    print(r.stdout[-5000:], flush=True)
    if r.returncode != 0:
        print("STDERR", r.stderr[-2000:], flush=True)
        return "FALLO entrenamiento"
    vol.commit()
    return json.dumps(json.load(open("/emb/Modelos/Animales_v2fix2/test_metrics.json")), indent=2)


@app.local_entrypoint()
def main(solo: str = ""):
    if solo:
        print("RESULTADO:", completar.remote(solo, FALTA[solo]))
        return
    for res in completar.map(list(FALTA.keys()), list(FALTA.values())):
        print(res, flush=True)
    print(entrenar.remote())
