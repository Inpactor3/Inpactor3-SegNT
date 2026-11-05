"""Entrena el cabezal v2 sobre los embeddings guardados en el Volume de Modal.

Uso:
  modal run training/entrenar_modal.py --reino Animales --salida Modelos/Animales_v2
"""
import modal

app = modal.App("inpactor-entrenar")

SHA = "792f65e"
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "numpy")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}/training/train_mlp.py -o /code/train_mlp.py",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=IMAGE, gpu="A10G", volumes={"/emb": vol}, timeout=60 * 90)
def entrenar(reino: str, salida: str, epochs: int = 100, patience: int = 10,
             standardize: bool = False):
    import json
    import subprocess
    import sys

    r = subprocess.run([
        sys.executable, "/code/train_mlp.py",
        "--emb_dir", f"/emb/{reino}", "--out_dir", f"/emb/{salida}",
        "--epochs", str(epochs), "--patience", str(patience),
        "--batch_size", "4096", "--balance", "global",
        "--class_weight", "--calibrate", "--select", "balanced_accuracy",
    ] + (["--standardize"] if standardize else []), capture_output=True, text=True)
    print(r.stdout[-6000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], flush=True)
        return f"FALLO rc={r.returncode}"
    vol.commit()
    try:
        with open(f"/emb/{salida}/test_metrics.json") as f:
            return json.dumps(json.load(f), indent=2)
    except FileNotFoundError:
        return "sin test_metrics"


@app.local_entrypoint()
def main(reino: str = "Animales", salida: str = "Modelos/Animales_v2",
         standardize: bool = False):
    print(entrenar.remote(reino, salida, 100, 10, standardize))
