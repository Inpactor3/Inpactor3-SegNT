"""Entrena el cabezal MIL por bloque de hongos sobre la cache HongosMIL."""
import modal

app = modal.App("inpactor-train-mil")
SHA = "63bac7d"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "numpy")
    .run_commands(f"mkdir -p /code",
                  f"curl -sL {GH}/training/train_mil.py -o /code/train_mil.py")
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=10800,
              memory=16384)
def entrenar(epochs: int = 15, lr: float = 1e-3, dropout: float = 0.2):
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "/code/train_mil.py",
                        "--emb_dir", "/emb/HongosMIL",
                        "--out_dir", "/emb/Modelos/HongosMIL",
                        "--epochs", str(epochs), "--lr", str(lr),
                        "--dropout", str(dropout),
                        "--class_weight", "--balance", "global"],
                       capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], flush=True)
        return "FALLO entrenamiento"
    vol.commit()
    return "ok"


@app.local_entrypoint()
def main(epochs: int = 15):
    print(entrenar.remote(epochs), flush=True)
