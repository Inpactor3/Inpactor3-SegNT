"""Metricas por grupo taxonomico del test de animales (modelo v2)."""
import modal

app = modal.App("inpactor-grupos-animales")
SHA = "b9e10b5"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "numpy")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/eval_por_grupo.py -o /code/eval_por_grupo.py",
        f"curl -sL {GH}/training/train_mlp.py -o /code/train_mlp.py",
        f"curl -sL {GH}/docs/animales_taxonomia.csv -o /code/animales_taxonomia.csv",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=image, volumes={"/emb": vol}, timeout=2400, cpu=4)
def grupos():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "/code/eval_por_grupo.py",
                        "--emb_dir", "/emb/AnimalesNuevo", "--split", "test",
                        "--head", "/emb/Modelos/Animales_v2fix3/best.pt",
                        "--group_csv", "/code/animales_taxonomia.csv",
                        "--out", "/tmp/por_grupo.csv"],
                       capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)
    return r.stdout


@app.local_entrypoint()
def main():
    print(grupos.remote())
