"""Test oro TAIR10 con el cuerpo v2: 24 kb (4 bloques) vs 48 kb (8 bloques)."""
import modal

app = modal.App("inpactor-gold-v2")
SHA = "30f4947"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"
NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "transformers==4.46.3", "numpy", "kaggle", "six")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/evaluate_gold.py -o /code/evaluate_gold.py",
        f"curl -sL {GH}/training/block_layout.py -o /code/block_layout.py",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=3600,
              secrets=[modal.Secret.from_name("kaggle-token")])
def evaluar(chunks: int, mode: str, salida: str):
    import glob
    import json
    import os
    import subprocess
    import sys

    tok = os.environ.get("KAGGLE_API_TOKEN", "")
    if tok:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        p = os.path.expanduser("~/.kaggle/access_token")
        open(p, "w").write(tok.strip())
        os.chmod(p, 0o600)

    os.makedirs("/tmp/gold", exist_ok=True)
    # el dataset tiene la ruta exacta TAIR10.fasta/TAIR10.fasta
    for f in ("TAIR10.fasta/TAIR10.fasta", "tair10_te.bed"):
        base = os.path.basename(f)
        if not glob.glob(f"/tmp/gold/**/{base}", recursive=True):
            r = subprocess.run(["kaggle", "datasets", "download", "-d",
                                "cristianosoriog/inpactor2-gold", "-f", f,
                                "-p", "/tmp/gold", "--quiet"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return f"FALLO descarga {f}: {r.stderr.strip()[-200:]}"
    gen = glob.glob("/tmp/gold/**/TAIR10.fasta", recursive=True)
    bed = glob.glob("/tmp/gold/**/tair10_te.bed", recursive=True)
    if not gen or not bed:
        return "FALLO: no encuentro TAIR10 o el BED"
    # si Kaggle expandio el .gz, el archivo esta dentro de una carpeta
    gen = [g for g in gen if os.path.isfile(g)]
    gen = max(gen, key=os.path.getsize)

    r = subprocess.run([sys.executable, "/code/evaluate_gold.py",
                        "--genome", gen, "--bed", bed[0],
                        "--head", "/emb/Modelos/Plantas_v2/best.pt",
                        "--out", f"/emb/Gold/{salida}", "--reino", "Plantas",
                        "--chunks", str(chunks), "--mode", mode,
                        "--trust_remote_code", "--nt_model", NT,
                        "--batch", "8", "--token_batch", "32"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)
        return f"FALLO rc={r.returncode}"
    vol.commit()
    d = json.load(open(f"/emb/Gold/{salida}/gold_metrics.json"))
    m = d["at_calibrated_threshold"]
    return (f"chunks={chunks} thr={d['threshold']} BA={m['balanced_accuracy']:.4f} "
            f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"PR-AUC={m['pr_auc']:.4f} n={m['n']}")


@app.local_entrypoint()
def main():
    for chunks, mode, salida in ((4, "windowed", "24kb"), (8, "eval_full", "48kb")):
        print(evaluar.remote(chunks, mode, salida), flush=True)
