"""Test oro de Drosophila melanogaster con el cuerpo v2 (modelo de animales)."""
import modal

app = modal.App("inpactor-gold-dm")
SHA = "caf5c41"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"
NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
CHROMS = "chr2L,chr2R,chr3L,chr3R,chr4,chrX"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "transformers==4.46.3", "numpy")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/evaluate_gold.py -o /code/evaluate_gold.py",
        f"curl -sL {GH}/training/block_layout.py -o /code/block_layout.py",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=3600)
def evaluar(chunks: int, mode: str, salida: str):
    import json
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "/code/evaluate_gold.py",
                        "--genome", "/emb/GoldDM/dm6.fa.gz",
                        "--bed", "/emb/GoldDM/dm6_te.bed",
                        "--head", "/emb/Modelos/Animales_v2fix3/best.pt",
                        "--out", f"/emb/GoldDM/{salida}", "--reino", "Animales",
                        "--chroms", CHROMS,
                        "--chunks", str(chunks), "--mode", mode,
                        "--trust_remote_code", "--nt_model", NT,
                        "--batch", "8", "--token_batch", "32"],
                       capture_output=True, text=True)
    print(r.stdout[-2500:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:], flush=True)
        return f"FALLO rc={r.returncode}"
    vol.commit()
    d = json.load(open(f"/emb/GoldDM/{salida}/gold_metrics.json"))
    m = d["at_calibrated_threshold"]
    return (f"Drosophila chunks={chunks} thr={d['threshold']} BA={m['balanced_accuracy']:.4f} "
            f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"PR-AUC={m['pr_auc']:.4f} n={m['n']}")


@app.local_entrypoint()
def main():
    for chunks, mode, salida in ((4, "windowed", "24kb"), (8, "eval_full", "48kb")):
        print(evaluar.remote(chunks, mode, salida), flush=True)
