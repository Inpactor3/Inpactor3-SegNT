"""Diagnostico de hebra: test oro normal vs reverse complement."""
import modal

app = modal.App("inpactor-gold-revcomp")
SHA = "b0330a0"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"
NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

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

CASOS = {
    "arabidopsis": {
        "genome": "/emb/GoldRC/TAIR10.fasta.gz",
        "bed": "/emb/GoldRC/tair10_te.bed",
        "head": "/emb/Modelos/Plantas_v2/best.pt",
        "reino": "Plantas", "chroms": None,
    },
    "drosophila": {
        "genome": "/emb/GoldDM/dm6.fa.gz",
        "bed": "/emb/GoldDM/dm6_te.bed",
        "head": "/emb/Modelos/Animales_v2fix3/best.pt",
        "reino": "Animales", "chroms": "chr2L,chr2R,chr3L,chr3R,chr4,chrX",
    },
}
CHROMS_REF = {"arabidopsis": None, "drosophila": "chr2L,chr2R,chr3L,chr3R,chr4,chrX"}


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=3600)
def evaluar(caso: str, chunks: int, mode: str, rc: bool):
    import json
    import subprocess
    import sys

    c = CASOS[caso]
    salida = f"/emb/GoldRC/{caso}_{'rc' if rc else 'fw'}_{chunks}"
    cmd = [sys.executable, "/code/evaluate_gold.py",
           "--genome", c["genome"], "--bed", c["bed"], "--head", c["head"],
           "--out", salida, "--reino", c["reino"],
           "--chunks", str(chunks), "--mode", mode,
           "--trust_remote_code", "--nt_model", NT,
           "--batch", "8", "--token_batch", "32"]
    if CHROMS_REF[caso]:
        cmd += ["--chroms", CHROMS_REF[caso]]
    if rc:
        cmd.append("--revcomp")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:], flush=True)
        return f"FALLO {caso} chunks={chunks} rc={rc}"
    vol.commit()
    d = json.load(open(f"{salida}/gold_metrics.json"))
    m = d["at_calibrated_threshold"]
    return (f"{caso:12s} {'revcomp' if rc else 'forward':7s} chunks={chunks} "
            f"BA={m['balanced_accuracy']:.4f} F1={m['f1']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} PR-AUC={m['pr_auc']:.4f}")


@app.local_entrypoint()
def main():
    trabajos = [(caso, ch, "windowed" if ch == 4 else "eval_full", rc)
                for caso in ("arabidopsis", "drosophila")
                for ch in (4, 8) for rc in (False, True)]
    for res in evaluar.map(*zip(*trabajos)):
        print(res, flush=True)
