"""Precompute por bloque (N,C,H) para hongos, en 3 shards paralelos."""
import modal

app = modal.App("inpactor-mil-hongos")
SHA = "8420458"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"
NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar")
    .pip_install("torch", "transformers==4.46.3", "numpy")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/precompute_embeddings.py -o /code/precompute_embeddings.py",
        f"curl -sL {GH}/training/block_layout.py -o /code/block_layout.py",
        f"curl -sL {GH}/colab_bridge/src/model.py -o /code/model.py",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=21600)
def preparar():
    import os
    import subprocess

    if not os.path.isdir("/emb/seq/Hongos/train"):
        os.makedirs("/emb/seq", exist_ok=True)
        subprocess.run(["tar", "xzf", "/emb/seq/hongos_bundle.tar.gz",
                        "-C", "/emb/seq"], check=True)
        vol.commit()
    n = len(os.listdir("/emb/seq/Hongos/train"))
    return f"bundle listo: {n} archivos de entrenamiento"


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=21600)
def shard(i: int, n: int = 3):
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "/code/precompute_embeddings.py",
                        "--seq_dir", "/emb/seq/Hongos", "--out_dir", "/emb/HongosMIL",
                        "--splits", "train,val,test", "--per_block",
                        "--trust_remote_code", "--nt_model", NT,
                        "--batch_size", "8", "--token_batch", "32",
                        "--num_shards", str(n), "--shard_index", str(i)],
                       capture_output=True, text=True)
    print(r.stdout[-2500:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)
        return f"FALLO shard {i}"
    vol.commit()
    return f"ok shard {i}"


@app.local_entrypoint()
def main():
    print(preparar.remote(), flush=True)
    for res in shard.map([0, 1, 2]):
        print(res, flush=True)
