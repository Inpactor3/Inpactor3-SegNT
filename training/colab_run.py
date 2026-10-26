"""Orquestador de entrenamiento para Colab/Kaggle/GPU.

Crea en Drive (o /kaggle/working):
  Inpactor2_gpu/<reino>/
    emb/      embeddings (4 bloques x 6 kb, max pooling) por especie
    ckpt/     best.pt (incluye umbral calibrado), metrics.csv,
              calibration.json, test_metrics.json
    logs/     gpu_info.txt, precompute.log, train.log

Flujo (por reino):
  1. precalculo de embeddings con max pooling sobre bloques (reanudable por especie).
  2. entrenamiento del MLP con `--balance 1` + `--class_weight` y calibracion
     del umbral en validacion (`--calibrate`).

Uso:
  python colab_run.py --reino Hongos --seq_dir /content/data/Hongos \
      --drive_root /content/drive/MyDrive/Inpactor2_gpu --epochs 100
  python colab_run.py --reino Plantas --seq_dir ... --balance 1 --class_weight --calibrate
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(map(str, cmd))}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return p.wait()


def gpu_info(out):
    import torch
    lines = [f"fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"python: {sys.version.split()[0]}",
             f"torch: {torch.__version__}",
             f"cuda disponible: {torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        lines.append(f"vram: {props.total_memory / 1e9:.1f} GB")
        lines.append(f"capability: {props.major}.{props.minor}")
        lines.append(f"cuda version: {torch.version.cuda}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reino", required=True)
    ap.add_argument("--drive_root", required=True)
    ap.add_argument("--jsonl_dir", default=None)
    ap.add_argument("--genomes_dir", default=None)
    ap.add_argument("--seq_dir", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--token_batch", type=int, default=64)
    ap.add_argument("--chunks", type=int, default=4,
                    help="bloques de 6 kb por ventana (4 por defecto)")
    ap.add_argument("--balance", choices=("none", "global", "species"), default="none",
                    help="1 por defecto para plantas/animales via --balance 1/global")
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    root = Path(args.drive_root) / args.reino
    emb, ckpt, logs = root / "emb", root / "ckpt", root / "logs"
    for d in (emb, ckpt, logs):
        d.mkdir(parents=True, exist_ok=True)

    gpu_info(logs / "gpu_info.txt")

    have_all = all((emb / f"{s}_emb.npy").exists() or any((emb / s).glob("*_emb.npy"))
                   for s in args.splits.split(","))
    if have_all:
        print("embeddings ya existen, se salta el precalculo")
    else:
        cmd = [sys.executable, str(HERE / "precompute_embeddings.py"),
               "--out_dir", str(emb), "--splits", args.splits,
               "--chunks", str(args.chunks), "--token_batch", str(args.token_batch)]
        if args.seq_dir:
            cmd += ["--seq_dir", args.seq_dir]
        else:
            cmd += ["--jsonl_dir", args.jsonl_dir, "--genomes_dir", args.genomes_dir]
        print("precalculando embeddings (max pooling sobre bloques)...")
        rc = sh(cmd, logs / "precompute.log")
        if rc != 0:
            print("ERROR en precalculo; revisa", logs / "precompute.log")
            return rc
        print("embeddings listos")

    cmd = [sys.executable, str(HERE / "train_mlp.py"),
           "--emb_dir", str(emb), "--out_dir", str(ckpt),
           "--epochs", str(args.epochs), "--patience", str(args.patience),
           "--batch_size", str(args.batch_size)]
    if args.balance not in ("none",):
        cmd += ["--balance", "global" if args.balance == "1" else args.balance]
    for flag in ("class_weight", "calibrate", "standardize"):
        if getattr(args, flag):
            cmd.append(f"--{flag}")
    print("entrenando...")
    rc = sh(cmd, logs / "train.log")
    print("terminado con codigo", rc)
    print("resultados en:", root)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
