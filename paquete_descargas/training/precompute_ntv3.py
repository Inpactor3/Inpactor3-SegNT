"""Precalcula embeddings de NTv3 (tokenizacion por base) para los bundles del proyecto.

Reemplaza a precompute_embeddings.py cuando el lector es Nucleotide Transformer v3.
Cada bloque de 6 kb se convierte en un vector de D dimensiones (768 en el 100M).

Uso:
  python precompute_ntv3.py --seq_dir <bundle>/<Reino> --out_dir <salida> \
      --splits val,test train --repo InstaDeepAI/NTv3_100M_pre --batch 4
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

CHUNK = 6000


def limpiar(texto):
    """NTv3 solo entiende A C G T N (mayusculas)."""
    return re.sub(r"[^ACGTN]", "N", texto.upper())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--repo", default="InstaDeepAI/NTv3_100M_pre")
    ap.add_argument("--batch", type=int, default=4, help="ventanas por paso (4 bloques cada una)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"cargando {args.repo} en {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
    model = (AutoModelForMaskedLM.from_pretrained(args.repo, trust_remote_code=True)
             .eval().to(args.device))

    for split in args.splits.split(","):
        files = sorted(glob.glob(str(args.seq_dir / split / "*.npz")))
        if not files:
            print(f"{split}: sin npz", flush=True)
            continue
        outdir = args.out_dir / split
        outdir.mkdir(parents=True, exist_ok=True)
        for f in files:
            tag = os.path.basename(f)[:-4]
            dst = outdir / f"{tag}_emb.npy"
            if dst.exists():
                print("SKIP", split, tag, flush=True)
                continue
            z = np.load(f, allow_pickle=True)
            seqs, labels = z["seq"], z["label"]
            N, C, _ = seqs.shape
            if N == 0:
                print(f"SKIP vacio {split} {tag}", flush=True)
                continue
            trozos = []
            for i in range(0, N, args.batch):
                bloque = seqs[i:i + args.batch]              # (b, C, 6000)
                plano = bloque.reshape(-1, CHUNK)            # (b*C, 6000)
                textos = [limpiar(bytes(r).decode("ascii", "ignore")) for r in plano]
                enc = tok(textos, add_special_tokens=False, padding=True,
                          pad_to_multiple_of=128, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                h = out.hidden_states[-1]                    # (b*C, L, D)
                emb = h[:, :CHUNK].mean(dim=1)               # media sobre bases reales
                trozos.append(emb.float().cpu().numpy().astype(np.float16)
                              .reshape(bloque.shape[0], C, -1))
                del out, h, emb
                if i % (args.batch * 25) == 0:
                    print(f"  {split} {tag} {i}/{N}", flush=True)
            if not trozos:
                print(f"SKIP sin trozos {split} {tag}", flush=True)
                continue
            emb = np.concatenate(trozos, axis=0)
            np.save(dst, emb)
            np.save(outdir / f"{tag}_labels.npy", labels)
            print(f"OK {split} {tag} {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
