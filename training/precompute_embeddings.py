"""Precalcula embeddings del backbone congelado NT-500M por ventana con max pooling.

Cada ventana aporta `chunks` bloques (4 por defecto => 24 kb; 8 en modo eval => 48 kb).
Los bloques se apilan en la dimension de batch y se hace **max pooling** sobre los
bloques: `emb = max_b(mean_tokens(NT(block_b)))`. Asi el cabezal ve, por ventana,
la region mas informativa y su tamano de entrada no cambia con el numero de bloques.

Salida por split (una por especie si se usa --seq_dir):
  <split>/<tag>_emb.npy    (N, 1280) float16  (max pooling sobre bloques)
  <split>/<tag>_labels.npy (N,) int8
  <split>/<tag>_meta.jsonl (window_id, species, split, negative_type)
  <split>/<tag>_pool.npy   (N, chunks) uint8 · fraccion de N por bloque (diagnostico)
  <split>/<tag>_offsets.npy (chunks,) int64 · offsets usados

Uso:
  python precompute_embeddings.py --seq_dir <seq_bundle> --out_dir <emb> --batch_size 8
  python precompute_embeddings.py --jsonl_dir <consolidated/R> --genomes_dir <genomes/R> \
      --out_dir <emb> --chunks 4

  # paso de evaluacion 48 kb (8 bloques), una sola vez en la A100:
  python precompute_embeddings.py --seq_dir <seq_bundle_eval8> --out_dir <emb48> \
      --splits val,test --batch_size 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent.parent / "colab_bridge" / "src"
sys.path.insert(0, str(SRC))

from model import Inpactor2Detect, load_tokenizer  # noqa: E402


def block_offsets(n_blocks, window_size, chunk_bp):
    span = window_size - chunk_bp
    return np.array([round(i * span / max(n_blocks - 1, 1)) for i in range(n_blocks)],
                    dtype=np.int64)


def embed_blocks(tok, backbone, seqs, chunk_bp, max_tokens, device, batch_size):
    """seqs: (B, C, chunk_bp) uint8 -> (embs (B,H) float32, n_frac (B,C) float32)."""
    arr = np.asarray(seqs)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    B, C, _ = arr.shape
    flat = arr.reshape(B * C, -1)
    texts = [x.tobytes().decode("ascii", "ignore") for x in flat]

    hidden = backbone.config.hidden_size
    out = torch.empty((B * C, hidden), dtype=torch.float32)
    n_frac = np.zeros((B, C), dtype=np.float32)
    if C > 0:
        n_frac = (flat == ord("N")).mean(axis=1).reshape(B, C)

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding="max_length",
                  max_length=max_tokens)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                             enabled=(device == "cuda")):
            last = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (last.float() * m).sum(1) / m.sum(1).clamp(min=1)
        out[i:i + len(chunk)] = pooled.cpu()

    per_block = out.reshape(B, C, hidden)
    emb = per_block.max(dim=1).values
    return emb.numpy(), n_frac


def run_npz_dir(args, tok, backbone, hidden, device):
    for split in args.splits.split(","):
        sdir = args.out_dir / split
        sdir.mkdir(parents=True, exist_ok=True)
        files = sorted((args.seq_dir / split).glob("*.npz"))
        if args.num_shards > 1:
            files = files[args.shard_index::args.num_shards]
        if not files:
            print(f"{split}: sin npz, se omite", flush=True)
            continue
        print(f"{split}: {len(files)} archivos · max pooling sobre bloques", flush=True)
        for fp in files:
            out_emb = sdir / f"{fp.stem}_emb.npy"
            out_lab = sdir / f"{fp.stem}_labels.npy"
            out_meta = sdir / f"{fp.stem}_meta.jsonl"
            out_pool = sdir / f"{fp.stem}_pool.npy"
            out_off = sdir / f"{fp.stem}_offsets.npy"
            if out_emb.exists() and out_lab.exists() and out_pool.exists():
                print(f"SKIP {split} {fp.stem}", flush=True)
                continue
            z = np.load(fp, allow_pickle=True)
            seqs = z["seq"]
            labs = z["label"]
            ids = z["window_id"]
            chunk_bp = int(z["chunk_bp"]) if "chunk_bp" in z else args.chunk_bp
            offs = (z["offsets"] if "offsets" in z
                    else block_offsets(seqs.shape[1], args.window_size, chunk_bp))
            if len(seqs) == 0:
                np.save(out_emb, np.zeros((0, hidden), dtype=np.float16))
                np.save(out_lab, np.zeros((0,), dtype=np.int8))
                np.save(out_pool, np.zeros((0, seqs.shape[1]), dtype=np.float32))
                np.save(out_off, offs)
                open(out_meta, "w").close()
                print(f"SKIP empty {split} {fp.stem}", flush=True)
                continue
            if args.max_windows and len(seqs) > args.max_windows:
                seqs = seqs[:args.max_windows]
                labs = labs[:args.max_windows]
                ids = ids[:args.max_windows]

            embs, pools = [], []
            for i in range(0, len(seqs), args.batch_size):
                e, p = embed_blocks(tok, backbone, seqs[i:i + args.batch_size], chunk_bp,
                                    args.max_tokens, device, args.token_batch)
                embs.append(e)
                pools.append(p)
                if i % (args.batch_size * 20) == 0:
                    print(f"  {split} {fp.stem} {i}/{len(seqs)}", flush=True)
            emb = np.concatenate(embs, axis=0).astype(np.float16)
            pool = np.concatenate(pools, axis=0).astype(np.float16)
            np.save(out_emb, emb)
            np.save(out_lab, labs.astype(np.int8))
            np.save(out_pool, pool)
            np.save(out_off, offs)
            meta = [{"window_id": str(ids[j]), "species": fp.stem, "split": split,
                     "negative_type": None} for j in range(len(seqs))]
            with open(out_meta, "w") as f:
                for r in meta:
                    f.write(json.dumps(r) + "\n")
            print(f"OK {split} {fp.stem}: {emb.shape} (chunks={seqs.shape[1]})", flush=True)


def run_jsonl_dir(args, tok, backbone, hidden, device):
    from dataset import WindowDataset  # noqa: F401 (validacion de entorno)
    for split in args.splits.split(","):
        files = sorted((args.jsonl_dir / split).glob("*.jsonl"))
        if not files:
            print(f"{split}: sin archivos, se omite", flush=True)
            continue
        meta, embs, labels, pools = [], [], [], []
        offsets = block_offsets(args.chunks, args.window_size, args.chunk_bp)
        for jf in files:
            rows = [json.loads(line) for line in open(jf)]
            if args.limit:
                rows = rows[:args.limit]
            by_chrom = {}
            for i, r in enumerate(rows):
                by_chrom.setdefault(r["chrom"], []).append(i)
            seqarr = np.full((len(rows), args.chunks, args.chunk_bp), ord("N"), dtype=np.uint8)
            import gzip

            def stream():

                matches = list(Path(args.genomes_dir).rglob(f"{jf.stem}*.fna.gz"))
                if not matches:
                    raise SystemExit(f"sin genoma para {jf.stem}")
                cur, buf = None, []
                with gzip.open(matches[0], "rt", errors="ignore") as f:
                    for line in f:
                        if line.startswith(">"):
                            if cur in by_chrom:
                                yield cur, "".join(buf)
                            cur = line[1:].split()[0]
                            buf = []
                        else:
                            buf.append(line.strip())
                if cur in by_chrom:
                    yield cur, "".join(buf)

            for chrom, seq in stream():
                for i in by_chrom[chrom]:
                    st = rows[i]["start"]
                    for c, off in enumerate(offsets):
                        piece = seq[st + int(off): st + int(off) + args.chunk_bp].upper()
                        if len(piece) >= 100:
                            seqarr[i, c] = np.frombuffer(
                                piece[:args.chunk_bp].ljust(args.chunk_bp, "N").encode(),
                                dtype=np.uint8)
                del seq
            for i in range(0, len(rows), args.batch_size):
                e, p = embed_blocks(tok, backbone, seqarr[i:i + args.batch_size], args.chunk_bp,
                                    args.max_tokens, device, args.token_batch)
                embs.append(e)
                pools.append(p)
            labels.extend(int(r.get("label", 0)) for r in rows)
            meta.extend({"window_id": r.get("window_id"), "species": r.get("species"),
                         "split": split, "negative_type": r.get("negative_type")}
                        for r in rows)
            print(f"OK {split} {jf.stem}: {len(rows)} ventanas", flush=True)

        if not embs:
            continue
        emb = np.concatenate(embs).astype(np.float16)
        np.save(args.out_dir / f"{split}_emb.npy", emb)
        np.save(args.out_dir / f"{split}_labels.npy", np.asarray(labels, dtype=np.int8))
        np.save(args.out_dir / f"{split}_pool.npy", np.concatenate(pools).astype(np.float16))
        np.save(args.out_dir / f"{split}_offsets.npy", offsets)
        with open(args.out_dir / f"{split}_meta.jsonl", "w") as f:
            for r in meta:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {emb.shape} guardado", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_dir", type=Path, default=None)
    ap.add_argument("--genomes_dir", type=Path, default=None)
    ap.add_argument("--seq_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="ventanas por paso (cada una multipla el trabajo por chunks)")
    ap.add_argument("--token_batch", type=int, default=64,
                    help="bloques de texto por forward del backbone")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk_bp", type=int, default=6000)
    ap.add_argument("--max_tokens", type=int, default=1000)
    ap.add_argument("--window_size", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--nt_model",
                    default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    tok = load_tokenizer(args.nt_model)
    model = Inpactor2Detect(nt_model_name=args.nt_model, freeze_backbone=True).to(device).eval()
    backbone = model.backbone
    hidden = backbone.config.hidden_size

    if args.seq_dir is not None:
        run_npz_dir(args, tok, backbone, hidden, device)
    else:
        run_jsonl_dir(args, tok, backbone, hidden, device)


if __name__ == "__main__":
    main()
