"""Precalcula embeddings del backbone congelado NT-500M por ventana.

Uso:
  python precompute_embeddings.py --jsonl_dir <consolidated/R> --genomes_dir <genomes/R> \
      --out_dir <out> --batch_size 32 [--limit 100000]

Genera por split:
  <split>_emb.npy    (N, 1280) float16  (mean pooling del backbone)
  <split>_labels.npy (N,) int8
  <split>_meta.jsonl (window_id, species, split, negative_type)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent.parent / "colab_bridge" / "src"
sys.path.insert(0, str(SRC))

from dataset import WindowDataset
from model import Inpactor2Detect, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_dir", type=Path, default=None)
    ap.add_argument("--genomes_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seq_dir", type=Path, default=None)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--nt_model", default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tok = load_tokenizer(args.nt_model)
    model = Inpactor2Detect(nt_model_name=args.nt_model, freeze_backbone=True).to(device).eval()
    backbone = model.backbone

    for split in args.splits.split(","):
        if args.seq_dir is not None:
            import numpy as np2
            sdir = args.out_dir / split
            sdir.mkdir(parents=True, exist_ok=True)
            files = sorted((args.seq_dir / split).glob("*.npz"))
            if args.num_shards > 1:
                files = files[args.shard_index::args.num_shards]
            if not files:
                print(f"{split}: sin npz, se omite"); continue
            for fp in files:
                out_emb = sdir / f"{fp.stem}_emb.npy"
                out_lab = sdir / f"{fp.stem}_labels.npy"
                out_meta = sdir / f"{fp.stem}_meta.jsonl"
                if out_emb.exists() and out_lab.exists():
                    print(f"SKIP {split} {fp.stem}", flush=True); continue
                z = np2.load(fp)
                seqs = z["seq"]; labs = z["label"]; ids = z["window_id"]
                if len(seqs) == 0:
                    hid = backbone.config.hidden_size
                    np2.save(out_emb, np2.zeros((0, hid), dtype=np.float16))
                    np2.save(out_lab, np2.zeros((0,), dtype=np.int8))
                    open(out_meta, "w").close()
                    print(f"SKIP empty {split} {fp.stem}", flush=True)
                    continue
                if args.max_windows and len(seqs) > args.max_windows:
                    seqs = seqs[:args.max_windows]; labs = labs[:args.max_windows]; ids = ids[:args.max_windows]
                embs, labels, meta = [], [], []
                for i in range(0, len(seqs), args.batch_size):
                    chunk = [x.tobytes().decode("ascii", "ignore") for x in seqs[i:i + args.batch_size]]
                    enc = tok(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=1000)
                    ids_t = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                        out = backbone(input_ids=ids_t, attention_mask=mask).last_hidden_state
                    m = mask.unsqueeze(-1).float()
                    pooled = (out.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                    embs.append(pooled.half().cpu().numpy()); labels.append(labs[i:i + args.batch_size])
                    for j in range(i, min(i + args.batch_size, len(seqs))):
                        meta.append({"window_id": str(ids[j]), "species": fp.stem, "split": split, "negative_type": None})
                emb = np2.concatenate(embs, axis=0); lab = np2.concatenate(labels, axis=0)
                np2.save(out_emb, emb); np2.save(out_lab, lab)
                with open(out_meta, "w") as f:
                    for r in meta: f.write(json.dumps(r) + "\n")
                print(f"OK {split} {fp.stem}: {emb.shape}", flush=True)
            continue
        files = sorted((args.jsonl_dir / split).glob("*.jsonl"))
        if not files:
            print(f"{split}: sin archivos, se omite")
            continue
        ds = WindowDataset(files, args.genomes_dir, tok, max_chunks_per_window=1)
        if args.limit:
            ds.rows = ds.rows[: args.limit]
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device == "cuda"))
        embs, labels, meta = [], [], []
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    out = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).float()
                pooled = (out.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                embs.append(pooled.half().cpu().numpy())
                labels.append(batch["label"].numpy().astype(np.int8))
                for row in dl.dataset.rows[len(meta): len(meta) + len(batch["label"])]:
                    meta.append({
                        "window_id": row.get("window_id"),
                        "species": row.get("species"),
                        "split": split,
                        "negative_type": row.get("negative_type"),
                    })
                if bi % 50 == 0:
                    print(f"{split} batch {bi} · acumuladas {len(labels)*args.batch_size}", flush=True)

        emb = np.concatenate(embs, axis=0)
        lab = np.concatenate(labels, axis=0)
        np.save(args.out_dir / f"{split}_emb.npy", emb)
        np.save(args.out_dir / f"{split}_labels.npy", lab)
        with open(args.out_dir / f"{split}_meta.jsonl", "w") as f:
            for r in meta:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {emb.shape} guardado", flush=True)


if __name__ == "__main__":
    main()
