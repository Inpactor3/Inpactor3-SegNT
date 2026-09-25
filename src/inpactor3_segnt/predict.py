"""
Inferencia sobre un genoma con NTSegmentation.

Toma un checkpoint entrenado y produce Inpactor3_predictions.tab
con el mismo formato que Inpactor2, para permitir comparación directa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import NucleotideSegDataset, collate, parse_fasta
from .model import NTSegmentation, decode_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--scaffold", type=str, default=None,
                    help="Solo predecir sobre este scaffold. Si None, todos.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = NTSegmentation(
        base_model_name=cfg["model"]["base_model"],
        num_classes=cfg["model"]["num_classes"],
        head_channels=cfg["model"].get("head_channels"),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[model] cargado {args.checkpoint} · epoch={ckpt.get('epoch')}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"],
                                              trust_remote_code=True)
    sequences = parse_fasta(args.genome)
    if args.scaffold:
        sequences = {args.scaffold: sequences[args.scaffold]}
    print(f"[data] {len(sequences)} scaffold(s) a procesar")

    ds = NucleotideSegDataset(
        sequences, annotations={}, tokenizer=tokenizer,
        window_size=cfg["data"]["window_size"],
        window_stride=cfg["data"]["window_stride"],
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as out, torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            token_to_bp = [
                [(m["window_start"] + s, m["window_start"] + e)
                 for (s, e) in m["offsets"]]
                for m in batch["meta"]
            ]
            boxes_batch = decode_predictions(
                logits, token_to_bp, threshold=args.threshold,
                cell_size=cfg["data"]["cell_size"],
            )
            for meta, boxes in zip(batch["meta"], boxes_batch):
                scaf = meta["scaffold"]
                for start, end, cls, score in boxes:
                    length = end - start
                    out.write(
                        f"{scaf}\t{start}\t{end}\t{length}\tclass_{cls}\t"
                        f"{score:.4f}\t-\t{score:.4f}\n"
                    )
                    n += 1

    print(f"[ok  ] {n} predicciones escritas en {args.out}")


if __name__ == "__main__":
    main()
