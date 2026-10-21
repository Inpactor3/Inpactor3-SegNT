"""
Loop de entrenamiento Inpactor2 Detect.
Entrena sobre un conjunto de JSONL + genomas, valida sobre un holdout.
Metricas: precision, recall, F1, AUROC.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import WindowDataset
from model import Inpactor2Detect, load_tokenizer


def metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(labels)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_dir", type=Path, required=True,
                    help="Directorio con los *.jsonl (train+val+test en subdirs)")
    ap.add_argument("--genomes_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("/content/inpactor2_detect/checkpoints"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_chunks_per_window", type=int, default=1)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = load_tokenizer()
    print("Tokenizer OK")

    train_files = sorted((args.jsonl_dir / "train").glob("*.jsonl"))
    val_files = sorted((args.jsonl_dir / "val").glob("*.jsonl"))
    print(f"train: {len(train_files)} archivos, val: {len(val_files)} archivos")

    train_ds = WindowDataset(train_files, args.genomes_dir, tokenizer,
                             max_chunks_per_window=args.max_chunks_per_window)
    val_ds = WindowDataset(val_files, args.genomes_dir, tokenizer,
                           max_chunks_per_window=args.max_chunks_per_window)
    if args.limit_train:
        train_ds.rows = train_ds.rows[:args.limit_train]
    if args.limit_val:
        val_ds.rows = val_ds.rows[:args.limit_val]
    print(f"Ventanas train: {len(train_ds)}  val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

    print("Cargando modelo...")
    model = Inpactor2Detect(freeze_backbone=args.freeze_backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params totales: {n_params/1e6:.1f}M  entrenables: {n_train_params/1e6:.1f}M")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"ep {epoch+1}/{args.epochs} train")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            lab = batch["label"].to(device)
            out = model(input_ids, att)
            loss = loss_fn(out["logits"], lab)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"ep {epoch+1}/{args.epochs} val"):
                out = model(batch["input_ids"].to(device),
                            batch["attention_mask"].to(device))
                all_logits.append(out["logits"].cpu().numpy())
                all_labels.append(batch["label"].numpy())
        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        m = metrics(logits, labels)
        dt = time.time() - t0
        print(f"\n[ep {epoch+1}] loss={total_loss/n_batches:.4f} "
              f"F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"acc={m['accuracy']:.3f}  tiempo={dt/60:.1f} min")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            ckpt = args.out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m}, ckpt)
            print(f"  checkpoint guardado en {ckpt}  (F1 mejor)")

        with open(args.out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "loss": total_loss/n_batches, **m}) + "\n")

    print(f"\n=== Entrenamiento OK · mejor F1 val = {best_f1:.4f} ===")


if __name__ == "__main__":
    main()
