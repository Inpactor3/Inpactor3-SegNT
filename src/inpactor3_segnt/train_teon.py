"""
Entrenamiento cross-kingdom con TeonWindowDataset (PanTEon plants+animals+fungi).

Uso:
    python -m inpactor3_segnt.train_teon --config configs/nt_50m_teon.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .model import NTSegmentation, token_loss
from .teon_dataset import TeonWindowDataset


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def eval_f1(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(-1).cpu()
            for b in range(input_ids.size(0)):
                gt_pos = (labels[b] > 0).any().item()
                pr_pos = ((preds[b] > 0) & (attention_mask[b].cpu() == 1)).any().item()
                if gt_pos and pr_pos: tp += 1
                elif gt_pos: fn += 1
                elif pr_pos: fp += 1
                else: tn += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"precision": p, "recall": r, "f1_window": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} · torch={torch.__version__}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"], trust_remote_code=True
    )

    # Datasets lazy
    print("[data] indexando genomas de fondo (una vez)...")
    make_ds = lambda split: TeonWindowDataset(
        manifest=Path(cfg["data"]["manifest"]),
        genomes_dir=Path(cfg["data"]["genomes_dir"]),
        panteon_subset=Path(cfg["data"]["panteon_subset"]),
        tokenizer=tokenizer,
        split=split,
        window_size=cfg["data"].get("window_size", 6000),
        max_tokens=cfg["data"].get("max_tokens", 1000),
        cell_size=cfg["data"].get("cell_size", 100),
    )
    train_ds = make_ds("train")
    val_ds = make_ds("val")
    print(f"[data] train ventanas: {len(train_ds)} · val: {len(val_ds)}")

    # Oversample de positivos
    pos_idx = [i for i, r in enumerate(train_ds.recipes) if r.has_ltr_rt]
    neg_idx = [i for i, r in enumerate(train_ds.recipes) if not r.has_ltr_rt]
    weights = [0.0] * len(train_ds)
    for i in pos_idx: weights[i] = 0.5 / max(len(pos_idx), 1)
    for i in neg_idx: weights[i] = 0.5 / max(len(neg_idx), 1)
    n_samples = min(len(train_ds), max(500, 20 * len(pos_idx)))
    sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)
    print(f"[data] positivos: {len(pos_idx)} · negativos: {len(neg_idx)} · "
          f"muestras/época: {n_samples}")

    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          sampler=sampler, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, collate_fn=collate, num_workers=2)

    # Modelo
    model = NTSegmentation(
        base_model_name=cfg["model"]["base_model"],
        num_classes=cfg["model"]["num_classes"],
        head_channels=cfg["model"].get("head_channels"),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[model] {cfg['model']['base_model']} · {n_params:.1f} M params entrenables")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    total_steps = len(train_dl) * cfg["train"]["epochs"]
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=cfg["train"].get("warmup_steps", 100),
        num_training_steps=total_steps,
    )

    out_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    patience = cfg["train"].get("early_stopping_patience", 5)
    bad = 0
    history = []

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        losses = []
        t0 = time.time()
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = token_loss(logits, labels, attention_mask,
                              pos_weight=cfg["train"].get("pos_weight", 50.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        val = eval_f1(model, val_dl, device)
        dt = time.time() - t0
        print(f"[ep {ep:02d}] loss={avg_loss:.4f} · val_f1={val['f1_window']:.3f} "
              f"(P={val['precision']:.2f} R={val['recall']:.2f}) · {dt:.0f}s")
        history.append({"epoch": ep, "loss": avg_loss, **val, "time_s": dt})

        if val["f1_window"] > best_f1:
            best_f1 = val["f1_window"]
            torch.save({"model": model.state_dict(), "config": cfg,
                        "epoch": ep, "val_metrics": val},
                       out_dir / "best_teon.pt")
            print(f"    ↳ mejor · guardado")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[stop] early stopping tras {patience} épocas sin mejora")
                break

    log_dir = Path(cfg["output"].get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_teon_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[best] val_f1 = {best_f1:.3f}")


if __name__ == "__main__":
    main()
