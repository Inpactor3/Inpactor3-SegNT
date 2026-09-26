"""
Entrenamiento de NTSegmentation sobre ventanas de Arabidopsis.

Uso:
    python -m inpactor3_segnt.train --config configs/nt_50m.yaml
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

from .dataset import (
    NucleotideSegDataset, build_splits, collate,
    parse_fasta, parse_inpactor2_tab,
)
from .model import NTSegmentation, token_loss


def evaluate_window_f1(model, loader, device) -> dict:
    """
    Evalúa F1 de PRESENCIA POR VENTANA (métrica del informe):
    una ventana cuenta como positiva si ≥1 token predicho es LTR-RT.
    """
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(-1).cpu()
            # positivo por ventana si algún token != 0 y != -100
            for b in range(input_ids.size(0)):
                gt_pos = (labels[b] > 0).any().item()
                pr_pos = ((preds[b] > 0) & (attention_mask[b].cpu() == 1)).any().item()
                if gt_pos and pr_pos:
                    tp += 1
                elif gt_pos and not pr_pos:
                    fn += 1
                elif not gt_pos and pr_pos:
                    fp += 1
                else:
                    tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1_window": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} · pytorch={torch.__version__}")

    # ---- cargar datos ----
    print(f"[data] genome:      {cfg['data']['genome']}")
    print(f"[data] annotations: {cfg['data']['annotations']}")
    sequences = parse_fasta(Path(cfg["data"]["genome"]))
    annotations = parse_inpactor2_tab(Path(cfg["data"]["annotations"]))
    print(f"[data] {len(sequences)} scaffolds · "
          f"{sum(len(v) for v in annotations.values())} anotaciones")

    tr_seq, tr_ann, va_seq, va_ann, te_seq, te_ann = build_splits(
        sequences, annotations,
        cfg["data"]["train_chromosomes"],
        cfg["data"]["val_chromosomes"],
        cfg["data"]["test_chromosomes"],
    )
    print(f"[split] train: {list(tr_seq)} · val: {list(va_seq)} · test: {list(te_seq)}")

    # ---- tokenizer + datasets ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"],
                                              trust_remote_code=True)
    make_ds = lambda s, a: NucleotideSegDataset(
        s, a, tokenizer,
        window_size=cfg["data"]["window_size"],
        window_stride=cfg["data"]["window_stride"],
        min_positive_frac=cfg["data"].get("min_positive_frac", 0.0),
    )
    train_ds, val_ds = make_ds(tr_seq, tr_ann), make_ds(va_seq, va_ann)
    print(f"[data] train ventanas: {len(train_ds)} · val: {len(val_ds)}")
    print(f"[data] train positivas: {len(train_ds.positive_indices)} · "
          f"negativas: {len(train_ds.negative_indices)}")

    # WeightedRandomSampler: cada ventana positiva pesa mucho más para que
    # cada batch tenga ~50% positivos aunque solo sean 1% del dataset.
    n_total = len(train_ds)
    n_pos = max(len(train_ds.positive_indices), 1)
    n_neg = max(len(train_ds.negative_indices), 1)
    weights = [0.0] * n_total
    for i in train_ds.positive_indices:
        weights[i] = 0.5 / n_pos
    for i in train_ds.negative_indices:
        weights[i] = 0.5 / n_neg
    # Muestreamos N samples por época (usamos min de 4x positivos, o total)
    n_samples_per_epoch = min(n_total, max(200, 20 * n_pos))
    sampler = WeightedRandomSampler(weights, num_samples=n_samples_per_epoch,
                                    replacement=True)
    print(f"[data] muestras por época: {n_samples_per_epoch}")

    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          sampler=sampler, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, collate_fn=collate, num_workers=2)

    # ---- modelo ----
    model = NTSegmentation(
        base_model_name=cfg["model"]["base_model"],
        num_classes=cfg["model"]["num_classes"],
        head_channels=cfg["model"].get("head_channels"),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {cfg['model']['base_model']} · {n_params:.1f} M params")

    # ---- optimizer ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = len(train_dl) * cfg["train"]["epochs"] // \
        cfg["train"].get("gradient_accumulation", 1)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=cfg["train"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # ---- entrenamiento ----
    out_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    patience = cfg["train"].get("early_stopping_patience", 3)
    bad = 0
    history = []
    accum = cfg["train"].get("gradient_accumulation", 1)

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        t0 = time.time()
        losses = []
        opt.zero_grad()
        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = token_loss(logits, labels, attention_mask,
                              pos_weight=cfg["train"]["pos_weight"])
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        val = evaluate_window_f1(model, val_dl, device)
        dt = time.time() - t0
        print(f"[ep {ep:02d}] loss={avg_loss:.4f} · val_f1_window={val['f1_window']:.3f} "
              f"(P={val['precision']:.2f} R={val['recall']:.2f}) · {dt:.0f}s")
        history.append({"epoch": ep, "loss": avg_loss, **val, "time_s": dt})

        # guardar mejor
        if val["f1_window"] > best_f1:
            best_f1 = val["f1_window"]
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "epoch": ep,
                "val_metrics": val,
            }, out_dir / "best.pt")
            print(f"    ↳ nuevo mejor · guardado en {out_dir/'best.pt'}")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[stop] early stopping tras {patience} épocas sin mejora")
                break

    # log final
    (Path(cfg["output"].get("log_dir", "logs"))).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["output"].get("log_dir", "logs")) / "train_history.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, indent=2))
    print(f"[out ] historial → {log_path}")
    print(f"[best] val_f1_window = {best_f1:.3f}")


if __name__ == "__main__":
    main()
