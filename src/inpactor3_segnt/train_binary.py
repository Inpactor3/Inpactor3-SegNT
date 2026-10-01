"""
Entrenamiento binario de SegNT sobre inpactor2_detect_panteon_v1 (80k ventanas).

Config: configs/nt_50m_binary.yaml

Uso en servidor:
    python -m inpactor3_segnt.train_binary \
        --config configs/nt_50m_binary.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .binary_model import NTBinaryClassifier, binary_loss
from .npy_dataset import PanteonBinaryDataset, collate


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    all_probs = []
    all_labels = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        logits = model(ids, mask)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()
        tn += ((preds == 0) & (y == 0)).sum().item()
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": tp + fp + fn + tn}


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

    # Datasets
    npy_dir = Path(cfg["data"]["npy_dir"])
    print(f"[data] cargando .npy de {npy_dir}")
    train_ds = PanteonBinaryDataset(npy_dir, "train", tokenizer=tokenizer,
                                     chunk_size=cfg["data"].get("chunk_size", 5000),
                                     n_chunks=cfg["data"].get("n_chunks", 10),
                                     max_tokens=cfg["data"].get("max_tokens", 1000))
    dev_ds = PanteonBinaryDataset(npy_dir, "dev", tokenizer=tokenizer,
                                   chunk_size=cfg["data"].get("chunk_size", 5000),
                                   n_chunks=cfg["data"].get("n_chunks", 10),
                                   max_tokens=cfg["data"].get("max_tokens", 1000))
    print(f"[data] train={len(train_ds)}  dev={len(dev_ds)}")
    pos_train = int(train_ds.y[:].sum())
    print(f"[data] train positivos={pos_train} ({100*pos_train/len(train_ds):.1f}%)")

    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          shuffle=True, collate_fn=collate,
                          num_workers=cfg["train"].get("num_workers", 2))
    dev_dl = DataLoader(dev_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, collate_fn=collate,
                        num_workers=cfg["train"].get("num_workers", 2))

    # Modelo
    model = NTBinaryClassifier(
        base_model_name=cfg["model"]["base_model"],
        freeze_encoder=cfg["model"].get("freeze_encoder", True),
        head_hidden=cfg["model"].get("head_hidden", 256),
        dropout=cfg["model"].get("dropout", 0.1),
        pool=cfg["model"].get("pool", "max"),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg['model']['base_model']} · "
          f"{n_trainable/1e6:.2f} M params entrenables (pool={model.pool})")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    total_steps = len(train_dl) * cfg["train"]["epochs"]
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=cfg["train"].get("warmup_steps", 200),
        num_training_steps=total_steps,
    )

    out_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["output"].get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience = cfg["train"].get("early_stopping_patience", 15)
    bad = 0
    history: list[dict] = []

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        losses = []
        t0 = time.time()
        for step, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(ids, mask)
            loss = binary_loss(logits, y, pos_weight=cfg["train"].get("pos_weight", 1.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        val = evaluate(model, dev_dl, device)
        dt = time.time() - t0
        print(f"[ep {ep:03d}] loss={avg_loss:.4f} · "
              f"dev_f1={val['f1']:.4f} (P={val['precision']:.3f} R={val['recall']:.3f}) · "
              f"acc={val['accuracy']:.4f} · {dt:.0f}s")
        history.append({"epoch": ep, "loss": avg_loss, **val, "time_s": dt})

        # Guarda historial cada época (para no perder si se cae)
        (log_dir / "train_binary_history.json").write_text(json.dumps(history, indent=2))

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "config": cfg,
                        "epoch": ep, "val_metrics": val},
                       out_dir / "best_binary.pt")
            print(f"    ↳ nuevo mejor · guardado")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[stop] early stopping tras {patience} épocas sin mejora")
                break

        # Guardado periódico cada 10 épocas
        if ep % 10 == 0:
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep},
                       out_dir / f"ckpt_ep{ep:03d}.pt")

    print(f"\n[best] dev_f1 = {best_f1:.4f}")

    # Evaluación final sobre test
    print("\n=== Evaluación final sobre test ===")
    test_ds = PanteonBinaryDataset(npy_dir, "test", tokenizer=tokenizer,
                                    chunk_size=cfg["data"].get("chunk_size", 5000),
                                    n_chunks=cfg["data"].get("n_chunks", 10),
                                    max_tokens=cfg["data"].get("max_tokens", 1000))
    test_dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                         shuffle=False, collate_fn=collate, num_workers=2)
    # Cargar best
    ckpt = torch.load(out_dir / "best_binary.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_dl, device)
    print(f"[test] f1={test['f1']:.4f} · P={test['precision']:.3f} · "
          f"R={test['recall']:.3f} · acc={test['accuracy']:.4f}")
    (log_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
