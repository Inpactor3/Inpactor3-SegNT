"""LoRA sobre el lector v2 (100M) + cabezal, para hongos 24 kb.

Entrena solo los adaptadores LoRA y el cabezal; el resto del lector queda congelado.
Compara contra la referencia congelada: BA 0,8261 · F1 0,7365 · PR-AUC 0,8094.

Uso:
  python lora_hongos.py --seq_dir <dir con train/val/test> --out_dir <salida> --epochs 3
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

CHUNK = 6000
MAX_TOKENS = 1024


class LoRALinear(nn.Module):
    """Capa lineal congelada + adaptadores de bajo rango (LoRA) entrenables."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inyectar_lora(model, r=8, alpha=16, objetivos=("query", "key", "value")):
    """Reemplaza las proyecciones de atencion por versiones con LoRA."""
    for nombre, modulo in list(model.named_modules()):
        if not isinstance(modulo, nn.Linear):
            continue
        if nombre.split(".")[-1] not in objetivos:
            continue
        partes = nombre.split(".")
        padre = model
        for p in partes[:-1]:
            padre = getattr(padre, p)
        setattr(padre, partes[-1], LoRALinear(modulo, r, alpha))


class Head(nn.Module):
    def __init__(self, hidden=512, dropout=0.2):
        super().__init__()
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(512, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):                       # x: (B, C, H)
        B, C, H = x.shape
        f = self.dropout(torch.relu(self.pool_proj(x.reshape(B * C, H))))
        return self.cls(f).view(B, C).max(dim=1).values


def metricas(logits, y, thr=0.5):
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    esp = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    orden = np.argsort(-p); yy = y[orden]
    tpc = np.cumsum(yy); fpc = np.cumsum(1 - yy)
    pc = tpc / np.maximum(tpc + fpc, 1); rc = tpc / max(int(y.sum()), 1)
    return {"precision": prec, "recall": rec, "specificity": esp, "f1": f1,
            "balanced_accuracy": (rec + esp) / 2,
            "pr_auc": float(np.sum(np.diff(np.concatenate([[0], rc])) * pc)),
            "n": int(len(y)), "prevalence": float(y.mean())}


def calibrar(logits, y):
    mejor, thr = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        m = metricas(logits, y, t)
        if m["balanced_accuracy"] > mejor:
            mejor, thr = m["balanced_accuracy"], float(t)
    return round(thr, 3)


def cargar(d, split, cap=0):
    files = sorted(glob.glob(os.path.join(d, split, "*.npz")))
    xs, ys = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        s, y = z["seq"], z["label"]
        if cap and len(y) > cap:
            s, y = s[:cap], y[:cap]
        xs.append(s); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys).astype(np.float32)
    print(f"{split}: {len(files)} archivos · {len(y)} ventanas", flush=True)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repo", default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2, help="ventanas por paso (4 bloques cada una)")
    ap.add_argument("--val_cap", type=int, default=400)
    ap.add_argument("--test_cap", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
    try:
        backbone = AutoModel.from_pretrained(args.repo, trust_remote_code=True)
    except Exception:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module("modeling_esm.EsmModel", args.repo)
        backbone = cls.from_pretrained(args.repo)
    backbone = backbone.to(dev)
    for p in backbone.parameters():
        p.requires_grad = False

    inyectar_lora(backbone, r=args.rank, alpha=args.rank * 2)
    backbone = backbone.to(dev)
    entrenables = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"parámetros LoRA entrenables: {entrenables:,}", flush=True)
    if entrenables == 0:
        raise SystemExit("LoRA no enganchó ningún módulo: revisa target_modules")

    hidden = backbone.config.hidden_size
    head = Head(hidden).to(dev)
    mu = torch.zeros(hidden, device=dev); sd = torch.ones(hidden, device=dev)

    xtr, ytr = cargar(args.seq_dir, "train")
    xva, yva = cargar(args.seq_dir, "val", args.val_cap)
    xte, yte = cargar(args.seq_dir, "test", args.test_cap)

    def bloques(x, idx):
        return x[idx].reshape(-1, CHUNK)

    def forward(xb):                            # xb: (B, C, 6000) uint8
        textos = [bytes(r).decode("ascii", "ignore").upper() for r in xb.reshape(-1, CHUNK)]
        enc = tok(textos, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=MAX_TOKENS).to(dev)
        out = backbone(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        feat = (out * m).sum(1) / m.sum(1).clamp(min=1)
        feat = (feat - mu) / sd
        return feat.view(xb.shape[0], xb.shape[1], -1)

    # mu/sd con 512 bloques de entrenamiento, antes de tocar LoRA
    with torch.no_grad():
        idx = np.random.default_rng(0).choice(len(xtr), min(128, len(xtr)), replace=False)
        f = forward(xtr[idx]).reshape(-1, hidden)
        mu = f.mean(0).clone(); sd = f.std(0).clamp(min=1e-4).clone()

    opt = torch.optim.AdamW([p for p in backbone.parameters() if p.requires_grad]
                            + list(head.parameters()), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def predecir(x, cap_batch=2):
        backbone.eval(); head.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), cap_batch):
                out.append(head(forward(x[i:i + cap_batch])).float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    mejor = -1.0; log = open(os.path.join(args.out_dir, "metrics.csv"), "w")
    log.write("epoch,loss,val_bal_acc,val_f1,val_pr_auc\n")
    for ep in range(1, args.epochs + 1):
        backbone.train(); head.train()
        orden = np.random.default_rng(ep).permutation(len(xtr))
        tot, n = 0.0, 0
        for i in range(0, len(orden), args.batch):
            idx = orden[i:i + args.batch]
            if len(idx) < 2:
                continue
            xb = xtr[idx]
            yb = torch.from_numpy(ytr[idx]).to(dev)
            opt.zero_grad()
            loss = loss_fn(head(forward(xb)), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
            if (i // args.batch) % 200 == 0:
                print(f"  ep{ep} {i}/{len(orden)} loss {tot/max(n,1):.4f}", flush=True)
        lv = predecir(xva)
        m = metricas(lv, yva)
        log.write(f"{ep},{tot/max(n,1):.4f},{m['balanced_accuracy']:.4f},{m['f1']:.4f},{m['pr_auc']:.4f}\n")
        log.flush()
        print(f"ep {ep} loss={tot/max(n,1):.4f} val_BA={m['balanced_accuracy']:.4f} "
              f"val_F1={m['f1']:.4f} val_PR-AUC={m['pr_auc']:.4f}", flush=True)
        if m["balanced_accuracy"] > mejor:
            mejor = m["balanced_accuracy"]
            torch.save({"lora": {k: v for k, v in backbone.state_dict().items() if "lora" in k},
                        "head": head.state_dict(), "mu": mu.cpu().numpy(),
                        "sd": sd.cpu().numpy(), "epoch": ep}, os.path.join(args.out_dir, "best.pt"))

    ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=dev, weights_only=False)
    backbone.load_state_dict({**backbone.state_dict(), **ck["lora"]}, strict=False)
    head.load_state_dict(ck["head"])
    thr = calibrar(predecir(xva), yva)
    res = {"epoch": ck["epoch"], "threshold": thr, "rank": args.rank, "lr": args.lr,
           "at_calibrated_threshold": metricas(predecir(xte), yte, thr),
           "at_050": metricas(predecir(xte), yte, 0.5)}
    json.dump(res, open(os.path.join(args.out_dir, "test_metrics.json"), "w"), indent=2)
    m = res["at_calibrated_threshold"]
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
