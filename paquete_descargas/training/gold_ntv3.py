"""Evalua un cabezal entrenado con embeddings NTv3 sobre un genoma oro.

No reentrena nada: carga el cabezal de la corrida de plantas (o la que se indique),
calcula los embeddings NTv3 de las ventanas del oro y aplica ese cabezal.

Uso:
  python gold_ntv3.py --genome <TAIR10.fasta> --bed <tair10_te.bed> \
      --head <ntv3_plantas/modelo/best.pt> --out <salida> --repo InstaDeepAI/NTv3_100M_pre
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_gold import label_window, read_genome, read_ltr_bed  # noqa: E402
from block_layout import block_offsets  # noqa: E402

W, CHUNK = 50000, 6000


class Head(nn.Module):
    def __init__(self, hidden=768, dropout=0.2):
        super().__init__()
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(512, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):                        # (B, C, H)
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
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": int(len(y)), "prevalence": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--bed", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="InstaDeepAI/NTv3_100M_pre")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2, help="ventanas por paso")
    ap.add_argument("--chroms", default="")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.head, map_location=dev, weights_only=False)
    hidden = int(ck.get("hidden", 768))
    modo = str(ck.get("mode", "logits_max"))
    head = Head(hidden).to(dev)
    head.load_state_dict(ck["head"]); head.eval()
    mu = torch.as_tensor(ck["mu"], device=dev, dtype=torch.float32)
    sd = torch.as_tensor(ck["sd"], device=dev, dtype=torch.float32)
    thr = float(ck.get("threshold", 0.5))
    print(f"cabezal: hidden={hidden} modo={modo} umbral={thr}", flush=True)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.repo, trust_remote_code=True).eval().to(dev)

    offs = block_offsets(args.chunks, W, CHUNK, "windowed")
    genome = read_genome(args.genome)
    keep = set(args.chroms.split(",")) if args.chroms else None
    hits, n_ltr, n_non = read_ltr_bed(args.bed, set())
    print(f"cromosomas {len(genome)} · LTR {n_ltr} · no-LTR {n_non}", flush=True)

    windows = []
    for chrom, seq in genome.items():
        if keep is not None and chrom not in keep:
            continue
        L = len(seq)
        for st in range(0, L - W + 1, W):
            lab, _ = label_window(hits, chrom, st, st + W, "ref", 50, 0.2)
            blocks = [seq[st + int(o): st + int(o) + CHUNK].upper() for o in offs]
            blocks = [b if len(b) >= 100 else "N" * CHUNK for b in blocks]
            windows.append((lab, blocks))
    y = np.array([w[0] for w in windows], dtype=np.int32)
    print(f"ventanas {len(windows)} · positivas {int(y.sum())} ({y.mean():.3f})", flush=True)

    logits = np.zeros(len(windows), dtype=np.float32)
    C = len(offs)
    for i in range(0, len(windows), args.batch):
        lote = windows[i:i + args.batch]
        planos = [b for _, bl in lote for b in bl]
        textos = [t if t else "N" * CHUNK for t in planos]
        enc = tok(textos, add_special_tokens=False, padding=True,
                  pad_to_multiple_of=128, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
            h = o.hidden_states[-1][:, :CHUNK].mean(dim=1)      # (b*C, H)
            h = (h.float() - mu) / sd
            per = h.view(len(lote), C, -1)
            if modo.startswith("logits"):
                f = torch.relu(head.pool_proj(per.reshape(-1, hidden)))
                lg = head.cls(f).view(len(lote), C).max(dim=1).values
            else:
                lg = head(per)
        logits[i:i + len(lote)] = lg.float().cpu().numpy()
        del o, h, per, lg
        if i % (args.batch * 50) == 0:
            print(f"  {i}/{len(windows)}", flush=True)

    res = {"head": str(args.head), "threshold": thr, "chunks": args.chunks,
           "at_calibrated_threshold": metricas(logits, y, thr),
           "at_050": metricas(logits, y, 0.5),
           "always_positive": metricas(np.full_like(logits, 10.0), y, 0.5)}
    json.dump(res, open(out / "gold_metrics.json", "w"), indent=2)
    np.savez_compressed(out / "probs.npz", prob=1 / (1 + np.exp(-logits)), label=y)
    m = res["at_calibrated_threshold"]
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
