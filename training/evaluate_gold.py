"""Evalua un modelo Inpactor2 contra anotacion curada (test oro).

Ejemplo TAIR10:
  python evaluate_gold.py --genome data/tair10_full/TAIR10.fasta \
    --bed data/curated_te/tair10_te.bed --head results/Plantas/ckpt/best.pt \
    --out results/Gold_TAIR10 --reino Plantas
"""
import argparse, csv, gzip, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

W = 50000
CHUNK = 6000
MAXLEN = 1000
MIN_BP = 50
MIN_FRAC = 0.2


class Head(nn.Module):
    def __init__(self, hidden=1280, dropout=0.2):
        super().__init__()
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        f = self.dropout(torch.relu(self.pool_proj(x)))
        return self.cls_head(f).squeeze(-1)


def read_genome(path):
    op = gzip.open if str(path).endswith(".gz") else open
    seqs, cur, buf = {}, None, []
    with op(path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cur: seqs[cur] = "".join(buf)
    return seqs


def read_ltr_bed(path):
    hits = {}
    ltr = nonltr = 0
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4: continue
            name = c[3]
            cls = name.split("|")[1] if "|" in name else ""
            if "LTR" not in cls:
                nonltr += 1
                continue
            ltr += 1
            hits.setdefault(c[0], []).append((int(c[1]), int(c[2])))
    return hits, ltr, nonltr


def metrics(labels, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    bal = (rec + spec) / 2
    # PR-AUC manual
    order = np.argsort(-probs)
    y = labels[order]; p = probs[order]
    tp_c = np.cumsum(y); fp_c = np.cumsum(1 - y)
    prec_c = tp_c / np.maximum(tp_c + fp_c, 1)
    rec_c = tp_c / max(int(labels.sum()), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec_c])) * prec_c))
    return dict(precision=prec, recall=rec, specificity=spec, f1=f1,
                accuracy=acc, balanced_accuracy=bal, pr_auc=ap,
                tp=tp, fp=fp, fn=fn, tn=tn, n=len(labels),
                prevalence=float(labels.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--bed", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reino", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chroms", default=None)
    ap.add_argument("--nt_model", default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    print("device:", device, flush=True)

    genome = read_genome(args.genome)
    hits, n_ltr, n_non = read_ltr_bed(args.bed)
    print(f"cromosomas: {len(genome)} · TEs LTR curados: {n_ltr} · no-LTR: {n_non}", flush=True)

    windows = []
    keep = set(args.chroms.split(",")) if args.chroms else None
    for chrom, seq in genome.items():
        if keep is not None and chrom not in keep:
            continue
        L = len(seq)
        for st in range(0, L - W + 1, W):
            en = st + W
            label = 0
            for hs, he in hits.get(chrom, []):
                if hs >= en or he <= st: continue
                ov = min(he, en) - max(hs, st)
                thr = max(MIN_BP, int(MIN_FRAC * (he - hs)))
                if ov >= thr:
                    label = 1; break
            windows.append((chrom, st, en, label, seq[st:st + CHUNK].upper()))
    print(f"ventanas: {len(windows)} · positivas: {sum(w[3] for w in windows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.nt_model)
    backbone = AutoModel.from_pretrained(args.nt_model).to(device).eval()
    hidden = backbone.config.hidden_size
    head = Head(hidden).to(device)
    ck = torch.load(args.head, map_location=device)
    head.load_state_dict(ck["head"]); head.eval()
    print("modelo cargado · head epoch", ck.get("epoch"), flush=True)

    probs = []
    with torch.no_grad():
        for i in range(0, len(windows), args.batch):
            batch = windows[i:i + args.batch]
            seqs = [w[4] for w in batch]
            enc = tok(seqs, return_tensors="pt", truncation=True,
                      padding="max_length", max_length=MAXLEN)
            ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
            out = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
            logits = head(pooled).float().cpu().numpy()
            probs.extend(1 / (1 + np.exp(-logits)))
            if i % 400 == 0:
                print(f"{i}/{len(windows)}", flush=True)

    labels = np.array([w[3] for w in windows])
    probs = np.array(probs)
    res = metrics(labels, probs)
    always_pos = metrics(labels, np.ones_like(probs))
    best = (-1, 0.5)
    for thr in np.arange(0.05, 0.96, 0.05):
        m = metrics(labels, probs, thr)
        if m["f1"] > best[0]:
            best = (m["f1"], float(thr))
    res["best_threshold"] = best[1]
    res["f1_at_best_threshold"] = best[0]
    res["always_positive"] = always_pos

    with open(args.out / "windows.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["chrom", "start", "end", "label", "prob", "pred"])
        for (chrom, st, en, lab, _), p in zip(windows, probs):
            w.writerow([chrom, st, en, lab, f"{p:.6f}", int(p >= 0.5)])
    with open(args.out / "gold_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
