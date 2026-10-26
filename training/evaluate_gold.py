"""Evalua un modelo Inpactor2 contra anotacion curada (test oro).

Cambios v2:
  * Entrada de 4 bloques de 6 kb dentro de la ventana (24 kb) con **max pooling**,
    igual que en entrenamiento. `--chunks 8` usa 48 kb para el paso A100.
  * Definicion LTR **armonizada** con el entrenamiento (`relabel_ltr.py`):
      - se acepta si el `class` del nombre del BED contiene "LTR"
        (cubre `LTR/Gypsy`, `LTR/Copia`, `LTR/Gypsy|...`) y `--ltr_orders` lo amplia;
      - el solapamiento se mide contra la *referencia* (LTR del BED) o contra la
        *ventana* (`--overlap_mode window`, canon de Inpactor/EDTA), con
        `--min_bp 50` y `--min_frac 0.2`.
    El criterio se escribe en `ltr_definition.json` junto al resultado.
  * Umbral **calibrado en validacion** (`best.pt["threshold"]`); se reportan las
    metricas en el umbral calibrado y en 0.5, mas PR-AUC y balanced accuracy.

Ejemplo TAIR10:
  python evaluate_gold.py --genome TAIR10.fasta.gz --bed tair10_te.bed \
    --head ckpt/Plantas/best.pt --out results/Gold_TAIR10 --reino Plantas --chunks 4
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from block_layout import block_offsets

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
                if cur:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


def parse_class(name):
    parts = name.split("|")
    return parts[1] if len(parts) > 1 else ""


def is_ltr(cls, ltr_orders):
    tokens = [t.strip().upper() for t in cls.split("/") if t.strip()]
    if not tokens:
        return False
    if tokens[0] == "LTR":
        return True
    return any(t in ltr_orders for t in tokens[1:])


def read_ltr_bed(path, ltr_orders):
    hits = {}
    n_ltr = n_non = 0
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            if is_ltr(parse_class(c[3]), ltr_orders):
                n_ltr += 1
                hits.setdefault(c[0], []).append((int(c[1]), int(c[2])))
            else:
                n_non += 1
    return hits, n_ltr, n_non


def label_window(hits, chrom, st, en, overlap_mode, min_bp, min_frac):
    """overlap_mode='ref' -> >=min_frac del TE; 'window' -> >=min_frac de la ventana."""
    for hs, he in hits.get(chrom, []):
        if hs >= en or he <= st:
            continue
        ov = min(he, en) - max(hs, st)
        base = (he - hs) if overlap_mode == "ref" else (en - st)
        thr = max(min_bp, int(min_frac * base))
        if ov >= thr:
            return 1, ov
    return 0, 0


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
    bal = (rec + spec) / 2
    order = np.argsort(-probs)
    y = labels[order]
    tp_c = np.cumsum(y)
    fp_c = np.cumsum(1 - y)
    prec_c = tp_c / np.maximum(tp_c + fp_c, 1)
    rec_c = tp_c / max(int(labels.sum()), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec_c])) * prec_c))
    return dict(threshold=thr, precision=prec, recall=rec, specificity=spec, f1=f1,
                accuracy=(tp + tn) / max(len(labels), 1), balanced_accuracy=bal,
                pr_auc=ap, tp=tp, fp=fp, fn=fn, tn=tn, n=len(labels),
                prevalence=float(labels.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--bed", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reino", default="")
    ap.add_argument("--chroms", default=None)
    ap.add_argument("--chunks", type=int, default=4,
                    help="4 = entrada de entrenamiento (24 kb); 8 = 48 kb en A100")
    ap.add_argument("--mode", choices=("windowed", "eval_full"), default="windowed")
    ap.add_argument("--max_tokens", type=int, default=MAXLEN)
    ap.add_argument("--batch", type=int, default=8, help="ventanas por paso")
    ap.add_argument("--token_batch", type=int, default=64)
    ap.add_argument("--overlap_mode", choices=("ref", "window"), default="ref")
    ap.add_argument("--min_bp", type=int, default=MIN_BP)
    ap.add_argument("--min_frac", type=float, default=MIN_FRAC)
    ap.add_argument("--ltr_orders", default="")
    ap.add_argument("--threshold", type=float, default=None,
                    help="si se omite se usa best.pt['threshold'] o 0.5")
    ap.add_argument("--nt_model",
                    default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ltr_orders = {t.strip().upper() for t in args.ltr_orders.split(",") if t.strip()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    print("device:", device, "chunks:", args.chunks, flush=True)

    offs = block_offsets(args.chunks, W, CHUNK, args.mode)
    genome = read_genome(args.genome)
    keep_chroms = set(args.chroms.split(",")) if args.chroms else None
    hits, n_ltr, n_non = read_ltr_bed(args.bed, ltr_orders)
    print(f"cromosomas: {len(genome)} · TEs LTR curados: {n_ltr} · no-LTR: {n_non}",
          flush=True)

    windows = []
    for chrom, seq in genome.items():
        if keep_chroms is not None and chrom not in keep_chroms:
            continue
        L = len(seq)
        for st in range(0, L - W + 1, W):
            en = st + W
            lab, ov = label_window(hits, chrom, st, en, args.overlap_mode,
                                   args.min_bp, args.min_frac)
            blocks = []
            for off in offs:
                piece = seq[st + int(off): st + int(off) + CHUNK].upper()
                blocks.append(piece if len(piece) >= 100 else "N" * CHUNK)
            windows.append((chrom, st, en, lab, ov, blocks))
    npos = sum(w[3] for w in windows)
    print(f"ventanas: {len(windows)} · positivas: {npos} "
          f"({npos / max(len(windows), 1):.3f}) · bloques/ventana: {len(offs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.nt_model)
    backbone = AutoModel.from_pretrained(args.nt_model).to(device).eval()
    hidden = backbone.config.hidden_size
    ck = torch.load(args.head, map_location=device)
    if ck.get("hidden") and int(ck["hidden"]) != hidden:
        raise SystemExit(f"best.pt espera hidden={ck['hidden']} pero {args.nt_model} da "
                         f"hidden={hidden}: usa el mismo --nt_model que en entrenamiento")
    head = Head(hidden).to(device)
    head.load_state_dict(ck["head"])
    head.eval()
    mu, sd = ck.get("mu"), ck.get("sd")
    thr = args.threshold if args.threshold is not None else float(ck.get("threshold", 0.5))
    print(f"modelo cargado · head epoch {ck.get('epoch')} · umbral {thr}"
          f"{' · estandarizado' if mu is not None else ''}", flush=True)

    C = len(offs)
    probs = np.zeros(len(windows))
    for i in range(0, len(windows), args.batch):
        batch = windows[i:i + args.batch]
        flat = [b for w in batch for b in w[5]]
        enc = tok(flat, return_tensors="pt", truncation=True,
                  padding="max_length", max_length=args.max_tokens)
        block_feats = []
        for j in range(0, len(flat), args.token_batch):
            sl = slice(j, j + args.token_batch)
            ids = enc["input_ids"][sl].to(device)
            mask = enc["attention_mask"][sl].to(device)
            with torch.no_grad():
                out = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).float()
                pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
                feat = pooled if mu is None else (pooled - torch.as_tensor(
                    mu, device=device, dtype=torch.float32)) / torch.as_tensor(
                    sd, device=device, dtype=torch.float32)
                block_feats.append(feat.float())
        # Igual que en entrenamiento: max pooling sobre los EMBEDDINGS de los
        # bloques y despues una sola pasada del cabezal. Asi 4 y 8 bloques son
        # comparables y la evaluacion coincide con lo que se entreno.
        per_window = torch.cat(block_feats).view(len(batch), C, -1)
        pooled_emb = per_window.max(dim=1).values
        with torch.no_grad():
            logits = head(pooled_emb).float()
        probs[i:i + len(batch)] = torch.sigmoid(logits).cpu().numpy()
        if i % (args.batch * 50) == 0:
            print(f"{i}/{len(windows)}", flush=True)

    labels = np.array([w[3] for w in windows])
    res = {"threshold": thr, "chunks": C, "mode": args.mode,
           "bp_per_window": C * CHUNK, "overlap_mode": args.overlap_mode,
           "min_bp": args.min_bp, "min_frac": args.min_frac}
    res["at_calibrated_threshold"] = metrics(labels, probs, thr)
    res["at_050"] = metrics(labels, probs, 0.5)
    res["always_positive"] = metrics(labels, np.ones_like(probs))
    res["best_f1_threshold"] = float(
        max(np.arange(0.05, 0.96, 0.05), key=lambda t: metrics(labels, probs, t)["f1"]))

    with open(args.out / "ltr_definition.json", "w") as f:
        json.dump({"source": str(args.bed), "rule": "class contains 'LTR' (harmonized "
                   "with relabel_ltr.py) or order in --ltr_orders",
                   "ltr_orders": sorted(ltr_orders), "n_ltr": n_ltr, "n_non_ltr": n_non,
                   "overlap_mode": args.overlap_mode, "min_bp": args.min_bp,
                   "min_frac": args.min_frac, "window_bp": W,
                   "blocks": offs.tolist(), "chunk_bp": CHUNK}, f, indent=2)
    with open(args.out / "windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chrom", "start", "end", "label", "overlap_bp", "prob",
                    "pred_at_thr", "pred_at_0.5"])
        for (chrom, st, en, lab, ov, _), p in zip(windows, probs):
            w.writerow([chrom, st, en, lab, ov, f"{p:.6f}", int(p >= thr), int(p >= 0.5)])
    with open(args.out / "gold_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: round(vv, 4) if isinstance(vv, float) else vv
                           for kk, vv in v.items()}) for k, v in res.items()}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
