"""Evalua la anotacion RM+PanTEon contra la anotacion curada TAIR10.

Calcula F1 / Recall / Precision a nivel de elemento (IoU >= 0.5).

- Verdad: data/curated_te/tair10_te.bed       (31,189 TEs curados)
- Prediccion: annotations/arabidopsis.bed      (21,409 TEs de RM+PanTEon)

Metricas:
    - RECALL   = TP / (TP + FN)   (de los TEs curados reales, cuántos encontramos)
    - PRECISION = TP / (TP + FP)  (de los que marcamos, cuántos son reales)
    - F1       = 2PR / (P+R)
    - IoU por elemento (TP cuando IoU >= 0.5)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRUE_BED = ROOT / "data" / "curated_te" / "tair10_te.bed"
PRED_BED = ROOT / "annotations" / "arabidopsis.bed"
OUT_MD = ROOT / "docs" / "test_oro_arabidopsis.md"

IOU_THRESHOLD = 0.5


def read_bed(path: Path):
    data = defaultdict(list)
    with path.open() as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 3:
                continue
            chrom = s[0]
            start = int(s[1])
            end = int(s[2])
            name = s[3] if len(s) > 3 else ""
            data[chrom].append((start, end, name))
    for k in data:
        data[k].sort()
    return data


def iou(a_start, a_end, b_start, b_end):
    inter_s = max(a_start, b_start)
    inter_e = min(a_end, b_end)
    inter = max(0, inter_e - inter_s)
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def match_elements(true_hits, pred_hits):
    """Para cada TE verdadero, encontrar la mejor prediccion solapante."""
    tp = 0
    fn = 0
    matched_preds = set()  # indices de preds que ya fueron TP
    true_matches = []  # para analisis

    for chrom in sorted(true_hits):
        true_list = true_hits.get(chrom, [])
        pred_list = pred_hits.get(chrom, [])
        for i, (ts, te_end, tname) in enumerate(true_list):
            # buscar prediccion con mayor IoU
            best_iou = 0.0
            best_j = -1
            for j, (ps, pe, pname) in enumerate(pred_list):
                if pe < ts or ps > te_end:
                    continue
                score = iou(ts, te_end, ps, pe)
                if score > best_iou:
                    best_iou = score
                    best_j = j
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_preds.add((chrom, best_j))
                true_matches.append((tname, best_iou, "TP"))
            else:
                fn += 1
                true_matches.append((tname, best_iou, "FN"))

    # FPs: predicciones sin match
    fp = 0
    for chrom, pred_list in pred_hits.items():
        for j, _ in enumerate(pred_list):
            if (chrom, j) not in matched_preds:
                fp += 1

    return tp, fp, fn, true_matches


def main():
    print(f"=== Test ORO: Arabidopsis (TAIR10 curado vs RM+PanTEon) ===\n")

    true_hits = read_bed(TRUE_BED)
    pred_hits = read_bed(PRED_BED)

    n_true = sum(len(v) for v in true_hits.values())
    n_pred = sum(len(v) for v in pred_hits.values())
    print(f"TEs verdad (TAIR10 curado):   {n_true:,}")
    print(f"TEs prediccion (RM+PanTEon):  {n_pred:,}")
    print(f"Umbral IoU: {IOU_THRESHOLD}")
    print(f"Cromosomas comunes: {set(true_hits) & set(pred_hits)}")

    tp, fp, fn, true_matches = match_elements(true_hits, pred_hits)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n=== RESULTADOS ===")
    print(f"TP (acertados):     {tp:,}")
    print(f"FP (falsas alarmas): {fp:,}")
    print(f"FN (perdidos):      {fn:,}")
    print(f"")
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # Breakdown por superfamilia (de la verdad)
    print(f"\n=== Recall por superfamilia TAIR10 ===")
    by_family_true = defaultdict(int)
    by_family_found = defaultdict(int)
    for name, iou_v, status in true_matches:
        tokens = name.split("|")
        superfam = tokens[1] if len(tokens) > 1 else "Unknown"
        by_family_true[superfam] += 1
        if status == "TP":
            by_family_found[superfam] += 1

    rows_fam = []
    for sf, n_t in sorted(by_family_true.items(), key=lambda x: -x[1]):
        n_f = by_family_found.get(sf, 0)
        rec = n_f / n_t if n_t else 0.0
        rows_fam.append((sf, n_t, n_f, rec))
        print(f"  {sf:25s}: {n_f:,} / {n_t:,}  (recall={rec:.3f})")

    # Reporte markdown
    lines = [
        "# Test \"ORO\" · Arabidopsis thaliana",
        "",
        "**Verdad externa**: `TAIR10_Transposable_Elements.txt` (anotación curada TAIR, 31,189 TEs).",
        "",
        "**Predicción evaluada**: `annotations/arabidopsis.bed` (RM + biblioteca PanTEon de Arabidopsis).",
        "",
        f"**Criterio de match**: IoU ≥ {IOU_THRESHOLD} con el elemento verdadero.",
        "",
        "## Métricas globales",
        "",
        f"| Métrica | Valor |",
        f"|---|---:|",
        f"| **Precisión** | **{precision:.4f}** |",
        f"| **Recall** | **{recall:.4f}** |",
        f"| **F1** | **{f1:.4f}** |",
        f"| TP | {tp:,} |",
        f"| FP | {fp:,} |",
        f"| FN | {fn:,} |",
        f"| TEs verdad (TAIR10) | {n_true:,} |",
        f"| TEs predicción (PanTEon) | {n_pred:,} |",
        "",
        "## Recall por superfamilia TAIR10",
        "",
        "| Superfamilia | Verdad | Encontrados | Recall |",
        "|---|---:|---:|---:|",
    ]
    for sf, n_t, n_f, rec in rows_fam:
        lines.append(f"| {sf} | {n_t:,} | {n_f:,} | {rec:.3f} |")

    lines += [
        "",
        "## Interpretación",
        "",
        f"- **Recall de {recall*100:.1f}%**: de los TEs curados TAIR10, encontramos {tp:,} de {n_true:,}.",
        f"- **Precisión de {precision*100:.1f}%**: de lo que marcamos, {tp:,} de {n_pred:,} son TE reales (según TAIR).",
        f"- **F1 global: {f1:.4f}**.",
        "",
        "## Comparación con el modelo previo",
        "",
        "| Comparación | F1 |",
        "|---|---:|",
        "| **ANOTACIÓN** (RM+PanTEon vs TAIR10) | " + f"**{f1:.3f}**" + " |",
        "| MODELO SegNT (previo) en test balanceado PanTEon | 0.670 |",
        "| MODELO SegNT (previo) en Arabidopsis chr5 real | 0.080 |",
        "",
        "Esto nos dice qué tan buena es la **ANOTACIÓN** sobre la que entrenaremos.",
        "Si F1-anotación es alto, el ground truth es sólido y el modelo nuevo deberá acercarse.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")

    out_json = ROOT / "docs" / "test_oro_arabidopsis.json"
    out_json.write_text(json.dumps({
        "species": "arabidopsis",
        "true_source": "TAIR10_Transposable_Elements.txt",
        "n_true": n_true,
        "n_pred": n_pred,
        "iou_threshold": IOU_THRESHOLD,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "by_superfamily": {sf: {"n_true": nt, "n_found": nf, "recall": r}
                           for sf, nt, nf, r in rows_fam},
    }, indent=2))

    print(f"\n[md]   {OUT_MD}")
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
