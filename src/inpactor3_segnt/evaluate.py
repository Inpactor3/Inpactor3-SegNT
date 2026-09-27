"""
Evaluación al estilo del informe: F1 por ventana + confusión de linaje + IoU.

Divide el scaffold objetivo en ventanas de 50 kb (como Inpactor2),
compara predicciones vs verdad ventana-a-ventana, y reporta:
  - F1 de presencia por ventana
  - Matriz de confusión binaria (LTR-RT sí/no)
  - Matriz de confusión de linaje (si el .tab trae linaje)
  - IoU medio de coordenadas para ventanas TP
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .dataset import INPACTOR2_LINEAGES, parse_inpactor2_tab


def windows_from_length(length: int, size: int = 50_000) -> list[tuple[int, int]]:
    ws = []
    for s in range(0, length, size):
        ws.append((s, min(s + size, length)))
    return ws


def annotate_windows(annotations: list, windows: list[tuple[int, int]]) -> list[int]:
    """Devuelve 1 si la ventana solapa con al menos una anotación."""
    labels = [0] * len(windows)
    for i, (ws, we) in enumerate(windows):
        for ann in annotations:
            if ann.start < we and ann.end > ws:
                labels[i] = 1
                break
    return labels


def lineage_by_window(annotations: list, windows: list[tuple[int, int]]) -> list[int]:
    """
    Por cada ventana devuelve el id de linaje del elemento con mayor solape.
    0 si no hay ningún elemento.
    """
    labels = [0] * len(windows)
    for i, (ws, we) in enumerate(windows):
        best_over = 0
        best_cls = 0
        for ann in annotations:
            s = max(ann.start, ws)
            e = min(ann.end, we)
            over = max(0, e - s)
            if over > best_over:
                best_over = over
                best_cls = ann.cls
        labels[i] = best_cls
    return labels


def iou_1d(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """IoU entre dos intervalos 1D."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def match_iou(truth: list, pred: list, iou_thr: float = 0.5) -> list[float]:
    """
    Para cada anotación verdadera, busca la predicción con mayor IoU.
    Devuelve la lista de IoUs (0 si no hay match).
    """
    ious = []
    used = set()
    for t in truth:
        best_iou = 0.0
        best_j = -1
        for j, p in enumerate(pred):
            if j in used:
                continue
            v = iou_1d(t.start, t.end, p.start, p.end)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
        ious.append(best_iou)
    return ious


def genome_length(fasta: Path, scaffold: str) -> int:
    L = 0
    in_scaf = False
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                if in_scaf and L > 0:
                    return L
                in_scaf = (cur == scaffold)
                if in_scaf:
                    L = 0
            elif in_scaf:
                L += len(line.strip())
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True,
                    help="Inpactor3_predictions.tab de nuestra red")
    ap.add_argument("--truth", type=Path, required=True,
                    help="Inpactor2_predictions.tab (ground truth)")
    ap.add_argument("--scaffold", type=str, required=True,
                    help="Scaffold a evaluar (ej: 'Chr5')")
    ap.add_argument("--genome", type=Path, required=True,
                    help="FASTA del genoma para conocer la longitud del scaffold")
    ap.add_argument("--window-size", type=int, default=50_000)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--multiclass", action="store_true",
                    help="Evaluar por linaje (usa INPACTOR2_LINEAGES)")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    mode = "lineage" if args.multiclass else "binary"
    truth = parse_inpactor2_tab(args.truth, mode=mode).get(args.scaffold, [])
    pred = parse_inpactor2_tab(args.pred, mode=mode).get(args.scaffold, [])
    L = genome_length(args.genome, args.scaffold)
    print(f"[eval] scaffold {args.scaffold} · {L:,} bp · "
          f"truth={len(truth)} · pred={len(pred)}")

    windows = windows_from_length(L, args.window_size)
    y_true = annotate_windows(truth, windows)
    y_pred = annotate_windows(pred, windows)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # IoU coordenadas
    ious = match_iou(truth, pred, iou_thr=args.iou_thr)
    iou_mean = sum(ious) / len(ious) if ious else 0.0
    iou_matched = [i for i in ious if i >= args.iou_thr]
    match_rate = len(iou_matched) / len(ious) if ious else 0.0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        f.write(f"# Evaluación cromosoma {args.scaffold}\n\n")
        f.write(f"- Longitud: **{L:,}** bp\n")
        f.write(f"- Ventanas de {args.window_size:,} bp: **{len(windows)}**\n")
        f.write(f"- Ground truth (Inpactor2): {len(truth)} elementos, "
                f"{sum(y_true)} ventanas con LTR-RT\n")
        f.write(f"- Predicciones (Inpactor3-SegNT): {len(pred)} elementos, "
                f"{sum(y_pred)} ventanas marcadas\n\n")

        f.write("## Matriz de confusión (ventanas)\n\n")
        f.write("|              | Predijo vacío | Predijo LTR-RT |\n")
        f.write("|--------------|--------------:|---------------:|\n")
        f.write(f"| **Vacío ({tn+fp})**    | {tn} | {fp} |\n")
        f.write(f"| **LTR-RT ({tp+fn})**   | {fn} | {tp} |\n\n")
        f.write("## Métricas de presencia (ventana)\n\n")
        f.write(f"- **Recall**     = {rec:.3f}  ({tp}/{tp+fn})\n")
        f.write(f"- **Precisión**  = {prec:.3f}  ({tp}/{tp+fp})\n")
        f.write(f"- **F1**         = {f1:.3f}\n\n")

        f.write(f"## Precisión de coordenadas (IoU 1D, umbral={args.iou_thr})\n\n")
        f.write(f"- IoU medio (por elemento verdadero): **{iou_mean:.3f}**\n")
        f.write(f"- Elementos con IoU ≥ {args.iou_thr}: "
                f"**{len(iou_matched)}/{len(ious)}** ({match_rate*100:.1f}%)\n\n")

        # Matriz de confusión de linaje (si multiclass)
        if args.multiclass and len(truth) > 0:
            lin_true = lineage_by_window(truth, windows)
            lin_pred = lineage_by_window(pred, windows)
            f.write("## Confusión de linaje por ventana positiva\n\n")
            # solo ventanas donde ground truth tiene un LTR-RT
            conf: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
            for lt, lp in zip(lin_true, lin_pred):
                if lt != 0:
                    conf[lt][lp] += 1

            # Encabezado con nombres cortos
            from .dataset import LINEAGE_ID_TO_NAME
            all_pred = sorted({lp for row in conf.values() for lp in row})
            f.write("| ↓verdad \\ pred → | " +
                    " | ".join(LINEAGE_ID_TO_NAME.get(i, str(i)) for i in all_pred) +
                    " |\n")
            f.write("|" + "---|" * (len(all_pred) + 1) + "\n")
            for lt in sorted(conf):
                row = [str(conf[lt][lp]) for lp in all_pred]
                f.write(f"| **{LINEAGE_ID_TO_NAME.get(lt, lt)}** | " +
                        " | ".join(row) + " |\n")
            f.write("\n")
            correct = sum(conf[lt][lt] for lt in conf)
            total = sum(sum(row.values()) for row in conf.values())
            acc = correct / total if total else 0.0
            f.write(f"- **Accuracy de linaje** (sobre ventanas TP): "
                    f"**{acc:.3f}** ({correct}/{total})\n\n")

        f.write("## Referencia (informe interno, chr5 Arabidopsis)\n\n")
        f.write("| Enfoque | F1 |\n|---|---|\n")
        f.write("| Fondo aleatorio uniforme | 0-7% |\n")
        f.write("| Fondo otro transposón | 7.5% |\n")
        f.write("| Fondo genoma real (informe intento 3) | **26.3%** |\n")
        f.write(f"| **Este modelo (SegNT)** | **{f1*100:.1f}%** |\n")

    print(f"\n[out ] {args.report}")
    print(f"F1_window = {f1:.3f}  (P={prec:.3f}  R={rec:.3f})")
    print(f"IoU medio = {iou_mean:.3f}   ·   match rate (IoU≥{args.iou_thr}) = {match_rate:.3f}")


if __name__ == "__main__":
    main()
