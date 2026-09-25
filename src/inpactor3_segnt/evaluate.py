"""
Evaluación al estilo del informe: F1 por ventana + confusión.

Divide el scaffold objetivo en ventanas de 50 kb (como Inpactor2),
compara predicciones vs verdad ventana-a-ventana, y reporta.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .dataset import parse_inpactor2_tab


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


def genome_length(fasta: Path, scaffold: str) -> int:
    L = 0
    in_scaf = False
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                in_scaf = (cur == scaffold)
                if in_scaf:
                    L = 0
                elif L > 0:
                    return L
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
                    help="Scaffold a evaluar (ej: '5')")
    ap.add_argument("--genome", type=Path, required=True,
                    help="FASTA del genoma para conocer la longitud del scaffold")
    ap.add_argument("--window-size", type=int, default=50_000)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    truth = parse_inpactor2_tab(args.truth).get(args.scaffold, [])
    pred = parse_inpactor2_tab(args.pred).get(args.scaffold, [])
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
        f.write("## Métricas\n\n")
        f.write(f"- **Recall**     = {rec:.3f}  ({tp}/{tp+fn})\n")
        f.write(f"- **Precisión**  = {prec:.3f}  ({tp}/{tp+fp})\n")
        f.write(f"- **F1**         = {f1:.3f}\n")
        f.write(f"\n## Referencia (informe interno, chr5 Arabidopsis)\n\n")
        f.write("| Enfoque | F1 |\n|---|---|\n")
        f.write("| Fondo aleatorio uniforme | 0-7% |\n")
        f.write("| Fondo otro transposón | 7.5% |\n")
        f.write("| Fondo genoma real (informe intento 3) | **26.3%** |\n")
        f.write(f"| **Este modelo (SegNT)** | **{f1*100:.1f}%** |\n")

    print(f"\n[out ] {args.report}")
    print(f"F1_window = {f1:.3f}  (P={prec:.3f}  R={rec:.3f})")


if __name__ == "__main__":
    main()
