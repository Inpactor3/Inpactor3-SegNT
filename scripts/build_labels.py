"""
Verifica anotaciones Inpactor2 sobre el FASTA y produce un pequeño reporte
por cromosoma (ventanas totales vs con LTR-RT).

Uso:
    python scripts/build_labels.py \\
        --genome data/raw/TAIR10.fasta \\
        --annotations data/raw/Inpactor2_predictions.tab \\
        --window-size 50000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inpactor3_segnt.dataset import parse_fasta, parse_inpactor2_tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--window-size", type=int, default=50_000)
    args = ap.parse_args()

    seqs = parse_fasta(args.genome)
    ann = parse_inpactor2_tab(args.annotations)

    print(f"{'Cromosoma':<12} {'Bases':>12} {'Ventanas':>10} {'Con LTR-RT':>12}"
          f" {'Elementos':>10}")
    print("-" * 60)
    total_w = total_p = total_e = 0
    for scaf, seq in seqs.items():
        L = len(seq)
        n_win = (L + args.window_size - 1) // args.window_size
        annots = ann.get(scaf, [])
        # ventanas con elemento
        pos_w = set()
        for a in annots:
            w = a.start // args.window_size
            pos_w.add(w)
        print(f"{scaf:<12} {L:>12,} {n_win:>10} {len(pos_w):>12} {len(annots):>10}")
        total_w += n_win
        total_p += len(pos_w)
        total_e += len(annots)
    print("-" * 60)
    print(f"{'TOTAL':<12} {'':>12} {total_w:>10} {total_p:>12} {total_e:>10}")

    # linajes
    lin_counts = Counter()
    for hits in ann.values():
        for a in hits:
            lin_counts[a.cls] += 1
    print(f"\n[linajes] {dict(lin_counts)}")


if __name__ == "__main__":
    main()
