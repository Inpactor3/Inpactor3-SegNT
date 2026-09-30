"""
Reporte comparativo final: Inpactor 2 (original) vs Inpactor 2 (re-entrenado PanTEon)
vs SegNT sobre los 4 genomas de validación.

Compara para cada modelo:
  - % del genoma anotado como LTR-RT
  - Número de anotaciones
  - Distribución por linaje
  - Diferencia vs valor biológico esperado

Uso:
    python scripts/final_comparison.py \\
        --results-dir results \\
        --genomes-dir data/validation_genomes \\
        --out results/comparison_final.md
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


REFERENCE = {
    "Arabidopsis_thaliana": {"expected_pct": 10.0, "tair10_file": "TAIR10.fasta",
                              "source": "literatura"},
    "Oryza_sativa":         {"expected_pct": 22.5, "source": "paper Inpactor 2"},
    "Ananas_comosus":       {"expected_pct": 23.0, "source": "Orozco-Arias 2018"},
    "Zea_mays":             {"expected_pct": 75.0, "source": "literatura"},
}


def genome_size(fasta: Path) -> int:
    total = 0
    with open(fasta) as f:
        for line in f:
            if not line.startswith(">"):
                total += len(line.strip())
    return total


def merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [ivs[0]]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def load_predictions(tab: Path) -> tuple[int, int, Counter]:
    """Devuelve (n_anotaciones, bp_cubiertos, contador_linaje)."""
    if not tab.exists():
        return 0, 0, Counter()
    intervals: dict[str, list[tuple[int, int]]] = {}
    linajes = Counter()
    n = 0
    with open(tab) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue
            scaf, s, e, _, lin = cols[0], int(cols[1]), int(cols[2]), cols[3], cols[4]
            intervals.setdefault(scaf, []).append((s, e))
            linajes[lin] += 1
            n += 1
    total_bp = sum(sum(e - s for s, e in merge_intervals(ivs))
                   for ivs in intervals.values())
    return n, total_bp, linajes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/comparison_final.md"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ROOT = Path(__file__).resolve().parents[1]

    lines = ["# Reporte comparativo final\n"]
    lines.append("Comparación de tres modelos sobre 4 genomas de referencia:\n")
    lines.append("- **Inpactor 2 original** (entrenado con InpactorDB, solo plantas, 70k ventanas)")
    lines.append("- **Inpactor 2 re-entrenado** (entrenado con PanTEon cross-kingdom, 9.3k ventanas piloto)")
    lines.append("- **SegNT** (Transformer + cabeza segmentación, en Colab)\n")

    lines.append("## Tabla resumen: % del genoma anotado como LTR-RT\n")
    lines.append("| Genoma | Bases | Esperado bio | I2 original | I2 re-entrenado | SegNT |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    all_data = []
    for sp, info in REFERENCE.items():
        # Buscar genoma
        candidates = [
            ROOT / "data/validation_genomes" / f"{sp}.fasta",
            ROOT / "data/tair10_full/TAIR10.fasta" if sp == "Arabidopsis_thaliana" else None,
            ROOT / "data/common_dataset/genomes" / f"{sp}.fasta",
        ]
        genome = next((c for c in candidates if c and c.exists()), None)
        if not genome:
            print(f"[skip] {sp}: sin genoma en disco")
            continue
        total_bp = genome_size(genome)

        # Predicciones de cada modelo
        i2_orig = ROOT / "data/tair10_full/inpactor2_out/Inpactor2_predictions.tab" if sp == "Arabidopsis_thaliana" else ROOT / f"results/inpactor2_orig/{sp}/Inpactor2_predictions.tab"
        i2_teon = ROOT / f"results/inpactor2_teon/{sp.split('_')[0]}/Inpactor2_predictions.tab"
        segnt = ROOT / f"results/segnt/{sp}_predictions.tab"

        n1, bp1, lin1 = load_predictions(i2_orig)
        n2, bp2, lin2 = load_predictions(i2_teon)
        n3, bp3, lin3 = load_predictions(segnt)

        pct1 = 100 * bp1 / total_bp if bp1 else 0
        pct2 = 100 * bp2 / total_bp if bp2 else 0
        pct3 = 100 * bp3 / total_bp if bp3 else 0

        all_data.append({
            "sp": sp, "total_bp": total_bp, "exp": info["expected_pct"],
            "i2_orig": (n1, pct1, lin1),
            "i2_teon": (n2, pct2, lin2),
            "segnt":   (n3, pct3, lin3),
        })

        line = f"| {sp} | {total_bp:,} | {info['expected_pct']:.1f}% |"
        for n, pct in [(n1, pct1), (n2, pct2), (n3, pct3)]:
            if n == 0:
                line += " — |"
            else:
                delta = pct - info["expected_pct"]
                sig = "✅" if abs(delta) < 5 else ("⚠️" if abs(delta) < 15 else "❌")
                line += f" {pct:.2f}% ({n}) {sig} |"
        lines.append(line)

    # Detalle por linaje
    lines.append("\n## Detalle: distribución de linajes por modelo\n")
    for d in all_data:
        lines.append(f"### {d['sp']}\n")
        lines.append("| Linaje | I2 original | I2 re-entrenado | SegNT |")
        lines.append("|---|---:|---:|---:|")
        all_linajes = set(d["i2_orig"][2]) | set(d["i2_teon"][2]) | set(d["segnt"][2])
        for lin in sorted(all_linajes):
            row = f"| {lin} |"
            for _, _, c in [d["i2_orig"], d["i2_teon"], d["segnt"]]:
                row += f" {c.get(lin, 0)} |"
            lines.append(row)
        lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
