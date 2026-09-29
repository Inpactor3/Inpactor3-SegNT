"""
Corre un modelo entrenado sobre 4 genomas de referencia y compara
el % de LTR-RT anotado vs el valor biológico esperado.

Genomas de validación:
  - Arabidopsis thaliana (~10% LTR)
  - Oryza sativa (arroz, ~22.5%)
  - Ananas comosus (piña, ~23%)
  - Zea mays (maíz, ~75%)

Uso:
    python scripts/validate_on_genomes.py \\
        --model-type segnt \\
        --checkpoint models/best.pt \\
        --genomes-dir data/validation_genomes \\
        --report results/validation_report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

REFERENCE = {
    "Arabidopsis_thaliana": {"expected_pct": 10.0, "source": "literatura"},
    "Oryza_sativa":         {"expected_pct": 22.5, "source": "paper Inpactor 2"},
    "Ananas_comosus":       {"expected_pct": 23.0, "source": "Orozco-Arias 2018"},
    "Zea_mays":             {"expected_pct": 75.0, "source": "literatura"},
}


def parse_fasta_len(path: Path) -> dict[str, int]:
    lens: dict[str, int] = {}
    cur, n = None, 0
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur is not None:
                    lens[cur] = n
                cur = line[1:].split()[0]
                n = 0
            else:
                n += len(line)
        if cur is not None:
            lens[cur] = n
    return lens


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def load_predictions_tab(tab: Path) -> dict[str, list[tuple[int, int]]]:
    """Lee .tab formato Inpactor y devuelve {scaffold: [(start, end)]}."""
    by_scaf: dict[str, list[tuple[int, int]]] = {}
    if not tab.exists():
        return by_scaf
    with open(tab) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 3:
                continue
            scaf = cols[0]
            s, e = int(cols[1]), int(cols[2])
            by_scaf.setdefault(scaf, []).append((s, e))
    return by_scaf


def genome_ltr_pct(fasta: Path, pred_tab: Path) -> tuple[float, int, int]:
    """Devuelve (pct, total_bp, ltr_bp)."""
    scaf_lens = parse_fasta_len(fasta)
    total_bp = sum(scaf_lens.values())
    preds = load_predictions_tab(pred_tab)
    ltr_bp = 0
    for scaf, intervals in preds.items():
        merged = merge_intervals(intervals)
        ltr_bp += sum(e - s for s, e in merged)
    return (100 * ltr_bp / max(total_bp, 1), total_bp, ltr_bp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", required=True, type=Path,
                    help="Carpeta con {Species}_predictions.tab por especie")
    ap.add_argument("--genomes-dir", required=True, type=Path)
    ap.add_argument("--model-name", default="Inpactor3-SegNT")
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sp, info in REFERENCE.items():
        fasta = args.genomes_dir / f"{sp}.fasta"
        pred = args.predictions_dir / f"{sp}_predictions.tab"
        if not fasta.exists():
            print(f"[skip] {sp}: falta genoma {fasta}")
            rows.append((sp, info["expected_pct"], None, None, None))
            continue
        pct, total_bp, ltr_bp = genome_ltr_pct(fasta, pred)
        rows.append((sp, info["expected_pct"], pct, total_bp, ltr_bp))
        print(f"[{sp:25s}] esperado {info['expected_pct']:5.1f}% · "
              f"obtenido {pct:5.1f}% (Δ {pct - info['expected_pct']:+.1f} pp)")

    with open(args.report, "w") as f:
        f.write(f"# Validación de {args.model_name} sobre 4 genomas de referencia\n\n")
        f.write("Comparación entre el porcentaje del genoma anotado como LTR-RT\n"
                "por el modelo, y el valor biológico esperado según la literatura.\n\n")
        f.write("| Especie | Esperado | Obtenido | Diferencia | Fuente esperado |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for sp, exp, got, tot, ltr in rows:
            source = REFERENCE[sp]["source"]
            if got is None:
                f.write(f"| {sp} | {exp:.1f}% | — | — | {source} |\n")
            else:
                delta = got - exp
                sig = "✅" if abs(delta) < 5 else ("⚠️" if abs(delta) < 15 else "❌")
                f.write(f"| {sp} | {exp:.1f}% | {got:.1f}% | {delta:+.1f} pp {sig} | {source} |\n")

        # Métricas agregadas
        valid = [(exp, got) for _, exp, got, _, _ in rows if got is not None]
        if valid:
            mae = sum(abs(g - e) for e, g in valid) / len(valid)
            rmse = (sum((g - e) ** 2 for e, g in valid) / len(valid)) ** 0.5
            f.write(f"\n## Métricas agregadas\n\n")
            f.write(f"- **MAE** (error absoluto medio): {mae:.2f} pp\n")
            f.write(f"- **RMSE**: {rmse:.2f} pp\n")
            f.write(f"- **Interpretación**:\n")
            f.write(f"  - ✅ Δ < 5 pp: dentro del margen biológico razonable\n")
            f.write(f"  - ⚠️ 5-15 pp: aceptable con reservas\n")
            f.write(f"  - ❌ > 15 pp: modelo con sesgo importante\n")

    print(f"\n[out ] {args.report}")


if __name__ == "__main__":
    raise SystemExit(main())
