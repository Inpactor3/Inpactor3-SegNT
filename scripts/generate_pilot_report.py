"""Genera el reporte del piloto RepeatMasker + PanTEon por especie.

Compara:
- % del genoma cubierto por TEs (vs literatura)
- Distribucion por clase/orden
- Numero de elementos encontrados
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "docs" / "piloto_rm_reporte.md"

# Expectativa de literatura para validar la anotacion
LITERATURE_TE_PCT = {
    "arabidopsis":  15.0,
    "scerevisiae":   3.0,
    "celegans":     17.0,
    "brachypodium": 22.0,
    "neurospora":   10.0,
    "drosophila":   20.0,
    "spombe":        1.0,
    "aspergillus":   5.0,
    "oryza":        22.0,
}

KINGDOM = {
    "arabidopsis":  "Plantae",
    "brachypodium": "Plantae",
    "oryza":        "Plantae",
    "scerevisiae":  "Fungi",
    "neurospora":   "Fungi",
    "spombe":       "Fungi",
    "aspergillus":  "Fungi",
    "celegans":     "Animalia",
    "drosophila":   "Animalia",
}

GENOME_PATHS = {
    "arabidopsis":  ROOT / "data" / "tair10_full"      / "TAIR10.fasta",
    "scerevisiae":  ROOT / "data" / "pilot_genomes"    / "scerevisiae.fasta",
    "celegans":     ROOT / "data" / "pilot_genomes"    / "celegans.fasta",
    "brachypodium": ROOT / "data" / "pilot_genomes"    / "brachypodium.fasta",
    "neurospora":   ROOT / "data" / "pilot_genomes"    / "neurospora.fasta",
    "drosophila":   ROOT / "data" / "pilot_genomes"    / "drosophila.fasta",
    "spombe":       ROOT / "data" / "pilot_genomes"    / "spombe.fasta",
    "aspergillus":  ROOT / "data" / "pilot_genomes"    / "aspergillus.fasta",
    "oryza":        ROOT / "data" / "validation_genomes" / "Oryza_sativa.fasta",
}


def genome_length(fa: Path) -> int:
    total = 0
    with fa.open() as f:
        for line in f:
            if not line.startswith(">"):
                total += len(line.strip())
    return total


def parse_rm_out(out_file: Path):
    """Lee el .out de RepeatMasker.

    Formato:
       SW  div del ins  query  qstart qend  (qleft)  C?  repeat  class/family
    """
    rows = []
    if not out_file.exists():
        return rows
    with out_file.open() as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 11 or not s[0].replace(".", "").isdigit():
                continue
            try:
                qstart = int(s[5])
                qend = int(s[6])
                rep = s[9]  # repeat name
                cls = s[10]  # class/family
                rows.append((qstart, qend, rep, cls))
            except (ValueError, IndexError):
                continue
    return rows


def summarize(name: str) -> dict:
    out_dir = ROOT / "results" / "rm_pilots" / name
    genome = GENOME_PATHS[name]

    # Buscar archivo .out
    out_candidates = list(out_dir.glob("*.out"))
    out_file = out_candidates[0] if out_candidates else None

    glen = genome_length(genome) if genome.exists() else 0
    rows = parse_rm_out(out_file) if out_file else []

    # Sumar longitudes TE
    te_total_bp = sum(end - start + 1 for start, end, *_ in rows)
    te_pct = 100.0 * te_total_bp / glen if glen else 0.0

    # Distribucion por clase
    cls_counter = Counter(r[3] for r in rows)
    cls_bp = Counter()
    for start, end, _, cls in rows:
        cls_bp[cls] += end - start + 1

    return {
        "name": name,
        "genome_bp": glen,
        "n_elements": len(rows),
        "te_total_bp": te_total_bp,
        "te_pct": te_pct,
        "expected_pct": LITERATURE_TE_PCT.get(name, None),
        "cls_counter": cls_counter,
        "cls_bp": cls_bp,
        "out_file": str(out_file) if out_file else None,
    }


def status_mark(obs: float, exp: float) -> str:
    if exp is None:
        return "N/A"
    diff = abs(obs - exp)
    if diff < exp * 0.3:
        return "OK (dentro de 30%)"
    if diff < exp * 0.5:
        return "WARN (desviacion 30-50%)"
    return "FAIL (desviacion >50%)"


def main():
    # Solo incluir los que tienen output real
    all_species = ["arabidopsis", "brachypodium", "oryza",
                   "scerevisiae", "neurospora", "spombe", "aspergillus",
                   "celegans", "drosophila"]
    results = {}
    for sp in all_species:
        r = summarize(sp)
        # Solo incluir si tiene elementos (RM terminó)
        if r["out_file"]:
            results[sp] = r

    lines = []
    lines.append("# Reporte piloto · RepeatMasker + biblioteca PanTEon")
    lines.append("")
    lines.append(f"Validacion de la anotacion con RepeatMasker + PanTEon sobre "
                 f"**{len(results)} especies** (2 por reino). Criterio: el %TE "
                 "encontrado debe estar dentro de 30% del valor reportado en "
                 "literatura.")
    lines.append("")
    lines.append("## Resumen general (ordenado por reino)")
    lines.append("")
    lines.append("| Reino | Especie | Genoma | TEs enc. | %TE obs. | %TE esp. (lit) | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for name in sorted(results.keys(), key=lambda s: (KINGDOM.get(s, "?"), s)):
        r = results[name]
        glen_mb = r["genome_bp"] / 1e6
        obs = r["te_pct"]
        exp = r["expected_pct"]
        exp_s = f"{exp:.1f}%" if exp else "?"
        obs_s = f"{obs:.2f}%"
        status = status_mark(obs, exp)
        k = KINGDOM.get(name, "?")
        lines.append(f"| {k} | {name} | {glen_mb:.1f} Mb | {r['n_elements']:,} | "
                     f"{obs_s} | {exp_s} | {status} |")

    for name, r in results.items():
        lines.append("")
        lines.append(f"## Detalle · {name}")
        lines.append("")
        lines.append(f"- Genoma: `{GENOME_PATHS[name]}`")
        lines.append(f"- Longitud: {r['genome_bp']:,} bp")
        lines.append(f"- Elementos detectados: {r['n_elements']:,}")
        lines.append(f"- Total bp cubiertos por TE: {r['te_total_bp']:,}")
        lines.append(f"- %TE observado: **{r['te_pct']:.2f}%**")
        if r["expected_pct"]:
            lines.append(f"- %TE esperado (literatura): {r['expected_pct']:.1f}%")
        lines.append("")
        lines.append("### Top 10 clases/familias (por numero de elementos)")
        lines.append("")
        lines.append("| Clase/Familia | n elementos | bp cubiertos |")
        lines.append("|---|---:|---:|")
        for cls, n in r["cls_counter"].most_common(10):
            bp = r["cls_bp"][cls]
            lines.append(f"| {cls} | {n:,} | {bp:,} |")

    lines.append("")
    lines.append("## Conclusion para el equipo")
    lines.append("")
    ok_count = sum(1 for r in results.values()
                   if r["expected_pct"] and
                   abs(r["te_pct"] - r["expected_pct"]) < r["expected_pct"] * 0.3)
    lines.append(f"- {ok_count}/3 pilotos pasan el criterio (±30% vs literatura)")
    lines.append(f"- La biblioteca PanTEon filtrada por especie {'funciona' if ok_count >= 2 else 'tiene problemas'}")
    lines.append(f"- Decision: {'proceder con la anotacion masiva' if ok_count >= 2 else 'revisar la biblioteca o parametros antes de escalar'}")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[md] {OUT_MD}")

    # Resumen por consola
    print("\n=== Resumen ===")
    for name, r in results.items():
        exp = r["expected_pct"]
        print(f"{name:14s}: {r['n_elements']:6,} elementos  "
              f"{r['te_pct']:6.2f}% TE  (literatura: {exp}%)")


if __name__ == "__main__":
    main()
