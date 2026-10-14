"""Genera tabla de géneros PanTEon para discutir exclusión por homología.

Regla de Simón: si una especie va a test "oro", TAMBIEN hay que excluir del
train todas las especies de su MISMO GENERO (y familia si hay muchas cercanas)
para evitar fuga por homologia.

Salida:
    docs/panteon_genera.md   -> tabla markdown lista para revisar
    docs/panteon_genera.csv  -> para procesar programaticamente
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "raw" / "PanTEon_Database_metadata_v1.6.2.csv"
OUT_MD = ROOT / "docs" / "panteon_genera.md"
OUT_CSV = ROOT / "docs" / "panteon_genera.csv"

by_genus = defaultdict(list)  # genus -> list[(species, kingdom, te_count)]
by_kingdom = defaultdict(int)

with META.open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        sp = row["Species"]
        kingdom = row["Kingdom"]
        try:
            n = int(row["TE_count"])
        except Exception:
            n = 0
        # El genero es la primera palabra del nombre binomial
        genus = sp.split()[0] if sp else "Unknown"
        by_genus[genus].append((sp, kingdom, n))
        by_kingdom[kingdom] += 1

# Ordenar por reino + numero de especies por genero
rows = []
for genus, specs in by_genus.items():
    kingdoms = {s[1] for s in specs}
    k = next(iter(kingdoms)) if len(kingdoms) == 1 else "Mixed"
    total_sp = len(specs)
    total_te = sum(s[2] for s in specs)
    rows.append({
        "genus": genus,
        "kingdom": k,
        "n_species": total_sp,
        "total_consensos": total_te,
        "species_list": "; ".join(s[0] for s in specs),
    })

# Ordenar: por reino, luego por n_especies desc
rows.sort(key=lambda r: (r["kingdom"], -r["n_species"]))

# CSV
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["genus", "kingdom", "n_species",
                                       "total_consensos", "species_list"])
    w.writeheader()
    w.writerows(rows)

# Markdown
lines = [
    "# Tabla de géneros PanTEon",
    "",
    f"**Total especies**: {sum(len(v) for v in by_genus.values())}  ",
    f"**Total géneros**: {len(by_genus)}  ",
    f"**Por reino**: Animalia={by_kingdom['Animalia']} · "
    f"Fungi={by_kingdom['Fungi']} · "
    f"Plantae={by_kingdom['Plantae']}",
    "",
    "## Géneros con ≥3 especies (candidatos a exclusión completa si uno va a test)",
    "",
    "Simón: *\"excluir del train las especies de test oro **y todo su género** "
    "(y familia si hay muchas cercanas), para evitar fuga por homología\"*",
    "",
    "| Reino | Género | #especies | #consensos | Especies |",
    "|---|---|---:|---:|---|",
]
for r in rows:
    if r["n_species"] >= 3:
        spec_short = r["species_list"][:120] + "..." if len(r["species_list"]) > 120 else r["species_list"]
        lines.append(
            f"| {r['kingdom']} | **{r['genus']}** | {r['n_species']} | "
            f"{r['total_consensos']:,} | {spec_short} |"
        )

lines.append("")
lines.append("## Géneros con solo 1-2 especies (exclusión por familia opcional)")
lines.append("")
lines.append("Tabla completa en [panteon_genera.csv](panteon_genera.csv).")
lines.append("")
lines.append("## Candidatos directos de test \"oro\" (sugerencia Simón)")
lines.append("")
lines.append("Si decidimos estas, hay que excluir del train sus **géneros completos**:")
lines.append("")
test_candidates = [
    ("Oryza sativa", "Oryza"),
    ("Zea mays", "Zea"),
    ("Arabidopsis thaliana", "Arabidopsis"),
    ("Drosophila melanogaster", "Drosophila"),
    ("Caenorhabditis elegans", "Caenorhabditis"),
    ("Neurospora crassa", "Neurospora"),
    ("Saccharomyces cerevisiae", "Saccharomyces"),
]
lines.append("| Especie oro | Género a excluir | # especies del género en PanTEon |")
lines.append("|---|---|---:|")
for sp, g in test_candidates:
    n_sp = len(by_genus.get(g, []))
    lines.append(f"| {sp} | **{g}** | {n_sp} |")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"[md]  {OUT_MD}")
print(f"[csv] {OUT_CSV}")
print()
print("=== Top géneros por #especies ===")
for r in rows[:10]:
    print(f"  {r['kingdom']:10s}  {r['genus']:25s}  "
          f"{r['n_species']:3d} especies  "
          f"{r['total_consensos']:7,d} consensos")
