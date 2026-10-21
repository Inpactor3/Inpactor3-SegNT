"""Muestreo aleatorio estratificado por longitud (pb) para seleccionar
especies de los 3 reinos para el dataset.

Metodologia (segun Simon):
    1. Solo reinos Animales, Plantas, Hongos (los demas se excluyen).
    2. Estratificar por LONGITUD TOTAL (pb) en cada reino.
    3. Muestrear aleatoriamente dentro de cada estrato.
    4. Conservar la distribucion de longitud entre train/val/test.

Estratificacion por cuartiles de longitud dentro de cada reino:
    Q1: pequenas    (<= cuartil 25%)
    Q2: medianas    (25-50%)
    Q3: medianas-grandes (50-75%)
    Q4: grandes     (>= cuartil 75%)

Luego dentro de cada reino, dividimos:
    70% train / 15% val / 15% test  (proporcional por estrato)
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

PATH = Path("/home/cami/Desktop/ProyectoIntegrador/resumen_especies.csv")
OUT = Path("/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/docs")

# Numero objetivo por reino (configurable)
N_PER_KINGDOM = {
    "Plantas": 150,
    "Animales": 150,
    "Hongos": 150,
}
# Reinos a excluir (Simon: "los demas grupos quedan fuera")
EXCLUDE_KINGDOMS = {"Ambiguo", "No asignado", "Otros eucariotas", "Virus"}

# Proporciones de split
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15  # (=1 - train - val)


def load_species():
    """Cargar especies del CSV con filtros basicos."""
    data = defaultdict(list)
    with PATH.open() as f:
        for row in csv.DictReader(f):
            reino = row["reino"]
            if reino in EXCLUDE_KINGDOMS:
                continue
            try:
                pb = int(row["pb"])
                n_seq = int(row["secuencias"])
                n_sf = int(row["n_superfamilias"])
            except ValueError:
                continue
            # Filtro minimo: al menos 1 kb de secuencia total
            if pb < 1000:
                continue
            data[reino].append({
                "especie": row["especie"],
                "pb": pb,
                "secuencias": n_seq,
                "n_sf": n_sf,
            })
    return data


def stratify_by_length(specs, n_strata=4):
    """Divide especies en cuartiles por longitud."""
    sorted_specs = sorted(specs, key=lambda s: s["pb"])
    n = len(sorted_specs)
    strata = [[] for _ in range(n_strata)]
    for i, sp in enumerate(sorted_specs):
        stratum_idx = min(i * n_strata // n, n_strata - 1)
        strata[stratum_idx].append(sp)
    return strata


def sample_from_strata(strata, n_total):
    """Muestreo proporcional por estrato."""
    total_pool = sum(len(s) for s in strata)
    sampled = []
    for st in strata:
        n_take = round(n_total * len(st) / total_pool)
        n_take = min(n_take, len(st))
        sampled.extend(random.sample(st, n_take))
    return sampled


def split_train_val_test(specs):
    """Separa en train/val/test preservando distribucion de longitud (por estratos)."""
    strata = stratify_by_length(specs, n_strata=4)
    train, val, test = [], [], []
    for st in strata:
        st = list(st)
        random.shuffle(st)
        n = len(st)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        train.extend(st[:n_train])
        val.extend(st[n_train:n_train + n_val])
        test.extend(st[n_train + n_val:])
    return train, val, test


def summarize_split(name, specs):
    pbs = sorted([s["pb"] for s in specs])
    n = len(pbs)
    if n == 0:
        return f"{name}: 0 especies"
    return (f"{name}: {n} esp  "
            f"min={pbs[0]:,}  "
            f"Q1={pbs[n//4]:,}  "
            f"med={pbs[n//2]:,}  "
            f"Q3={pbs[3*n//4]:,}  "
            f"max={pbs[-1]:,}  pb")


def main():
    print("=== CARGA DE DATOS ===")
    data = load_species()
    for k, specs in sorted(data.items()):
        pbs = sorted(s["pb"] for s in specs)
        print(f"  {k:10s}: {len(specs):4d} especies | "
              f"pb Q1={pbs[len(pbs)//4]:>10,} "
              f"med={pbs[len(pbs)//2]:>10,} "
              f"Q3={pbs[3*len(pbs)//4]:>10,}")

    print("\n=== MUESTREO ESTRATIFICADO ===")
    final_selection = {"train": [], "val": [], "test": []}
    report = []

    for reino in ("Plantas", "Animales", "Hongos"):
        n_target = N_PER_KINGDOM[reino]
        print(f"\n--- {reino} (objetivo: {n_target}) ---")

        all_specs = data[reino]
        strata = stratify_by_length(all_specs, n_strata=4)
        for i, st in enumerate(strata):
            pbs = [s["pb"] for s in st]
            print(f"  Estrato Q{i+1}: {len(st)} especies disponibles, "
                  f"pb range {min(pbs):,} - {max(pbs):,}")

        # Muestreo proporcional por estrato
        sampled = sample_from_strata(strata, n_target)
        print(f"  Muestreadas: {len(sampled)} especies")

        # Split train/val/test
        tr, va, te = split_train_val_test(sampled)
        for s in tr: s["reino"] = reino
        for s in va: s["reino"] = reino
        for s in te: s["reino"] = reino
        final_selection["train"].extend(tr)
        final_selection["val"].extend(va)
        final_selection["test"].extend(te)
        print(f"    Train: {len(tr)}  Val: {len(va)}  Test: {len(te)}")

        report.append({
            "reino": reino,
            "total_disponible": len(all_specs),
            "muestreadas": len(sampled),
            "train": len(tr),
            "val": len(va),
            "test": len(te),
            "pb_median_train": sorted(s["pb"] for s in tr)[len(tr)//2] if tr else 0,
            "pb_median_val": sorted(s["pb"] for s in va)[len(va)//2] if va else 0,
            "pb_median_test": sorted(s["pb"] for s in te)[len(te)//2] if te else 0,
        })

    print("\n=== RESUMEN FINAL ===")
    for split in ("train", "val", "test"):
        specs = final_selection[split]
        print(f"\n{split.upper()}: {len(specs)} especies totales")
        for reino in ("Plantas", "Animales", "Hongos"):
            sub = [s for s in specs if s["reino"] == reino]
            print("  " + summarize_split(reino, sub))

    # Guardar CSV
    out_csv = OUT / "species_split_proposal.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "reino", "especie", "pb", "secuencias", "n_superfamilias"])
        for split in ("train", "val", "test"):
            for s in final_selection[split]:
                w.writerow([split, s["reino"], s["especie"],
                           s["pb"], s["secuencias"], s["n_sf"]])
    print(f"\n[csv] {out_csv}")

    # Reporte markdown
    out_md = OUT / "species_split_proposal.md"
    lines = [
        "# Propuesta de muestreo estratificado por longitud",
        "",
        "Siguiendo la instrucción de Simón (chat 14/10): muestreo aleatorio "
        "estratificado por longitud (pb) para seleccionar especies de los 3 "
        "reinos, conservando la distribución de longitud entre splits.",
        "",
        "## Metodología",
        "",
        "1. **Excluir reinos no relevantes**: Ambiguo, No asignado, Otros "
        "eucariotas, Virus (según indicación previa de Simón).",
        "2. **Filtrar especies con <1 kb** (demasiado pequeñas para generar ventanas).",
        f"3. **Objetivo por reino**: {N_PER_KINGDOM['Plantas']} especies cada "
        "uno (ajustable).",
        "4. **Estratificar por cuartiles de longitud** dentro de cada reino "
        "(Q1=pequeñas, Q4=grandes).",
        "5. **Muestreo proporcional** desde cada estrato.",
        f"6. **Split 70/15/15** (train/val/test) por estrato, para conservar "
        "distribución de longitud.",
        "7. **Semilla fija 42** para reproducibilidad.",
        "",
        "## Resumen por reino",
        "",
        "| Reino | Total disponible | Muestreadas | Train | Val | Test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report:
        lines.append(f"| {r['reino']} | {r['total_disponible']:,} | "
                     f"{r['muestreadas']} | {r['train']} | {r['val']} | {r['test']} |")

    lines += ["",
              "## Mediana de longitud (pb) por split (debe ser similar)",
              "",
              "| Reino | Train | Val | Test |",
              "|---|---:|---:|---:|"]
    for r in report:
        lines.append(f"| {r['reino']} | "
                     f"{r['pb_median_train']:,} | "
                     f"{r['pb_median_val']:,} | "
                     f"{r['pb_median_test']:,} |")

    lines += ["",
              "## Observaciones",
              "",
              "- Si las medianas por split son similares (±20%), la "
              "estratificación por longitud fue correcta.",
              "- El CSV completo con las especies seleccionadas está en "
              "`species_split_proposal.csv`.",
              "- El número por reino (150) es ajustable: puede subir a "
              "200-300 por reino si computacionalmente aguantamos.",
              ""]

    out_md.write_text("\n".join(lines))
    print(f"[md]  {out_md}")


if __name__ == "__main__":
    main()
