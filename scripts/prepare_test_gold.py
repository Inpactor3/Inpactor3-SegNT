"""Prepara la separación TEST ORO / TEST PLATA / VALIDACION / TRAIN.

Regla Simón (final del mensaje):
    1. Test "oro": 1-2 especies por reino con anotación de TEs curada por la
       comunidad. Teselado completo, prevalencia natural.
    2. Excluir del train esas especies y TODO SU GENERO.
    3. Test "plata": especies retenidas con etiqueta RepeatMasker, se reportan
       aparte del oro.
    4. Validación: otras especies retenidas, prevalencia natural, de ahí sale
       el umbral. Nunca del test.
    5. Regiones ambiguas (Unknown, satélites, baja complejidad) se marcan como
       ignorar en el test, no como negativo.

Uso:
    python scripts/prepare_test_gold.py \\
        --oro "arabidopsis,drosophila,scerevisiae" \\
        --plata "aspergillus,spombe" \\
        --val "neurospora,celegans"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "raw" / "PanTEon_Database_metadata_v1.6.2.csv"
ANN_DIR = ROOT / "annotations"
OUT_DIR = ROOT / "data" / "splits"

# Mapeo short-name -> especie completa (para buscar en PanTEon metadata)
SPECIES_FULL = {
    "arabidopsis":   "Arabidopsis thaliana",
    "brachypodium":  "Brachypodium distachyon",
    "oryza":         "Oryza sativa",
    "zea":           "Zea mays",
    "ananas":        "Ananas comosus",
    "sorghum":       "Sorghum bicolor",
    "solanum":       "Solanum lycopersicum",
    "scerevisiae":   "Saccharomyces cerevisiae",
    "neurospora":    "Neurospora crassa",
    "spombe":        "Schizosaccharomyces pombe",
    "aspergillus":   "Aspergillus nidulans",
    "celegans":      "Caenorhabditis elegans",
    "drosophila":    "Drosophila melanogaster",
    "danio":         "Danio rerio",
    "mus":           "Mus musculus",
}


def load_metadata():
    """Lee metadata y arma mapa especie -> genero/reino."""
    sp_info = {}
    genus_species = {}  # genus -> list[Species]
    with META.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sp = row["Species"]
            if not sp:
                continue
            genus = sp.split()[0]
            sp_info[sp] = {
                "genus": genus,
                "kingdom": row["Kingdom"],
                "family": row.get("Family", ""),
            }
            genus_species.setdefault(genus, []).append(sp)
    return sp_info, genus_species


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oro", required=True,
                    help="lista separada por comas (short names)")
    ap.add_argument("--plata", default="",
                    help="lista separada por comas (short names)")
    ap.add_argument("--val", default="",
                    help="lista separada por comas (short names)")
    ap.add_argument("--exclude-family", action="store_true",
                    help="además del género, excluir toda la familia")
    args = ap.parse_args()

    oro = [s.strip() for s in args.oro.split(",") if s.strip()]
    plata = [s.strip() for s in args.plata.split(",") if s.strip()]
    val = [s.strip() for s in args.val.split(",") if s.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp_info, genus_species = load_metadata()

    def expand_to_species(short_names):
        """Convertir nombres cortos a especies PanTEon."""
        out = []
        for sh in short_names:
            full = SPECIES_FULL.get(sh)
            if full:
                out.append(full)
        return out

    oro_full = expand_to_species(oro)
    plata_full = expand_to_species(plata)
    val_full = expand_to_species(val)

    # Expandir ORO a toda su familia/género para exclusión del train
    excluded_from_train = set(oro_full) | set(plata_full) | set(val_full)
    excluded_genera = set()
    for sp in oro_full:
        if sp in sp_info:
            g = sp_info[sp]["genus"]
            excluded_genera.add(g)
            for related in genus_species.get(g, []):
                excluded_from_train.add(related)

    print(f"\n=== Configuración del split ===")
    print(f"  TEST ORO:   {oro}  ({len(oro_full)} especies PanTEon)")
    print(f"  TEST PLATA: {plata}")
    print(f"  VAL:        {val}")
    print(f"  Géneros excluidos del train por homología (ORO):"
          f" {sorted(excluded_genera)}")
    print(f"  Total especies excluidas del train: {len(excluded_from_train)}")

    # Train = resto
    all_species_in_panteon = set(sp_info.keys())
    train_species = all_species_in_panteon - excluded_from_train

    # Breakdown por reino
    train_by_kingdom = {"Animalia": 0, "Fungi": 0, "Plantae": 0, "Other": 0}
    for sp in train_species:
        k = sp_info[sp]["kingdom"]
        train_by_kingdom[k if k in train_by_kingdom else "Other"] += 1

    print(f"\n  TRAIN: {len(train_species)} especies ({sum(train_by_kingdom.values())} totales)")
    for k, n in train_by_kingdom.items():
        print(f"    {k}: {n}")

    # Salida JSON
    manifest = {
        "test_oro": {
            "species_short": oro,
            "species_full": oro_full,
            "excluded_genera": sorted(excluded_genera),
            "related_species_excluded": sorted(
                [s for s in excluded_from_train
                 if s not in set(plata_full) | set(val_full)]
            ),
        },
        "test_plata": {
            "species_short": plata,
            "species_full": plata_full,
        },
        "validation": {
            "species_short": val,
            "species_full": val_full,
        },
        "train": {
            "n_species": len(train_species),
            "by_kingdom": train_by_kingdom,
            "excluded_total": len(excluded_from_train),
        },
    }
    out_file = OUT_DIR / "split_manifest.json"
    out_file.write_text(json.dumps(manifest, indent=2))

    # Lista de BEDs listos
    print(f"\n=== BEDs disponibles para TEST ===")
    for sh in oro + plata + val:
        bed = ANN_DIR / f"{sh}.bed"
        ignore = ANN_DIR / f"{sh}.ignore.bed"
        status_bed = "✓" if bed.exists() else "✗"
        status_ign = "✓" if ignore.exists() else "✗"
        print(f"  {sh:15s}  bed={status_bed}  ignore_mask={status_ign}")

    print(f"\n[saved] {out_file}")


if __name__ == "__main__":
    main()
