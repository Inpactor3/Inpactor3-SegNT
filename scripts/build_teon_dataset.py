"""
Construye el dataset común a partir de PanTEon (Plantae + Animalia + Fungi).

Salida: manifest.jsonl + panteon_ltr_subset.fasta (formato lazy).
Los genomas de fondo se descargan con scripts/download_backgrounds.sh.

Uso:
    python scripts/build_teon_dataset.py \\
        --panteon data/raw/PanTEon_Database_v1.6.2.fasta \\
        --metadata data/raw/PanTEon_Database_metadata_v1.6.2.csv \\
        --genomes-dir data/common_dataset/genomes \\
        --out-dir data/common_dataset \\
        --n-positives 50000 \\
        --n-negatives 50000 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Superfamilias LTR con id de clase
LTR_SUPERFAMILIES = {
    "COPIA": 1,
    "GYPSY": 2,
    "ERV": 3,
    "BELPAO": 4,
    "LARD": 5,
    "TRIM": 6,
}

# Reino → nombre de genoma de fondo
KINGDOM_GENOME = {
    "Plantae": "Zea_mays",
    "Animalia": "Homo_sapiens",
    "Fungi": "Neurospora_crassa",
}


def parse_panteon_header(h: str) -> tuple[str, str, str]:
    """
    Header PanTEon: SEQID#CLASSI/LTR/COPIA @Species name
    Devuelve (superfamily, species_normalized, ltr_class_bool_str).
    """
    if "#" not in h:
        return "unknown", "unknown", "unknown"
    seqid, rest = h.split("#", 1)
    if "@" in rest:
        classif, sp = rest.split("@", 1)
        species = sp.strip().replace(" ", "_")
    else:
        classif, species = rest.strip(), "unknown"
    parts = classif.strip().split("/")
    if len(parts) < 3 or parts[1] != "LTR":
        return "NONLTR", species, seqid.strip()
    return parts[2], species, seqid.strip()


def load_species_kingdom(metadata_csv: Path) -> dict[str, str]:
    """Devuelve {species_normalized: kingdom} desde metadata."""
    m: dict[str, str] = {}
    with open(metadata_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["Species"].strip().replace(" ", "_")
            m[sp] = row["Kingdom"].strip()
    return m


def iter_fasta(path: Path):
    head, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if head is not None:
                    yield head, "".join(buf).upper()
                head = line[1:]
                buf = []
            else:
                buf.append(line)
        if head is not None:
            yield head, "".join(buf).upper()


def scaffold_lengths(fasta: Path) -> dict[str, int]:
    """Devuelve {scaffold_id: longitud}."""
    lens: dict[str, int] = {}
    cur, n = None, 0
    with open(fasta) as f:
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


def species_split(all_species: list[str], seed: int = 42):
    """80/10/10 por especie."""
    rng = random.Random(seed)
    sp = sorted(all_species)
    rng.shuffle(sp)
    n = len(sp)
    n_train = int(0.8 * n)
    n_val = int(0.9 * n) - n_train
    return {
        **{s: "train" for s in sp[:n_train]},
        **{s: "val" for s in sp[n_train : n_train + n_val]},
        **{s: "test" for s in sp[n_train + n_val :]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panteon", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--genomes-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--window-size", type=int, default=50_000)
    ap.add_argument("--n-positives", type=int, default=50_000,
                    help="Cantidad total de ventanas positivas a generar")
    ap.add_argument("--n-negatives", type=int, default=50_000)
    ap.add_argument("--min-ltr-len", type=int, default=1000)
    ap.add_argument("--max-ltr-len", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- 1. cargar kingdom por especie ----
    print(f"[load] {args.metadata}")
    sp2king = load_species_kingdom(args.metadata)
    king_count = Counter(sp2king.values())
    print(f"[species] {len(sp2king)} en metadata · {dict(king_count)}")

    # ---- 2. escanear PanTEon: filtrar LTR-RTs y por reino conocido ----
    print(f"[scan] {args.panteon}")
    ltrs_by_kingdom: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    # cada entrada: (panteon_id, superfamily, species, sequence)
    total = kept = 0
    for h, seq in iter_fasta(args.panteon):
        total += 1
        sf, sp, seqid = parse_panteon_header(h)
        if sf not in LTR_SUPERFAMILIES:
            continue
        L = len(seq)
        if L < args.min_ltr_len or L > args.max_ltr_len:
            continue
        kingdom = sp2king.get(sp)
        if kingdom not in KINGDOM_GENOME:
            continue
        ltrs_by_kingdom[kingdom].append((seqid, sf, sp, seq))
        kept += 1
    print(f"[filter] escaneadas {total:,} · LTR aptas {kept:,}")
    for k, v in ltrs_by_kingdom.items():
        sf_counts = Counter(t[1] for t in v)
        print(f"  {k}: {len(v):,} · {dict(sf_counts)}")

    # ---- 3. split por especie ----
    all_species = sorted({t[2] for lst in ltrs_by_kingdom.values() for t in lst})
    sp_split = species_split(all_species, seed=args.seed)
    print(f"[split] por especie: {Counter(sp_split.values())}")

    # ---- 4. cargar longitudes de scaffolds por genoma ----
    genome_scaffolds: dict[str, dict[str, int]] = {}
    for kingdom, gname in KINGDOM_GENOME.items():
        gpath = args.genomes_dir / f"{gname}.fasta"
        if not gpath.exists():
            print(f"[warn] genoma no encontrado: {gpath} (correr download_backgrounds.sh)")
            genome_scaffolds[gname] = {}
            continue
        print(f"[scan] {gpath}")
        genome_scaffolds[gname] = scaffold_lengths(gpath)
        print(f"  {gname}: {len(genome_scaffolds[gname])} scaffolds · "
              f"{sum(genome_scaffolds[gname].values()):,} bp")

    # ---- 5. escribir manifest + FASTA subset ----
    manifest_path = args.out_dir / "manifest.jsonl"
    subset_path = args.out_dir / "panteon_ltr_subset.fasta"
    used_panteon_ids: set[str] = set()
    n_written = 0

    with open(manifest_path, "w") as mfout, open(subset_path, "w") as subout:
        # ---- positivos: balanceados por reino ----
        for kingdom, ltr_list in ltrs_by_kingdom.items():
            gname = KINGDOM_GENOME[kingdom]
            scaffolds = genome_scaffolds.get(gname, {})
            if not scaffolds:
                print(f"[skip] {kingdom} sin genoma de fondo disponible")
                continue
            n_target = args.n_positives // len(ltrs_by_kingdom)
            print(f"[pos ] {kingdom}: generando {n_target:,} ventanas")

            rng_king = random.Random(args.seed * 7 + hash(kingdom) % 100)
            usable_ltrs = [t for t in ltr_list]
            for i in range(n_target):
                seqid, sf, sp, seq = usable_ltrs[i % len(usable_ltrs)]
                # elige scaffold aleatorio con longitud suficiente
                usable_scaf = [s for s, L in scaffolds.items() if L >= args.window_size]
                if not usable_scaf:
                    break
                scaf = rng_king.choice(usable_scaf)
                sc_len = scaffolds[scaf]
                w_start = rng_king.randint(0, sc_len - args.window_size)
                w_end = w_start + args.window_size
                # posición de inserción dentro de la ventana
                ins_pos = rng_king.randint(0, args.window_size - len(seq))
                # guardar TE en subset si no está
                if seqid not in used_panteon_ids:
                    subout.write(f">{seqid}\n")
                    for k in range(0, len(seq), 80):
                        subout.write(seq[k : k + 80] + "\n")
                    used_panteon_ids.add(seqid)

                entry = {
                    "window_id": f"pos_{n_written:07d}",
                    "kingdom": kingdom,
                    "background_genome": gname,
                    "background_scaffold": scaf,
                    "background_start": w_start,
                    "background_end": w_end,
                    "has_ltr_rt": True,
                    "insertions": [{
                        "panteon_id": seqid,
                        "lineage": sf,
                        "lineage_id": LTR_SUPERFAMILIES[sf],
                        "insert_position": ins_pos,
                        "insert_length": len(seq),
                        "species_origin": sp,
                    }],
                    "split": sp_split.get(sp, "train"),
                }
                mfout.write(json.dumps(entry) + "\n")
                n_written += 1

        # ---- negativos: ventanas del genoma sin plantar nada ----
        for kingdom, gname in KINGDOM_GENOME.items():
            scaffolds = genome_scaffolds.get(gname, {})
            if not scaffolds:
                continue
            n_target = args.n_negatives // len(KINGDOM_GENOME)
            print(f"[neg ] {kingdom}: generando {n_target:,} ventanas")
            rng_neg = random.Random(args.seed * 13 + hash(kingdom) % 100)
            usable_scaf = [s for s, L in scaffolds.items() if L >= args.window_size]
            for i in range(n_target):
                scaf = rng_neg.choice(usable_scaf)
                sc_len = scaffolds[scaf]
                w_start = rng_neg.randint(0, sc_len - args.window_size)
                entry = {
                    "window_id": f"neg_{n_written:07d}",
                    "kingdom": kingdom,
                    "background_genome": gname,
                    "background_scaffold": scaf,
                    "background_start": w_start,
                    "background_end": w_start + args.window_size,
                    "has_ltr_rt": False,
                    "insertions": [],
                    "split": ["train", "val", "test"][i % 3 if i > 0 and i % 10 == 0 else 0],
                }
                mfout.write(json.dumps(entry) + "\n")
                n_written += 1

    print(f"\n[ok  ] {n_written:,} ventanas en {manifest_path}")
    print(f"[ok  ] {len(used_panteon_ids):,} LTR-RTs únicos en {subset_path}")
    print(f"[note] entrenar con: python -m inpactor3_segnt.train_teon "
          f"--manifest {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
