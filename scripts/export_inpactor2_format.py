"""
Exporta el manifiesto común al formato que Inpactor 2 espera:

  - Inpactor2_train_positive.fasta   (ventanas con LTR-RT)
  - Inpactor2_train_negative.fasta   (ventanas sin LTR-RT)
  - Inpactor2_class_train.fasta      (secuencias LTR-RT individuales para
                                       la red Class, con linaje en el header)

Uso:
    python scripts/export_inpactor2_format.py \\
        --manifest data/common_dataset/manifest.jsonl \\
        --genomes-dir data/common_dataset/genomes \\
        --panteon-subset data/common_dataset/panteon_ltr_subset.fasta \\
        --out-dir data/common_dataset/inpactor2_format \\
        --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inpactor3_segnt.teon_dataset import (
    FastaSlicer, PanteonSubset, load_manifest,
)

SUPERFAM_TO_INPACTOR2 = {
    "COPIA": "RLC/UNKNOWN",
    "GYPSY": "RLG/UNKNOWN",
    "ERV": "RLG/ERV",
    "BELPAO": "RLC/BELPAO",
    "LARD": "RLC/LARD",
    "TRIM": "RLC/TRIM",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--genomes-dir", required=True, type=Path)
    ap.add_argument("--panteon-subset", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recipes = load_manifest(args.manifest, split=args.split)
    print(f"[read] {len(recipes)} recetas del split {args.split}")

    # Índices lazy
    slicers = {}
    for g in {r.background_genome for r in recipes}:
        p = args.genomes_dir / f"{g}.fasta"
        if p.exists():
            print(f"[idx ] {p}")
            slicers[g] = FastaSlicer(p)
    panteon = PanteonSubset(args.panteon_subset)

    pos_fa = args.out_dir / f"Inpactor2_{args.split}_positive.fasta"
    neg_fa = args.out_dir / f"Inpactor2_{args.split}_negative.fasta"
    class_fa = args.out_dir / f"Inpactor2_class_{args.split}.fasta"

    n_pos = n_neg = n_cls = 0
    with open(pos_fa, "w") as pf, open(neg_fa, "w") as nf, open(class_fa, "w") as cf:
        for r in recipes:
            if r.background_genome not in slicers:
                continue
            seq = slicers[r.background_genome].slice(
                r.background_scaffold, r.background_start, r.background_end
            )
            if len(seq) < (r.background_end - r.background_start):
                seq = seq + "N" * ((r.background_end - r.background_start) - len(seq))
            seq = list(seq)
            for ins in r.insertions:
                te_seq = panteon.get(ins["panteon_id"])
                for i, c in enumerate(te_seq):
                    if ins["insert_position"] + i < len(seq):
                        seq[ins["insert_position"] + i] = c
            seq_str = "".join(seq)

            if r.has_ltr_rt:
                pf.write(f">{r.window_id}\n{seq_str}\n")
                n_pos += 1
                # Extraer el TE individual para la red Class
                for ins in r.insertions:
                    te_seq = panteon.get(ins["panteon_id"])
                    lineage_i2 = SUPERFAM_TO_INPACTOR2.get(ins["lineage"], "UNKNOWN")
                    cf.write(
                        f">{ins['panteon_id']}#{lineage_i2}#{ins['species_origin']}\n"
                        f"{te_seq}\n"
                    )
                    n_cls += 1
            else:
                nf.write(f">{r.window_id}\n{seq_str}\n")
                n_neg += 1

    print(f"[ok  ] positivos: {n_pos:,} → {pos_fa}")
    print(f"[ok  ] negativos: {n_neg:,} → {neg_fa}")
    print(f"[ok  ] LTR-RTs para Class: {n_cls:,} → {class_fa}")


if __name__ == "__main__":
    raise SystemExit(main())
