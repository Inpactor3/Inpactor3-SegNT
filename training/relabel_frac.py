"""Recalcula las etiquetas de las ventanas con otro umbral de solapamiento.

Regla original: una ventana es positiva si algun LTR la solapa al menos
max(50 pb, 0.2 * longitud_del_elemento). Aqui se puede cambiar ese 0.2.

No lee genomas: usa el window_id de cada npz y el BED de la especie. Los
embeddings no cambian, solo las etiquetas.

Uso:
  python relabel_frac.py --bundle_dir ../../seq_bundles4/Animales \
      --bed_root /media/.../InpactorRM --reino Animales --frac 0.5 \
      --out_dir /tmp/lab50
"""
import argparse
import glob
import json
import os
import re

import numpy as np

norm = lambda x: re.sub(r"_+", "_", x)


def read_bed(path):
    hits = {}
    with open(path, errors="ignore") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            try:
                st, en = int(c[1]), int(c[2])
            except ValueError:
                continue
            cls = c[3].split("|")[1] if "|" in c[3] else ""
            hits.setdefault(c[0], []).append((st, en, "LTR" in cls))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--bed_root", required=True)
    ap.add_argument("--reino", required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--min_bp", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    beds = {}
    for sub in ("annotations_dedup", "annotations"):
        for p in glob.glob(os.path.join(args.bed_root, sub, args.reino, "*", "*.bed")):
            if os.path.getsize(p) > 0:
                beds.setdefault(norm(os.path.basename(p)[:-4]), p)

    tot = flip = 0
    for split in args.splits.split(","):
        out = os.path.join(args.out_dir, split)
        os.makedirs(out, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(args.bundle_dir, split, "*.npz"))):
            tag = os.path.basename(f)[:-4]
            bedp = beds.get(norm(tag))
            if bedp is None:
                print("  SIN BED:", tag)
                continue
            hits = read_bed(bedp)
            z = np.load(f, allow_pickle=True)
            ids = z["window_id"]
            old = z["label"]
            new = np.zeros(len(ids), dtype=np.int8)
            f_flip = 0
            for i, wid in enumerate(ids):
                wid = str(wid)
                # el id es "<tag>_<cromosoma>_<inicio>"; el cromosoma puede tener
                # guiones bajos (NC_012346.1), asi que quitamos el tag conocido
                if not wid.startswith(tag):
                    new[i] = old[i]
                    continue
                resto = wid[len(tag) + 1:]
                if "_" not in resto:
                    new[i] = old[i]
                    continue
                chrom, st_s = resto.rsplit("_", 1)
                if not st_s.isdigit():
                    new[i] = old[i]
                    continue
                st = int(st_s)
                en = st + 50000
                pos = 0
                for hs, he, is_ltr in hits.get(chrom, []):
                    if not is_ltr or hs >= en or he <= st:
                        continue
                    ov = min(he, en) - max(hs, st)
                    if ov >= max(args.min_bp, int(args.frac * (he - hs))):
                        pos = 1
                        break
                new[i] = pos
                if new[i] != old[i]:
                    f_flip += 1
            np.save(os.path.join(out, f"{tag}_labels.npy"), new)
            tot += len(new)
            flip += f_flip
            print(f"  {split}/{tag[:40]:40s} n={len(new):6d} pos={int(new.sum()):5d} "
                  f"cambian={f_flip}")
    print(f"\nTOTAL ventanas={tot} · etiquetas que cambian={flip} "
          f"({100*flip/max(tot,1):.1f}%)")


if __name__ == "__main__":
    main()
