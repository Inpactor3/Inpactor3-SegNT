"""Downstream: BED + tamaños de cromosoma -> ventanas 50kb JSONL.
Uso: python3 build_windows.py <bed> <sizes.tsv> <out.jsonl> <reino> <split>
Misma lógica que pipeline_downstream_*: label=1 si algún TE solapa >= max(50bp, 20% del TE).
"""
import json
import sys
from collections import defaultdict

bed_file, sizes_file, out_file, reino, split = sys.argv[1:6]
WINDOW = 50000
MIN_OVERLAP_BP = 50
MIN_OVERLAP_FRAC = 0.2

sizes = {}
with open(sizes_file) as f:
    for line in f:
        c, L = line.rstrip("\n").split("\t")
        sizes[c] = int(L)

hits = defaultdict(list)
with open(bed_file) as f:
    for line in f:
        s = line.rstrip("\n").split("\t")
        if len(s) < 4:
            continue
        hits[s[0]].append((int(s[1]), int(s[2]), s[3]))

species = None
import os
species = os.path.basename(bed_file)[:-4]  # tag sin .bed

n_pos = n_neg = n_total = 0
with open(out_file, "w") as fo:
    for chrom, L in sizes.items():
        for start in range(0, max(0, L - WINDOW + 1), WINDOW):
            end = start + WINDOW
            te_in = [h for h in hits.get(chrom, []) if h[0] < end and h[1] > start]
            is_pos = False
            bp_total = 0
            orders = set()
            for s, e, name in te_in:
                overlap = min(e, end) - max(s, start)
                if overlap <= 0:
                    continue
                bp_total += overlap
                tokens = name.split("|")
                if len(tokens) >= 2:
                    ct = tokens[1].split("/")
                    if len(ct) >= 2:
                        orders.add(ct[1])
                elem_len = e - s
                thr = max(MIN_OVERLAP_BP, int(MIN_OVERLAP_FRAC * elem_len))
                if overlap >= thr:
                    is_pos = True
            rec = {
                "window_id": f"{species}_{chrom}_{start}",
                "species": species,
                "reino": reino,
                "split": split,
                "chrom": chrom,
                "start": start,
                "end": end,
                "label": 1 if is_pos else 0,
                "n_TE": len(te_in),
                "bp_TE": bp_total,
                "coverage_fraction": round(bp_total / WINDOW, 4),
                "orders_present": sorted(orders),
            }
            fo.write(json.dumps(rec) + "\n")
            n_total += 1
            if is_pos:
                n_pos += 1
            else:
                n_neg += 1

print(f"{species}: pos={n_pos} neg={n_neg} total={n_total}")
