"""QC de JSONL de ventanas: stats por especie + resumen global.
Uso: python3 qc_windows.py <dir_windows> <out_csv> [reino]
Recorre <dir_windows>/<split>/*.jsonl
"""
import json, sys, os, glob, csv

root, out_csv = sys.argv[1], sys.argv[2]
reino = sys.argv[3] if len(sys.argv) > 3 else ""

rows = []
tot_win = tot_pos = 0
seen_ids = set()
dup_ids = 0

for split in ("train", "val", "test"):
    for f in sorted(glob.glob(os.path.join(root, split, "*.jsonl"))):
        species = os.path.basename(f)[:-6]
        n = pos = neg = 0
        cov_sum = 0.0; cov_max = 0.0
        n_te_sum = 0
        ids_here = set(); dups_here = 0
        try:
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    n += 1
                    if r.get("label") == 1: pos += 1
                    else: neg += 1
                    c = float(r.get("coverage_fraction", 0))
                    cov_sum += c
                    cov_max = max(cov_max, c)
                    n_te_sum += int(r.get("n_TE", 0))
                    wid = r.get("window_id")
                    if wid in ids_here: dups_here += 1
                    ids_here.add(wid)
        except Exception as e:
            print(f"ERROR {f}: {e}", file=sys.stderr)
            continue
        dup_ids += dups_here
        rows.append({
            "reino": reino, "split": split, "species": species.encode("ascii", "ignore").decode(),
            "windows": n, "pos": pos, "neg": neg,
            "pos_frac": round(pos / n, 4) if n else 0,
            "cov_mean": round(cov_sum / n, 4) if n else 0,
            "cov_max": round(cov_max, 4),
            "n_te_mean": round(n_te_sum / n, 2) if n else 0,
            "dup_windows": dups_here, "status": "OK" if n > 0 else "EMPTY",
        })
        tot_win += n; tot_pos += pos

os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                       ["reino","split","species","windows","pos","neg","pos_frac","cov_mean","cov_max","n_te_mean","dup_windows","status"])
    w.writeheader(); w.writerows(rows)

print(f"Archivos: {len(rows)} | ventanas: {tot_win} | positivas: {tot_pos} ({round(100*tot_pos/max(tot_win,1),1)}%) | duplicados: {dup_ids}")
print(f"CSV: {out_csv}")
