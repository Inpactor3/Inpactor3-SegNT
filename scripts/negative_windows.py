import argparse, gzip, json, random, sys
from bisect import bisect_left

def read_sizes(p):
    d = {}
    with open(p) as f:
        for line in f:
            c, L = line.rstrip("\n").split("\t")
            d[c] = int(L)
    return d

def read_bed(p):
    hits = {}
    with open(p) as f:
        for line in f:
            s = line.rstrip("\n").split("\t")
            if len(s) < 4:
                continue
            try:
                st, en = int(s[1]), int(s[2])
            except ValueError:
                continue
            name = s[3]
            cls = name.split("|")[1] if "|" in name else ""
            order = ""
            parts = cls.split("/")
            if len(parts) >= 2:
                order = parts[1]
            hits.setdefault(s[0], []).append((st, en, cls, order))
    return hits

def read_gff(p):
    genes = {}
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 5 or c[2] != "gene":
                continue
            try:
                st, en = int(c[3]) - 1, int(c[4])
            except ValueError:
                continue
            genes.setdefault(c[0], []).append((st, en))
    return genes

def thr(st, en):
    return max(50, int(0.2 * (en - st)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--gff")
    ap.add_argument("--out", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--reino", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_jsonl", default=None)
    a = ap.parse_args()

    excluded = set()
    if a.exclude_jsonl:
        with open(a.exclude_jsonl) as f:
            for line in f:
                try:
                    excluded.add(json.loads(line)["window_id"])
                except Exception:
                    pass

    W = 50000
    BUF = 1000
    sizes = read_sizes(a.sizes)
    hits = read_bed(a.bed)
    genes = read_gff(a.gff) if a.gff else {}

    random.seed(a.seed)
    out = open(a.out, "w")
    counts = {"intergenic": 0, "genic": 0, "other_te": 0}

    for chrom, L in sizes.items():
        nw = L // W
        if nw == 0:
            continue
        any_hit = [False] * nw
        ltr_hit = [False] * nw
        nonltr_hit = [False] * nw
        near = [False] * nw
        gene_cov = [0] * nw
        bp_te = [0] * nw
        n_te = [0] * nw
        orders = [set() for _ in range(nw)]

        for st, en, cls, order in hits.get(chrom, []):
            if en <= st:
                continue
            t = thr(st, en)
            i0 = max(0, st // W)
            i1 = min(nw - 1, (en - 1) // W)
            is_ltr = "LTR" in cls
            for i in range(i0, i1 + 1):
                ws, we = i * W, (i + 1) * W
                ov = min(en, we) - max(st, ws)
                if ov >= t:
                    any_hit[i] = True
                    if is_ltr:
                        ltr_hit[i] = True
                    else:
                        nonltr_hit[i] = True
                if ov > 0:
                    bp_te[i] += ov
                    n_te[i] += 1
                    if order:
                        orders[i].add(order)
            if st - BUF < 0:
                near[0] = True
            else:
                j0 = min(nw - 1, max(0, (st - BUF) // W))
                near[j0] = True
            j1 = min(nw - 1, (en + BUF - 1) // W)
            near[j1] = True

        for st, en in genes.get(chrom, []):
            i0 = max(0, st // W)
            i1 = min(nw - 1, (en - 1) // W)
            for i in range(i0, i1 + 1):
                ws, we = i * W, (i + 1) * W
                ov = min(en, we) - max(st, ws)
                if ov > 0:
                    gene_cov[i] += ov

        cand = {"intergenic": [], "genic": [], "other_te": []}
        for i in range(nw):
            st = i * W
            if not any_hit[i] and gene_cov[i] == 0 and not near[i]:
                cand["intergenic"].append(i)
            if gene_cov[i] >= 0.1 * W and not ltr_hit[i]:
                cand["genic"].append(i)
            if nonltr_hit[i] and not ltr_hit[i]:
                cand["other_te"].append(i)

        for cat, idxs in cand.items():
            if not idxs:
                continue
            k = min(a.n, len(idxs))
            pick = random.sample(idxs, k)
            for i in pick:
                st = i * W
                wid = f"{a.species}_{chrom}_{st}"
                if wid in excluded:
                    continue
                rec = {
                    "window_id": wid,
                    "species": a.species,
                    "reino": a.reino,
                    "split": a.split,
                    "chrom": chrom,
                    "start": st,
                    "end": st + W,
                    "label": 0,
                    "n_TE": n_te[i] if cat == "other_te" else 0,
                    "bp_TE": bp_te[i] if cat == "other_te" else 0,
                    "coverage_fraction": round(bp_te[i] / W, 4) if cat == "other_te" else 0.0,
                    "orders_present": sorted(orders[i]) if cat == "other_te" else [],
                    "negative_type": cat,
                }
                if cat == "genic":
                    rec["gene_coverage_fraction"] = round(gene_cov[i] / W, 4)
                out.write(json.dumps(rec) + "\n")
                counts[cat] += 1
    out.close()
    print(f"{a.species}: " + " ".join(f"{k}={v}" for k, v in counts.items()))

if __name__ == "__main__":
    main()
