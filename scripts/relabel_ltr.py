import argparse, json

def read_bed(p):
    hits = {}
    with open(p) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            try:
                st, en = int(c[1]), int(c[2])
            except ValueError:
                continue
            name = c[3]
            cls = name.split("|")[1] if "|" in name else ""
            order = ""
            parts = cls.split("/")
            if len(parts) >= 2:
                order = parts[1]
            hits.setdefault(c[0], []).append((st, en, "LTR" in cls, order))
    return hits

def thr(st, en):
    return max(50, int(0.2 * (en - st)))

ap = argparse.ArgumentParser()
ap.add_argument("--bed", required=True)
ap.add_argument("--jsonl", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

hits = read_bed(a.bed)
W = 50000
n_in = n_pos = n_flip = 0
with open(a.jsonl) as fi, open(a.out, "w") as fo:
    for line in fi:
        r = json.loads(line)
        chrom, st, en = r["chrom"], int(r["start"]), int(r["end"])
        ltr_n = 0
        bp = 0
        orders = set()
        is_pos = False
        for hs, he, is_ltr, order in hits.get(chrom, []):
            if hs >= en or he <= st:
                continue
            ov = min(he, en) - max(hs, st)
            if ov <= 0:
                continue
            if is_ltr:
                ltr_n += 1
                bp += ov
                if order:
                    orders.add(order)
                if ov >= thr(hs, he):
                    is_pos = True
        new = dict(r)
        new["label"] = 1 if is_pos else 0
        new["n_TE"] = ltr_n
        new["bp_TE"] = bp
        new["coverage_fraction"] = round(bp / W, 4)
        new["orders_present"] = sorted(orders)
        new["ltr_only_scheme"] = True
        n_in += 1
        if r.get("label") == 1 and not is_pos:
            n_flip += 1
        if is_pos:
            n_pos += 1
        fo.write(json.dumps(new) + "\n")
print(f"{a.jsonl}: total={n_in} ltr_pos={n_pos} flip={n_flip}")
