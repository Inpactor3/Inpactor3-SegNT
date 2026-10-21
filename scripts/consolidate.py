import argparse, glob, json, os, collections

ap = argparse.ArgumentParser()
ap.add_argument("--pos_dir", required=True)
ap.add_argument("--neg_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--reino", required=True)
a = ap.parse_args()

stats = {"pos": 0, "neg": 0, "dup": 0, "files": 0}
neg_types = collections.Counter()

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(a.out_dir, split), exist_ok=True)
    pos_files = {os.path.basename(f)[:-6]: f for f in glob.glob(os.path.join(a.pos_dir, split, "*.jsonl"))}
    neg_files = {os.path.basename(f)[:-6]: f for f in glob.glob(os.path.join(a.neg_dir, split, "*.jsonl"))}
    for species in sorted(set(pos_files) | set(neg_files)):
        out = os.path.join(a.out_dir, split, species + ".jsonl")
        ids = set()
        npos = nneg = ndup = 0
        with open(out, "w") as fo:
            pf = pos_files.get(species)
            if pf:
                for line in open(pf):
                    r = json.loads(line)
                    if r.get("label") != 1:
                        continue
                    wid = r.get("window_id")
                    if wid in ids:
                        ndup += 1
                        continue
                    ids.add(wid)
                    fo.write(json.dumps(r) + "\n")
                    npos += 1
            nf = neg_files.get(species)
            if nf:
                for line in open(nf):
                    r = json.loads(line)
                    wid = r.get("window_id")
                    if wid in ids:
                        ndup += 1
                        continue
                    ids.add(wid)
                    fo.write(json.dumps(r) + "\n")
                    nneg += 1
                    neg_types[r.get("negative_type", "?")] += 1
        stats["pos"] += npos
        stats["neg"] += nneg
        stats["dup"] += ndup
        stats["files"] += 1
        print(f"{split}/{species}: pos={npos} neg={nneg} dup={ndup}")

print(f"TOTAL {a.reino}: files={stats['files']} pos={stats['pos']} neg={stats['neg']} dup={stats['dup']} {dict(neg_types)}")
