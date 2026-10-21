import gzip, sys
fa, out = sys.argv[1], sys.argv[2]
sizes = {}
cur, acc = None, 0
with open(fa) as f:
    for line in f:
        if line.startswith('>'):
            if cur: sizes[cur] = acc
            cur = line[1:].strip().split()[0]
            acc = 0
        else:
            acc += len(line.strip())
if cur: sizes[cur] = acc
with open(out, 'w') as fo:
    for c, L in sizes.items():
        fo.write(f"{c}\t{L}\n")
print(f"{out}: {len(sizes)} cromosomas")
