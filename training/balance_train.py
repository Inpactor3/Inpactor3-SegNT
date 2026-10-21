"""Balancea el split de entrenamiento: cap de positivas por especie.

Mantiene val/test intactos. Para cada especie del split elegido:
  - lee sus JSONL de windows_ltr (positivas y negativas)
  - lee sus JSONL de windows_neg (negativas adicionales)
  - conserva todas las negativas y muestrea positivas hasta ratio * n_neg
  - si la especie no tiene negativas, conserva como maximo `--orphan_cap` positivas

Uso:
  python balance_train.py --reino Plantas --ratio 2 --splits train
"""
import argparse, glob, json, os, random
from pathlib import Path

D = Path("/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM")

def read_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reino", required=True)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--orphan_cap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default=str(D / "consolidated_balanced"))
    args = ap.parse_args()
    random.seed(args.seed)

    total_p = total_n = 0
    for split in args.splits.split(","):
        src_dir = D / "consolidated" / args.reino / split
        out_dir = Path(args.out_root) / args.reino / split
        out_dir.mkdir(parents=True, exist_ok=True)
        species = sorted(p.stem for p in src_dir.glob("*.jsonl"))
        for sp in species:
            rows = read_jsonl(src_dir / f"{sp}.jsonl")
            pos = [r for r in rows if r.get("label") == 1]
            neg = [r for r in rows if r.get("label") != 1]
            if neg:
                k = int(args.ratio * len(neg))
                keep = pos if len(pos) <= k else random.sample(pos, k)
            else:
                keep = pos[: args.orphan_cap]
            with open(out_dir / f"{sp}.jsonl", "w") as f:
                for r in keep + neg:
                    f.write(json.dumps(r) + "\n")
            total_p += len(keep); total_n += len(neg)
            print(f"{split}/{sp}: pos={len(keep)} neg={len(neg)}")
    print(f"TOTAL {args.reino}: pos={total_p} neg={total_n} "
          f"ratio=1:{total_p/max(total_n,1):.2f}")

if __name__ == "__main__":
    main()
