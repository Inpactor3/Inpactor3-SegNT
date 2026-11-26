#!/usr/bin/env python3
"""Construye un test species-disjoint que cubra mas grupos taxonomicos.

No recalcula embeddings: reutiliza la cache existente y reubica especies
(por defecto con enlaces simbolicos) en un directorio nuevo.

Reglas:
  * El test actual se conserva, salvo especies degeneradas (todo positivo o
    todo negativo), que se excluyen.
  * Para cada grupo sin especie de test evaluable se intenta mover:
      - una especie de train, si el grupo tiene >= 2 especies en train
      - si no, una especie de val, si el grupo tiene >= 1 en train
  * Una especie movida solo se acepta si tiene al menos --min_class positivos
    y --min_class negativos (para que el PR-AUC del grupo sea medible).
  * Nunca se deja un grupo sin ninguna especie de train.

Uso:
  python resplit_test.py --emb_dir trabajo/ntv3_animales_full/emb \
      --group_csv docs/animales_taxonomia_v3.csv \
      --out_dir trabajo/ntv3_resplit/Animales
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ACC = re.compile(r"(GC[AF]_\d+\.\d+)")


def norm_tag(s):
    return re.sub(r"_+", "_", s.lower())


def load_groups(path):
    exact, norm, by_acc = {}, {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sp = (row.get("species") or row.get("especie") or "").strip()
            grp = (row.get("grupo") or row.get("group") or "sin_grupo").strip()
            if not sp:
                continue
            exact[sp] = grp
            norm[norm_tag(sp)] = grp
            m = ACC.search(sp)
            if m:
                by_acc[m.group(1)] = grp
    return exact, norm, by_acc


def group_for(stem, maps):
    exact, norm, by_acc = maps
    if stem in exact:
        return exact[stem]
    if norm_tag(stem) in norm:
        return norm[norm_tag(stem)]
    m = ACC.search(stem)
    if m and m.group(1) in by_acc:
        return by_acc[m.group(1)]
    return "sin_grupo"


def list_species(emb_dir, split):
    out = {}
    for f in sorted((emb_dir / split).glob("*_emb.npy")):
        stem = f.name[:-8]
        lab = f.with_name(f.name.replace("_emb.npy", "_labels.npy"))
        out[stem] = {"emb": f, "labels": lab if lab.exists() else None}
    return out


def stats(item):
    if item["labels"] is None:
        return 0, 0, 0
    y = np.load(item["labels"])
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return len(y), pos, neg


def place(item, dst, copy=False):
    dst.mkdir(parents=True, exist_ok=True)
    for key in ("emb", "labels"):
        src = item.get(key)
        if src is None:
            continue
        target = dst / src.name
        if target.exists():
            continue
        if copy:
            shutil.copy2(src, target)
        else:
            os.symlink(src.resolve(), target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=Path, required=True)
    ap.add_argument("--group_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--min_class", type=int, default=50,
                    help="positivos y negativos minimos en una especie movida al test")
    ap.add_argument("--copy", action="store_true", help="copiar en vez de enlazar")
    ap.add_argument("--force", action="store_true", help="sobrescribir out_dir")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    maps = load_groups(args.group_csv)
    source = {s: list_species(args.emb_dir, s) for s in ("train", "val", "test")}
    info = {}
    for split, items in source.items():
        for stem, item in items.items():
            n, pos, neg = stats(item)
            info[stem] = {"split": split, "group": group_for(stem, maps),
                          "n": n, "pos": pos, "neg": neg, **item}

    degenerate = {s for s, d in info.items() if d["n"] > 0 and (d["pos"] == 0 or d["neg"] == 0)}
    if degenerate:
        print(f"AVISO: {len(degenerate)} especies degeneradas se excluyen: "
              f"{', '.join(sorted(degenerate))}", flush=True)

    unmapped = [s for s, d in info.items() if d["group"] == "sin_grupo"]
    if unmapped:
        print(f"AVISO: {len(unmapped)} especies sin grupo, se omiten: "
              f"{', '.join(unmapped[:6])}{' ...' if len(unmapped) > 6 else ''}", flush=True)

    by_group = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for stem, d in info.items():
        if d["group"] != "sin_grupo" and stem not in degenerate:
            by_group[d["group"]][d["split"]].append(stem)

    def evaluable(stem, min_class=1):
        d = info[stem]
        return d["pos"] >= min_class and d["neg"] >= min_class

    # Test que se conserva: no degenerado
    keep_test = {s for s in source["test"] if s in info and evaluable(s, 1)}
    drop_test = sorted(set(source["test"]) - keep_test)
    test_groups = {info[s]["group"] for s in keep_test}

    moved = {}          # stem -> split origen
    reasons = {}
    for group in sorted(by_group):
        if group in test_groups:
            continue
        tr = [s for s in by_group[group]["train"] if s in info]
        va = [s for s in by_group[group]["val"] if s in info]
        tr_eval = [s for s in tr if evaluable(s, args.min_class)]
        va_eval = [s for s in va if evaluable(s, args.min_class)]

        pick, origin = None, None
        if len(tr) >= 2 and tr_eval:
            # la especie mas pequena que siga dejando train para el grupo
            pick = min(tr_eval, key=lambda s: (info[s]["n"], s))
            origin = "train"
        elif len(tr) >= 1 and va_eval:
            pick = min(va_eval, key=lambda s: (info[s]["n"], s))
            origin = "val"

        if pick:
            moved[pick] = origin
            reasons[group] = f"movida de {origin}: {pick}"
            test_groups.add(group)
        else:
            if group in by_group and not by_group[group]["train"]:
                reasons[group] = "sin especies en train"
            elif len(tr) < 2 and not va_eval:
                reasons[group] = "solo 1 especie en train y sin val evaluable"
            else:
                reasons[group] = "sin candidato con clases suficientes"

    print("\nMovimientos al test:")
    if not moved:
        print("  (ninguno)")
    for stem, origin in sorted(moved.items(), key=lambda kv: info[kv[0]]["group"]):
        d = info[stem]
        print(f"  {d['group']:28s} {stem:60s} n={d['n']:6d} "
              f"pos={d['pos']:5d} neg={d['neg']:5d} ({origin})")

    if drop_test:
        print("\nTest degenerado excluido: " + ", ".join(drop_test))

    # Reparto final
    final = {"train": [], "val": [], "test": []}
    for split in ("train", "val"):
        for stem in source[split]:
            if stem in info and stem not in moved and stem not in degenerate:
                final[split].append(stem)
    final["test"] = [s for s in source["test"] if s in keep_test] + sorted(moved)

    # Resumen por grupo
    print("\nCobertura por grupo (train / val / test):")
    rows = []
    for group in sorted(by_group):
        counts = {s: 0 for s in ("train", "val", "test")}
        for stem in final["train"]:
            if info[stem]["group"] == group:
                counts["train"] += 1
        for stem in final["val"]:
            if info[stem]["group"] == group:
                counts["val"] += 1
        for stem in final["test"]:
            if info[stem]["group"] == group:
                counts["test"] += 1
        tag = "" if counts["train"] else "  SIN TRAIN"
        print(f"  {group:28s} {counts['train']:3d} / {counts['val']:2d} / {counts['test']:2d}{tag}")
        rows.append({"grupo": group, **counts, "nota": reasons.get(group, "")})

    if not final["val"]:
        raise SystemExit("el val quedaria vacio; no se puede calibrar el umbral")

    if args.dry_run:
        print("\n(dry run: no se escribe nada)")
        return 0

    if args.out_dir.exists():
        if not args.force:
            raise SystemExit(f"{args.out_dir} existe; usa --force para reemplazarlo")
        shutil.rmtree(args.out_dir)
    for split in ("train", "val", "test"):
        for stem in final[split]:
            place(info[stem], args.out_dir / split, copy=args.copy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "split_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["grupo", "train", "val", "test", "nota"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCache nuevo: {args.out_dir}")
    print(f"train={len(final['train'])} val={len(final['val'])} test={len(final['test'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
