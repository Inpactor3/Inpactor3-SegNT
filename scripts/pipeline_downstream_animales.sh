#!/bin/bash
# Downstream pipeline Animales: espera MMseqs2 y genera ventanas 50 kb.
set -u

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
EXTERNAL=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM
GENOMES=$EXTERNAL/genomes_449/Animales
BEDS=$EXTERNAL/annotations/Animales
WIN_OUT=$EXTERNAL/windows/Animales
LOGDIR=$EXTERNAL/logs/pipeline_animales

mkdir -p "$WIN_OUT" "$LOGDIR"
MASTER_LOG="$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] Pipeline downstream Animales - inicio" | tee -a "$MASTER_LOG"

echo "[WAIT] Esperando que termine MMseqs2 Animales batch..." | tee -a "$MASTER_LOG"
while pgrep -f "run_mmseqs_animales_batch" > /dev/null; do
    sleep 30
done
echo "[OK]   MMseqs2 Animales terminado · empezando downstream" | tee -a "$MASTER_LOG"

WINDOW_SIZE=50000
MIN_OVERLAP_BP=50
MIN_OVERLAP_FRAC=0.2

N_OK=0; N_FAIL=0; IDX=0
N_TOTAL=$(find "$BEDS" -name "*.bed" 2>/dev/null | wc -l)

for split in train val test; do
    [ ! -d "$BEDS/$split" ] && continue
    for BED in "$BEDS/$split"/*.bed; do
        [ ! -f "$BED" ] && continue
        IDX=$((IDX + 1))

        BASE=$(basename "$BED" .bed)
        GZ="$GENOMES/$split/${BASE}.fna.gz"
        [ ! -f "$GZ" ] && { echo "[$IDX/$N_TOTAL] SKIP $BASE (no genoma)" | tee -a "$MASTER_LOG"; N_FAIL=$((N_FAIL+1)); continue; }

        WIN_JSONL="$WIN_OUT/$split/${BASE}.jsonl"
        mkdir -p "$WIN_OUT/$split"

        if [ -f "$WIN_JSONL" ] && [ -s "$WIN_JSONL" ]; then
            echo "[$IDX/$N_TOTAL] SKIP $BASE" | tee -a "$MASTER_LOG"
            continue
        fi

        T0=$(date +%s)
        echo "[$IDX/$N_TOTAL] PROC $BASE" | tee -a "$MASTER_LOG"

        python3 - "$GZ" "$BED" "$WIN_JSONL" "$BASE" "$split" "Animales" \
            "$WINDOW_SIZE" "$MIN_OVERLAP_BP" "$MIN_OVERLAP_FRAC" <<'PYEOF' 2>> "$MASTER_LOG"
import gzip, json, sys
from collections import defaultdict

gz_file, bed_file, out_file, species, split, reino, \
    window_size, min_overlap_bp, min_overlap_frac = sys.argv[1:10]
window_size = int(window_size)
min_overlap_bp = int(min_overlap_bp)
min_overlap_frac = float(min_overlap_frac)

sizes = {}
cur, acc = None, 0
with gzip.open(gz_file, "rt") as f:
    for line in f:
        if line.startswith(">"):
            if cur: sizes[cur] = acc
            cur = line[1:].strip().split()[0]
            acc = 0
        else:
            acc += len(line.strip())
if cur: sizes[cur] = acc

hits = defaultdict(list)
with open(bed_file) as f:
    for line in f:
        s = line.strip().split("\t")
        if len(s) < 4: continue
        hits[s[0]].append((int(s[1]), int(s[2]), s[3]))

n_pos = n_neg = n_total = 0
with open(out_file, "w") as fo:
    for chrom, L in sizes.items():
        for start in range(0, L - window_size + 1, window_size):
            end = start + window_size
            te_in = [(s, e, n) for (s, e, n) in hits.get(chrom, [])
                     if s < end and e > start]
            is_pos = False
            bp_total = 0
            orders = set()
            for s, e, name in te_in:
                overlap = min(e, end) - max(s, start)
                if overlap <= 0: continue
                bp_total += overlap
                tokens = name.split("|")
                if len(tokens) >= 2:
                    ct = tokens[1].split("/")
                    if len(ct) >= 2:
                        orders.add(ct[1])
                elem_len = e - s
                threshold = max(min_overlap_bp, int(min_overlap_frac * elem_len))
                if overlap >= threshold: is_pos = True
            rec = {"window_id": f"{species}_{chrom}_{start}",
                   "species": species, "reino": reino, "split": split,
                   "chrom": chrom, "start": start, "end": end,
                   "label": 1 if is_pos else 0,
                   "n_TE": len(te_in), "bp_TE": bp_total,
                   "coverage_fraction": round(bp_total / window_size, 4),
                   "orders_present": sorted(orders)}
            fo.write(json.dumps(rec) + "\n")
            n_total += 1
            if is_pos: n_pos += 1
            else: n_neg += 1
print(f"  pos={n_pos} neg={n_neg} total={n_total}")
PYEOF

        if [ -s "$WIN_JSONL" ]; then
            T1=$(date +%s); DT=$((T1 - T0))
            N_OK=$((N_OK + 1))
            N_WIN=$(wc -l < "$WIN_JSONL")
            echo "           OK ${DT}s · ${N_WIN} ventanas" | tee -a "$MASTER_LOG"
        else
            N_FAIL=$((N_FAIL + 1))
            echo "           FAIL" | tee -a "$MASTER_LOG"
        fi
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "[$(date '+%H:%M:%S')] === DOWNSTREAM ANIMALES COMPLETADO ===" | tee -a "$MASTER_LOG"
echo "  Total: $IDX" | tee -a "$MASTER_LOG"
echo "  OK:    $N_OK" | tee -a "$MASTER_LOG"
echo "  FAIL:  $N_FAIL" | tee -a "$MASTER_LOG"
