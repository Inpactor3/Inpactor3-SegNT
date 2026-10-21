#!/bin/bash
# Pipeline downstream para los hongos anotados con MMseqs2.
# Espera a que termine el batch de MMseqs2 y procesa cada hongo:
#   1. Generar ignore mask (buffer 1 kb alrededor de TE hits)
#   2. Generar ventanas de 50 kb con etiquetas positivo/negativo
#
# Nota: sample_5_negative_types se omite por ahora (requiere GFF de genes
# que no tenemos para los 117 hongos). Se agregara despues.
set -u

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
EXTERNAL=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM
GENOMES=$EXTERNAL/genomes_449/Hongos
BEDS=$EXTERNAL/annotations/Hongos
WIN_OUT=$EXTERNAL/windows/Hongos
LOGDIR=$EXTERNAL/logs/pipeline_hongos

mkdir -p "$WIN_OUT" "$LOGDIR"
MASTER_LOG="$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] Pipeline downstream Hongos - inicio" | tee -a "$MASTER_LOG"

# Esperar a que termine el batch de MMseqs2
echo "[WAIT] Esperando que termine MMseqs2 batch..." | tee -a "$MASTER_LOG"
while pgrep -f "run_mmseqs_hongos_batch" > /dev/null; do
    sleep 30
done
echo "[OK]   MMseqs2 batch terminado · empezando downstream" | tee -a "$MASTER_LOG"

# Parametros ventanas
WINDOW_SIZE=50000
MIN_OVERLAP_BP=50
MIN_OVERLAP_FRAC=0.2
BUFFER_BP=1000

N_OK=0; N_FAIL=0
IDX=0
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
            echo "[$IDX/$N_TOTAL] SKIP $BASE (ya procesado)" | tee -a "$MASTER_LOG"
            continue
        fi

        T0=$(date +%s)
        echo "[$IDX/$N_TOTAL] PROC $BASE" | tee -a "$MASTER_LOG"

        # Generar ventanas con Python inline
        python3 - "$GZ" "$BED" "$WIN_JSONL" "$BASE" "$split" "Hongos" \
            "$WINDOW_SIZE" "$MIN_OVERLAP_BP" "$MIN_OVERLAP_FRAC" <<'PYEOF' 2>> "$MASTER_LOG"
import gzip, json, sys
from collections import defaultdict

gz_file, bed_file, out_file, species, split, reino, \
    window_size, min_overlap_bp, min_overlap_frac = sys.argv[1:10]
window_size = int(window_size)
min_overlap_bp = int(min_overlap_bp)
min_overlap_frac = float(min_overlap_frac)

# Leer tamaños de cromosomas (nombres y longitudes)
sizes = {}
cur = None
acc = 0
with gzip.open(gz_file, "rt") as f:
    for line in f:
        if line.startswith(">"):
            if cur: sizes[cur] = acc
            cur = line[1:].strip().split()[0]
            acc = 0
        else:
            acc += len(line.strip())
if cur: sizes[cur] = acc

# Leer TE hits del BED
hits = defaultdict(list)
with open(bed_file) as f:
    for line in f:
        s = line.strip().split("\t")
        if len(s) < 4: continue
        chrom = s[0]
        start = int(s[1])
        end = int(s[2])
        name = s[3]  # formato: PDB...|CLASSI/LTR/GYPSY
        hits[chrom].append((start, end, name))

# Teselar y etiquetar
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
                    cls_tokens = tokens[1].split("/")
                    if len(cls_tokens) >= 2:
                        orders.add(cls_tokens[1])
                elem_len = e - s
                threshold = max(min_overlap_bp, int(min_overlap_frac * elem_len))
                if overlap >= threshold:
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
                "coverage_fraction": round(bp_total / window_size, 4),
                "orders_present": sorted(orders),
            }
            fo.write(json.dumps(rec) + "\n")
            n_total += 1
            if is_pos: n_pos += 1
            else:      n_neg += 1

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
echo "[$(date '+%H:%M:%S')] === PIPELINE DOWNSTREAM COMPLETADO ===" | tee -a "$MASTER_LOG"
echo "  Total:  $IDX" | tee -a "$MASTER_LOG"
echo "  OK:     $N_OK" | tee -a "$MASTER_LOG"
echo "  FAIL:   $N_FAIL" | tee -a "$MASTER_LOG"
