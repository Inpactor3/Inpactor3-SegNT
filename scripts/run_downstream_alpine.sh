#!/bin/bash
# Downstream en ALPINE: BED + sizes -> ventanas JSONL.
# Estructura esperada:
#   ~/InpactorRM_downstream/beds/<Reino>/<split>/*.bed
#   ~/InpactorRM_downstream/sizes/<Reino>/<split>/*.tsv
#   ~/InpactorRM_downstream/out/<Reino>/<split>/*.jsonl
set -u
B=/home/cami/InpactorRM_downstream
mkdir -p "$B/out"
LOG=$B/downstream.log
echo "[$(date '+%H:%M:%S')] downstream inicio" >> "$LOG"

for reino in Animales Plantas; do
    for split in train val test; do
        bdir="$B/beds/$reino/$split"
        sdir="$B/sizes/$reino/$split"
        odir="$B/out/$reino/$split"
        mkdir -p "$odir"
        for bed in "$bdir"/*.bed; do
            [ -e "$bed" ] || continue
            tag=$(basename "$bed" .bed)
            sz="$sdir/$tag.tsv"
            out="$odir/$tag.jsonl"
            [ -s "$out" ] && continue
            if [ ! -s "$sz" ]; then
                echo "SKIP $reino/$split/$tag (sin sizes)" >> "$LOG"
                continue
            fi
            python3 "$B/build_windows.py" "$bed" "$sz" "$out" "$reino" "$split" >> "$LOG" 2>&1
        done
    done
done
echo "[$(date '+%H:%M:%S')] downstream FIN" >> "$LOG"
