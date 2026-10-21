#!/bin/bash
# Descarga genomas al disco externo usando el manifest TSV.
#
# Uso:
#   bash scripts/download_species_batch.sh <reino> <split>
#   bash scripts/download_species_batch.sh Hongos all
#   bash scripts/download_species_batch.sh all all
set -u

REINO="${1:-all}"
SPLIT="${2:-all}"

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
MANIFEST="$PROJECT/docs/species_download_manifest.tsv"
TARGET_BASE="/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/genomes_449"
LOGDIR="/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/logs"
mkdir -p "$LOGDIR"

STAMP=$(date +%H%M%S)
LOG="$LOGDIR/download_${REINO}_${SPLIT}_${STAMP}.log"

echo "[$(date '+%H:%M:%S')] Iniciando descarga: reino=$REINO split=$SPLIT" | tee -a "$LOG"
echo "Target: $TARGET_BASE" | tee -a "$LOG"

n_total=0
n_ok=0
n_skip=0
n_fail=0

# Procesar manifest
while IFS=$'\t' read -r split reino especie status accession level cat size url; do
    [ "$split" = "split" ] && continue  # header
    [ "$status" != "found" ] && continue
    # Filtros
    [ "$REINO" != "all" ] && [ "$reino" != "$REINO" ] && continue
    [ "$SPLIT" != "all" ] && [ "$split" != "$SPLIT" ] && continue

    n_total=$((n_total + 1))

    # Nombre de archivo seguro (sin espacios ni caracteres raros)
    sp_safe=$(echo "$especie" | tr ' ' '_' | tr -c 'A-Za-z0-9._-' '_')
    out_dir="$TARGET_BASE/$reino/$split"
    mkdir -p "$out_dir"
    out_file="$out_dir/${sp_safe}_${accession}.fna.gz"

    if [ -f "$out_file" ] && [ -s "$out_file" ]; then
        echo "[SKIP] $especie (ya existe)" | tee -a "$LOG"
        n_skip=$((n_skip + 1))
        continue
    fi

    echo "[DL]   $especie  ($size MB)  $accession" | tee -a "$LOG"
    curl -sLk --max-time 1800 -o "$out_file" "$url" 2>&1
    if [ $? -eq 0 ] && [ -s "$out_file" ]; then
        actual_mb=$(du -m "$out_file" | cut -f1)
        echo "       OK ($actual_mb MB)" | tee -a "$LOG"
        n_ok=$((n_ok + 1))
    else
        echo "       FAIL" | tee -a "$LOG"
        rm -f "$out_file"
        n_fail=$((n_fail + 1))
    fi
done < "$MANIFEST"

echo "" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] COMPLETADO" | tee -a "$LOG"
echo "  Total intentadas: $n_total" | tee -a "$LOG"
echo "  OK:               $n_ok" | tee -a "$LOG"
echo "  Skip (ya exist):  $n_skip" | tee -a "$LOG"
echo "  Fail:             $n_fail" | tee -a "$LOG"
