#!/bin/bash
# Segundo batch de pilotos RepeatMasker.
# Espera a que termine el batch 1 (C. elegans) y luego lanza Brachypodium,
# Neurospora y Drosophila. Despues de los 6 pilotos, genera el reporte.
set -e

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
cd "$PROJECT"

RM=/home/cami/miniconda3/envs/Inpactor2/bin/RepeatMasker
export PATH="/home/cami/miniconda3/envs/Inpactor2/bin:$PATH"

LOGDIR=$PROJECT/logs/rm_pilots
OUT_BASE=$PROJECT/results/rm_pilots

echo "[$(date '+%H:%M:%S')] batch2 esperando a que termine el batch1..."

# Esperar a que no haya mas procesos RM del batch1 corriendo
while pgrep -f "run_pilot_rm.sh" > /dev/null; do
    sleep 30
done
# Esperar un poco mas para asegurar que RM termine de limpiar
while pgrep -f "RepeatMasker" > /dev/null; do
    sleep 15
done

echo "[$(date '+%H:%M:%S')] batch1 terminado. Lanzando batch2..."
echo "" >> "$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] === INICIO BATCH 2 ===" >> "$LOGDIR/_master.log"


run_one () {
    local NAME=$1
    local GENOME=$2
    local LIB=$3
    local NCPU=${4:-4}
    local OUT=$OUT_BASE/$NAME
    mkdir -p "$OUT"
    echo "" >> "$LOGDIR/_master.log"
    echo "========================================" >> "$LOGDIR/_master.log"
    echo "RepeatMasker piloto: $NAME" >> "$LOGDIR/_master.log"
    echo "  genoma: $GENOME" >> "$LOGDIR/_master.log"
    echo "  lib:    $LIB" >> "$LOGDIR/_master.log"
    echo "  cpu:    $NCPU" >> "$LOGDIR/_master.log"
    echo "  salida: $OUT" >> "$LOGDIR/_master.log"
    echo "========================================" >> "$LOGDIR/_master.log"
    local T0=$(date +%s)
    "$RM" -lib "$LIB" -pa $NCPU -gff -no_is -dir "$OUT" "$GENOME" \
        > "$LOGDIR/${NAME}.log" 2>&1
    local T1=$(date +%s)
    local DT=$((T1 - T0))
    echo "   -> ${NAME} completado en ${DT}s" >> "$LOGDIR/_master.log"
}


# Piloto 4: Brachypodium (planta, 263 Mb, 223 consensos propios)
run_one brachypodium \
    "$PROJECT/data/pilot_genomes/brachypodium.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_brachypodium.fasta"

# Piloto 5: Neurospora (hongo, 40 Mb, 2 consensos propios -> fallback Fungi)
run_one neurospora \
    "$PROJECT/data/pilot_genomes/neurospora.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_fungi.fasta"

# Piloto 6: Drosophila (animal, 140 Mb, 14 consensos -> fallback Animalia)
run_one drosophila \
    "$PROJECT/data/pilot_genomes/drosophila.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_animalia.fasta"

echo "" >> "$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] === TODOS LOS 6 PILOTOS COMPLETADOS ===" >> "$LOGDIR/_master.log"

# Generar reporte final automaticamente
cd "$PROJECT"
.venv_slide/bin/python scripts/generate_pilot_report.py \
    > "$LOGDIR/report.log" 2>&1 \
    || python3 scripts/generate_pilot_report.py > "$LOGDIR/report.log" 2>&1

echo "[$(date '+%H:%M:%S')] Reporte generado en docs/piloto_rm_reporte.md" >> "$LOGDIR/_master.log"
