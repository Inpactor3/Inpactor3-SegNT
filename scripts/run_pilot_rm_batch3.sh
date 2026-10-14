#!/bin/bash
# Batch 3: pilotos nocturnos.
#
# Espera a que terminen los batches anteriores y lanza:
#   - S. pombe (5 min)
#   - Aspergillus nidulans (30 min)
#   - Oryza sativa (7 h -- corre durante la noche)
#
# Al terminar, limpia los .out generados y actualiza el reporte piloto.
set -e

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
cd "$PROJECT"

RM=/home/cami/miniconda3/envs/Inpactor2/bin/RepeatMasker
export PATH="/home/cami/miniconda3/envs/Inpactor2/bin:$PATH"

LOGDIR=$PROJECT/logs/rm_pilots
OUT_BASE=$PROJECT/results/rm_pilots
mkdir -p "$LOGDIR"

echo "[$(date '+%H:%M:%S')] batch3 esperando a que terminen batch1 y batch2..."

# Esperar a que terminen todos los procesos RM y los scripts de batches
while pgrep -f "run_pilot_rm_batch2.sh" > /dev/null; do
    sleep 60
done
while pgrep -f "RepeatMasker" > /dev/null; do
    sleep 30
done
# Esperar a que termine la descarga de Oryza si acaso (ya la tenemos pero por si acaso)
while pgrep -f "curl -Lk" > /dev/null; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] todos los batches anteriores terminaron. Lanzando batch3..."
echo "" >> "$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] === INICIO BATCH 3 ===" >> "$LOGDIR/_master.log"


run_one () {
    local NAME=$1
    local GENOME=$2
    local LIB=$3
    local OUT=$OUT_BASE/$NAME
    mkdir -p "$OUT"
    echo "" >> "$LOGDIR/_master.log"
    echo "========================================" >> "$LOGDIR/_master.log"
    echo "RepeatMasker batch3: $NAME" >> "$LOGDIR/_master.log"
    echo "  genoma: $GENOME" >> "$LOGDIR/_master.log"
    echo "  lib:    $LIB" >> "$LOGDIR/_master.log"
    echo "========================================" >> "$LOGDIR/_master.log"
    local T0=$(date +%s)
    "$RM" -lib "$LIB" -pa 4 -gff -no_is -dir "$OUT" "$GENOME" \
        > "$LOGDIR/${NAME}.log" 2>&1
    local T1=$(date +%s)
    local DT=$((T1 - T0))
    echo "   -> ${NAME} completado en ${DT}s" >> "$LOGDIR/_master.log"
}


# PILOTO 7: S. pombe (hongo tiny, 12 MB, 2 cons propios -> fallback Fungi)
run_one spombe \
    "$PROJECT/data/pilot_genomes/spombe.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_fungi.fasta"

# PILOTO 8: Aspergillus nidulans (hongo chico, 29 MB, 25 cons propios -> fallback Fungi)
run_one aspergillus \
    "$PROJECT/data/pilot_genomes/aspergillus.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_fungi.fasta"

# PILOTO 9: Oryza sativa (planta mediana, 370 MB, 1090 cons propios)
run_one oryza \
    "$PROJECT/data/validation_genomes/Oryza_sativa.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_oryza.fasta"

echo "" >> "$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] === TODOS LOS BATCH3 COMPLETADOS ===" >> "$LOGDIR/_master.log"

# Procesar los .out con clean_rm_output.py
cd "$PROJECT"
python3 scripts/clean_rm_output.py all \
    > "$LOGDIR/clean_batch3.log" 2>&1 || true

# Regenerar reporte piloto
.venv_slide/bin/python scripts/generate_pilot_report.py \
    > "$LOGDIR/report_batch3.log" 2>&1 \
    || python3 scripts/generate_pilot_report.py \
    > "$LOGDIR/report_batch3.log" 2>&1 || true

echo "[$(date '+%H:%M:%S')] batch3 completado -- BEDs y reporte actualizados" >> "$LOGDIR/_master.log"
