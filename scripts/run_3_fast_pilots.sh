#!/bin/bash
# 3 pilotos rapidos: S. pombe + Aspergillus + Drosophila con lib propia.
# Tiempo total esperado: ~1h 15min
set -e

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
cd "$PROJECT"

RM=/home/cami/miniconda3/envs/Inpactor2/bin/RepeatMasker
export PATH="/home/cami/miniconda3/envs/Inpactor2/bin:$PATH"

LOGDIR=$PROJECT/logs/rm_pilots
OUT_BASE=$PROJECT/results/rm_pilots
mkdir -p "$LOGDIR"

# Remover el output viejo de drosophila (interrumpido)
rm -rf "$OUT_BASE/drosophila"
mkdir -p "$OUT_BASE/drosophila"


run_one () {
    local NAME=$1
    local GENOME=$2
    local LIB=$3
    local OUT=$OUT_BASE/$NAME
    mkdir -p "$OUT"
    echo "" >> "$LOGDIR/_master.log"
    echo "========================================" >> "$LOGDIR/_master.log"
    echo "RM 3-fast: $NAME" >> "$LOGDIR/_master.log"
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


# PILOTO 6: S. pombe (12 MB, fallback Fungi)
run_one spombe \
    "$PROJECT/data/pilot_genomes/spombe.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_fungi.fasta"

# PILOTO 7: Aspergillus (29 MB, fallback Fungi)
run_one aspergillus \
    "$PROJECT/data/pilot_genomes/aspergillus.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_fungi.fasta"

# PILOTO 8: Drosophila con biblioteca PROPIA (14 cons, rapido)
run_one drosophila \
    "$PROJECT/data/pilot_genomes/drosophila.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_drosophila.fasta"

echo "" >> "$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] === 3 PILOTOS RAPIDOS COMPLETADOS ===" >> "$LOGDIR/_master.log"

# Limpieza + reporte automatico
cd "$PROJECT"
python3 scripts/clean_rm_output.py all > "$LOGDIR/clean_final.log" 2>&1 || true

if [ -f .venv_slide/bin/python ]; then
    .venv_slide/bin/python scripts/generate_pilot_report.py > "$LOGDIR/report_final.log" 2>&1 || true
else
    python3 scripts/generate_pilot_report.py > "$LOGDIR/report_final.log" 2>&1 || true
fi

echo "[$(date '+%H:%M:%S')] === BEDs y REPORTE actualizados ===" >> "$LOGDIR/_master.log"
