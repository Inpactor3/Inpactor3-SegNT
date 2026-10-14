#!/bin/bash
# Lanza los 3 pilotos de RepeatMasker con PanTEon por especie.
# Sale sin ejecutar nada si algo falla.
set -e

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
cd "$PROJECT"

RM=/home/cami/miniconda3/envs/Inpactor2/bin/RepeatMasker
export PATH="/home/cami/miniconda3/envs/Inpactor2/bin:$PATH"

LOGDIR=$PROJECT/logs/rm_pilots
mkdir -p "$LOGDIR"

OUT_BASE=$PROJECT/results/rm_pilots
mkdir -p "$OUT_BASE"

run_one () {
    local NAME=$1
    local GENOME=$2
    local LIB=$3
    local NCPU=${4:-4}
    local OUT=$OUT_BASE/$NAME
    mkdir -p "$OUT"
    echo ""
    echo "========================================"
    echo "RepeatMasker piloto: $NAME"
    echo "  genoma: $GENOME"
    echo "  lib:    $LIB"
    echo "  cpu:    $NCPU"
    echo "  salida: $OUT"
    echo "========================================"
    local T0=$(date +%s)
    "$RM" -lib "$LIB" -pa $NCPU -gff -no_is -dir "$OUT" "$GENOME" \
        > "$LOGDIR/${NAME}.log" 2>&1
    local T1=$(date +%s)
    local DT=$((T1 - T0))
    echo "   -> completado en ${DT}s"
}

# Piloto 1: Arabidopsis thaliana (planta, 120 Mb, 217 consensos)
run_one arabidopsis \
    "$PROJECT/data/tair10_full/TAIR10.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_arabidopsis.fasta"

# Piloto 2: S. cerevisiae (hongo, 12 Mb, 5 consensos)
run_one scerevisiae \
    "$PROJECT/data/pilot_genomes/scerevisiae.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_scerevisiae.fasta"

# Piloto 3: C. elegans (animal, 100 Mb, 139 consensos)
run_one celegans \
    "$PROJECT/data/pilot_genomes/celegans.fasta" \
    "$PROJECT/data/panteon_libraries/panteon_celegans.fasta"

echo ""
echo "=== TODOS LOS PILOTOS COMPLETADOS ==="
echo "Resultados: $OUT_BASE/"
echo "Logs:       $LOGDIR/"
