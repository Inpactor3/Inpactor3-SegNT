#!/bin/bash
# Pilotos ADICIONALES de RepeatMasker (para despues del viernes).
#
# Uso:
#   bash scripts/run_extra_rm.sh <especie>
#   bash scripts/run_extra_rm.sh all
#
# Especies disponibles:
#   oryza       -> arroz (370 MB)       ~5-8 h
#   zea         -> maiz (2.5 GB)        ~24-48 h
#   ananas      -> pina (500 MB)        ~8-12 h
#   sorghum     -> sorgo (730 MB)       ~10-15 h
#   solanum     -> tomate (900 MB)      ~12-18 h
#   aspergillus -> hongo (29 MB)        ~15-30 min
#   spombe      -> hongo (12 MB)        ~5 min
#   danio       -> pez cebra (1.5 GB)   ~15-25 h
#   mus         -> raton (2.5 GB)       ~24-48 h
set -e

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
cd "$PROJECT"

RM=/home/cami/miniconda3/envs/Inpactor2/bin/RepeatMasker
export PATH="/home/cami/miniconda3/envs/Inpactor2/bin:$PATH"

LOGDIR=$PROJECT/logs/rm_extra
OUT_BASE=$PROJECT/results/rm_pilots
mkdir -p "$LOGDIR"

# Mapeo especie -> (genoma, library)
declare -A GENOME LIB
GENOME[oryza]="$PROJECT/data/validation_genomes/Oryza_sativa.fasta"
GENOME[zea]="$PROJECT/data/validation_genomes/Zea_mays.fasta"
GENOME[ananas]="$PROJECT/data/validation_genomes/Ananas_comosus.fasta"
GENOME[sorghum]="$PROJECT/data/pilot_genomes/sorghum.fasta"
GENOME[solanum]="$PROJECT/data/pilot_genomes/solanum.fasta"
GENOME[aspergillus]="$PROJECT/data/pilot_genomes/aspergillus.fasta"
GENOME[spombe]="$PROJECT/data/pilot_genomes/spombe.fasta"
GENOME[danio]="$PROJECT/data/pilot_genomes/danio.fasta"
GENOME[mus]="$PROJECT/data/pilot_genomes/mus.fasta"

LIB[oryza]="$PROJECT/data/panteon_libraries/panteon_oryza.fasta"
LIB[zea]="$PROJECT/data/panteon_libraries/panteon_zea.fasta"
LIB[ananas]="$PROJECT/data/panteon_libraries/panteon_plantae.fasta"       # fallback
LIB[sorghum]="$PROJECT/data/panteon_libraries/panteon_sorghum.fasta"
LIB[solanum]="$PROJECT/data/panteon_libraries/panteon_solanum.fasta"
LIB[aspergillus]="$PROJECT/data/panteon_libraries/panteon_fungi.fasta"    # fallback
LIB[spombe]="$PROJECT/data/panteon_libraries/panteon_fungi.fasta"         # fallback
LIB[danio]="$PROJECT/data/panteon_libraries/panteon_danio.fasta"
LIB[mus]="$PROJECT/data/panteon_libraries/panteon_animalia.fasta"         # fallback


run_one () {
    local NAME=$1
    local G=${GENOME[$NAME]}
    local L=${LIB[$NAME]}
    if [ -z "$G" ] || [ -z "$L" ]; then
        echo "ERROR: especie desconocida: $NAME"; return 1
    fi
    if [ ! -f "$G" ]; then
        echo "ERROR: no existe el genoma $G"; return 1
    fi
    local OUT=$OUT_BASE/$NAME
    mkdir -p "$OUT"
    echo ""
    echo "========================================"
    echo "RepeatMasker: $NAME"
    echo "  genoma: $G"
    echo "  lib:    $L"
    echo "  salida: $OUT"
    echo "========================================"
    local T0=$(date +%s)
    "$RM" -lib "$L" -pa 4 -gff -no_is -dir "$OUT" "$G" \
        > "$LOGDIR/${NAME}.log" 2>&1
    local T1=$(date +%s)
    local DT=$((T1 - T0))
    echo "   -> ${NAME} completado en ${DT}s"
}


if [ "$1" = "all" ]; then
    SPECIES="aspergillus spombe oryza sorghum ananas solanum danio zea mus"
else
    SPECIES="$1"
fi

for sp in $SPECIES; do
    run_one "$sp"
done
echo ""
echo "=== TODOS LOS EXTRA COMPLETADOS ==="
