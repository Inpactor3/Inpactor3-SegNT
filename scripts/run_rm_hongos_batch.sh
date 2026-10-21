#!/bin/bash
# Corre RepeatMasker sobre todos los Hongos descargados.
# Para cada especie:
#   1. Descomprime .fna.gz a /tmp (temporal)
#   2. Decide biblioteca: propia (si >=30 consensos) o Fungi fallback
#   3. Corre RM
#   4. Guarda salida en el disco externo
#   5. Borra .fasta temporal (ahorrar espacio)
set -u

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
GENOMES=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/genomes_449/Hongos
ANN_OUT=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/annotations/Hongos
TMP_DIR=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/tmp_fasta
LIB_DIR=$PROJECT/data/panteon_libraries
LOGDIR=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/logs/rm_hongos

RM=/home/cami/miniconda3/envs/Inpactor2/bin/RepeatMasker
export PATH="/home/cami/miniconda3/envs/Inpactor2/bin:$PATH"

mkdir -p "$ANN_OUT" "$TMP_DIR" "$LOGDIR"
MASTER_LOG="$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] RM Hongos batch - inicio" | tee -a "$MASTER_LOG"

# Fallback library
LIB_FUNGI="$LIB_DIR/panteon_fungi.fasta"

# Procesar cada genoma descargado
for split in train val test; do
    [ ! -d "$GENOMES/$split" ] && continue
    for GZ in "$GENOMES/$split"/*.fna.gz; do
        [ ! -f "$GZ" ] && continue

        BASE=$(basename "$GZ" .fna.gz)
        # Nombre de especie: antes del __accession
        SP_SAFE=$(echo "$BASE" | sed 's/__GC[AF]_.*//')
        SP_FULL=$(echo "$SP_SAFE" | tr '_' ' ')

        OUT_DIR="$ANN_OUT/$split/$BASE"
        if [ -f "$OUT_DIR"/*.out ]; then
            echo "[SKIP] $BASE (ya tiene .out)" | tee -a "$MASTER_LOG"
            continue
        fi
        mkdir -p "$OUT_DIR"

        # 1. Decidir biblioteca
        LIB_PROPIA="$LIB_DIR/panteon_${SP_SAFE,,}.fasta"
        if [ ! -f "$LIB_PROPIA" ]; then
            # Filtrar PanTEon para esta especie
            python3 "$PROJECT/scripts/filter_panteon_by_species.py" "$SP_FULL" "$LIB_PROPIA" >> "$MASTER_LOG" 2>&1
        fi
        # Si <30 consensos, usar fallback de Fungi
        N_CONS=$(grep -c "^>" "$LIB_PROPIA" 2>/dev/null || echo 0)
        if [ "$N_CONS" -lt 30 ]; then
            LIB_USE="$LIB_FUNGI"
            LIB_TYPE="fungi_fallback"
        else
            LIB_USE="$LIB_PROPIA"
            LIB_TYPE="especie_propia"
        fi

        # 2. Descomprimir temporal
        FASTA_TMP="$TMP_DIR/${BASE}.fasta"
        if [ ! -f "$FASTA_TMP" ]; then
            gunzip -c "$GZ" > "$FASTA_TMP"
        fi

        # 3. RM
        T0=$(date +%s)
        echo "[RM]   $SP_FULL · ${LIB_TYPE} (${N_CONS} cons)" | tee -a "$MASTER_LOG"
        "$RM" -lib "$LIB_USE" -pa 4 -gff -no_is -dir "$OUT_DIR" "$FASTA_TMP" \
            > "$LOGDIR/${BASE}.log" 2>&1
        RC=$?
        T1=$(date +%s)
        DT=$((T1 - T0))

        if [ $RC -eq 0 ] && [ -f "$OUT_DIR"/*.out ]; then
            echo "       OK ${DT}s" | tee -a "$MASTER_LOG"
            # Borrar FASTA temporal
            rm -f "$FASTA_TMP"
        else
            echo "       FAIL ${DT}s" | tee -a "$MASTER_LOG"
        fi
    done
done

echo "[$(date '+%H:%M:%S')] RM Hongos batch - fin" | tee -a "$MASTER_LOG"
rm -rf "$TMP_DIR" 2>/dev/null
