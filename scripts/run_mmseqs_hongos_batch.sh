#!/bin/bash
# Anotacion de los 117 hongos usando MMseqs2 (local, 30x mas rapido que RepeatMasker).
set -u

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
GENOMES=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/genomes_449/Hongos
ANN_OUT=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/annotations/Hongos
# Trabajo temporal en /tmp (ext4) porque NTFS no permite exec de scripts
TMP_WORK=/tmp/mmseqs_hongos_work
LIB_DIR=$PROJECT/data/panteon_libraries
LOGDIR=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/logs/mmseqs_hongos

MMSEQS=$HOME/bin/mmseqs
LIB_FUNGI=$LIB_DIR/panteon_fungi.fasta

mkdir -p "$ANN_OUT" "$TMP_WORK" "$LOGDIR"
MASTER_LOG="$LOGDIR/_master.log"
echo "[$(date '+%H:%M:%S')] MMseqs2 Hongos batch - inicio" | tee -a "$MASTER_LOG"

# Construir ref_db una sola vez (reutilizable)
REF_DB="$TMP_WORK/panteon_fungi_db"
if [ ! -f "${REF_DB}.index" ]; then
    echo "[DB]  Construyendo ref_db de PanTEon Fungi (71 MB)..." | tee -a "$MASTER_LOG"
    "$MMSEQS" createdb "$LIB_FUNGI" "$REF_DB" >> "$MASTER_LOG" 2>&1
fi

T0_GLOBAL=$(date +%s)
N_OK=0; N_SKIP=0; N_FAIL=0
N_TOTAL=$(find "$GENOMES" -name "*.fna.gz" 2>/dev/null | wc -l)
IDX=0

for split in train val test; do
    [ ! -d "$GENOMES/$split" ] && continue
    for GZ in "$GENOMES/$split"/*.fna.gz; do
        [ ! -f "$GZ" ] && continue
        IDX=$((IDX + 1))

        BASE=$(basename "$GZ" .fna.gz)
        OUT_BED="$ANN_OUT/$split/${BASE}.bed"
        mkdir -p "$ANN_OUT/$split"

        if [ -f "$OUT_BED" ] && [ -s "$OUT_BED" ]; then
            echo "[$IDX/$N_TOTAL] SKIP $BASE" | tee -a "$MASTER_LOG"
            N_SKIP=$((N_SKIP + 1))
            continue
        fi

        T0=$(date +%s)
        echo "[$IDX/$N_TOTAL] RUN  $BASE" | tee -a "$MASTER_LOG"

        # 1. Descomprimir temporalmente
        FASTA_TMP="$TMP_WORK/${BASE}.fasta"
        gunzip -c "$GZ" > "$FASTA_TMP"

        # 2. Crear query DB
        Q_DB="$TMP_WORK/query_${BASE}"
        "$MMSEQS" createdb "$FASTA_TMP" "$Q_DB" > /dev/null 2>&1

        # 3. Search (parametros validados con test en S. pombe → 300 hits)
        RES_DB="$TMP_WORK/result_${BASE}"
        TMP_MMSEQS="$TMP_WORK/mmseqs_tmp_${BASE}"
        "$MMSEQS" search "$Q_DB" "$REF_DB" "$RES_DB" "$TMP_MMSEQS" \
            --search-type 3 --threads 4 -s 5.7 \
            > /dev/null 2>&1

        # 4. Convertir a tabular y luego BED
        RES_TAB="$TMP_WORK/result_${BASE}.m8"
        "$MMSEQS" convertalis "$Q_DB" "$REF_DB" "$RES_DB" "$RES_TAB" \
            --format-output 'query,target,pident,alnlen,qstart,qend,evalue' \
            > /dev/null 2>&1

        if [ -s "$RES_TAB" ]; then
            # Convertir m8 a BED
            awk -F'\t' 'BEGIN{OFS="\t"} {
                chrom=$1; target=$2; pident=$3; qs=$5; qe=$6;
                start = (qs < qe ? qs : qe) - 1;
                end   = (qs > qe ? qs : qe);
                n = split(target, parts, "#");
                te_id = parts[1];
                cls   = (n >= 2 ? parts[2] : "Unknown");
                print chrom, start, end, te_id "|" cls, pident, "."
            }' "$RES_TAB" > "$OUT_BED"

            N_HITS=$(wc -l < "$OUT_BED")
            T1=$(date +%s)
            DT=$((T1 - T0))
            echo "           OK  ${DT}s  ${N_HITS} hits" | tee -a "$MASTER_LOG"
            N_OK=$((N_OK + 1))
        else
            echo "           FAIL (sin hits)" | tee -a "$MASTER_LOG"
            N_FAIL=$((N_FAIL + 1))
        fi

        # 5. Limpieza del temporal del hongo
        rm -f "$FASTA_TMP"
        rm -rf "$Q_DB"* "$RES_DB"* "$TMP_MMSEQS" "$RES_TAB"
    done
done

T_END=$(date +%s)
DT_GLOBAL=$((T_END - T0_GLOBAL))

echo "" | tee -a "$MASTER_LOG"
echo "[$(date '+%H:%M:%S')] === MMseqs2 Hongos COMPLETADO ===" | tee -a "$MASTER_LOG"
echo "  Total procesados: $IDX" | tee -a "$MASTER_LOG"
echo "  OK:               $N_OK" | tee -a "$MASTER_LOG"
echo "  SKIP:             $N_SKIP" | tee -a "$MASTER_LOG"
echo "  FAIL:             $N_FAIL" | tee -a "$MASTER_LOG"
echo "  Tiempo total:     $((DT_GLOBAL / 60)) min $((DT_GLOBAL % 60)) s" | tee -a "$MASTER_LOG"
