#!/bin/bash
# Plantas LOCAL: lib completa panteon_plantae (aprovecha swap de 32GB).
# Persistente en scripts/ para sobrevivir reinicios. SKIP de BED ya existentes.
set -u

S=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
MMSEQS=$HOME/bin/mmseqs
REF_LIB=$S/data/panteon_libraries/panteon_plantae.fasta
MANIFEST=$S/docs/species_download_manifest.tsv
WORK=/home/cami/Desktop/ProyectoIntegrador/plantas_local_work
RESULTS=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/annotations/Plantas

mkdir -p "$WORK"/{genomes,tmp,logs} "$RESULTS"
MASTER_LOG="$WORK/logs/plantas_local.log"
echo "[$(date '+%H:%M:%S')] Plantas LOCAL (lib completa + swap) - inicio" | tee -a "$MASTER_LOG"

# Descarga robusta con verificacion de tamano
cat > "$WORK/download_robusto.sh" <<'DL'
#!/bin/bash
URL="$1"; DEST="$2"; RETRIES="${3:-3}"
for i in $(seq 1 "$RETRIES"); do
    expected=$(curl -sILk --max-time 30 "$URL" | awk 'tolower($1)=="content-length:"{gsub("\r",""); print $2; exit}')
    [ -z "$expected" ] && expected="unknown"
    curl -sLk --retry 3 --retry-delay 10 -C - -o "$DEST" "$URL"
    if [ "$expected" != "unknown" ]; then
        actual=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
        if [ "$actual" -eq "$expected" ]; then echo "OK $actual bytes (completo)"; exit 0; fi
        echo "INCOMPLETO: $actual / $expected (intento $i/$RETRIES)"; sleep 5
    else
        if [ -s "$DEST" ] && gzip -t "$DEST" 2>/dev/null; then echo "OK (gzip valido)"; exit 0; fi
        echo "INCOMPLETO (intento $i/$RETRIES)"; sleep 5
    fi
done
echo "FAIL_TRAS_RETRIES"; exit 1
DL
chmod +x "$WORK/download_robusto.sh"

REF_DB="$WORK/panteon_plantae_db"
if [ ! -f "${REF_DB}.index" ]; then
    echo "[DB] createdb panteon_plantae (603 MB)..." | tee -a "$MASTER_LOG"
    "$MMSEQS" createdb "$REF_LIB" "$REF_DB" >> "$MASTER_LOG" 2>&1
fi

awk -F'\t' '$2=="Plantas" && $4=="found" {print $8"\t"$1"\t"$3"\t"$5"\t"$9}' "$MANIFEST" | sort -g > "$WORK/queue.txt"
TOTAL=$(wc -l < "$WORK/queue.txt")
echo "[queue] $TOTAL plantas" | tee -a "$MASTER_LOG"

N_OK=0; N_SKIP=0; N_FAIL=0; IDX=0
while IFS=$'\t' read -r size_mb split especie accession url; do
    IDX=$((IDX+1))
    sp_safe=$(echo "$especie" | tr ' ' '_' | tr -cd 'A-Za-z0-9._-')
    tag="${sp_safe}__${accession}"
    OUT_BED="$RESULTS/$split/${tag}.bed"
    mkdir -p "$RESULTS/$split"
    if [ -f "$OUT_BED" ] && [ -s "$OUT_BED" ]; then
        echo "[$IDX/$TOTAL] SKIP $tag" >> "$MASTER_LOG"; N_SKIP=$((N_SKIP+1)); continue
    fi
    T0=$(date +%s)
    echo "[$IDX/$TOTAL] RUN $tag (${size_mb} MB)" | tee -a "$MASTER_LOG"
    GZ="$WORK/genomes/${tag}.fna.gz"
    bash "$WORK/download_robusto.sh" "$url" "$GZ" 3 >> "$MASTER_LOG" 2>&1
    if [ ! -s "$GZ" ]; then echo "   FAIL (download)" | tee -a "$MASTER_LOG"; N_FAIL=$((N_FAIL+1)); rm -f "$GZ"; continue; fi
    FASTA="$WORK/tmp/${tag}.fasta"
    gunzip -c "$GZ" > "$FASTA" && rm "$GZ"
    Q_DB="$WORK/tmp/q_${tag}"
    "$MMSEQS" createdb "$FASTA" "$Q_DB" > /dev/null 2>&1
    rm -f "$FASTA"
    RES_DB="$WORK/tmp/r_${tag}"; TMP_D="$WORK/tmp/t_${tag}"
    "$MMSEQS" search "$Q_DB" "$REF_DB" "$RES_DB" "$TMP_D" --search-type 3 --threads 3 -s 5.7 > /dev/null 2>&1
    RES_TAB="$WORK/tmp/r_${tag}.m8"
    [ -s "${RES_DB}.index" ] && "$MMSEQS" convertalis "$Q_DB" "$REF_DB" "$RES_DB" "$RES_TAB" --format-output 'query,target,pident,alnlen,qstart,qend,evalue' > /dev/null 2>&1
    if [ -s "$RES_TAB" ]; then
        awk -F'\t' 'BEGIN{OFS="\t"}{chrom=$1;target=$2;pident=$3;qs=$5;qe=$6;start=(qs<qe?qs:qe)-1;end=(qs>qe?qs:qe);n=split(target,p,"#");te=p[1];cls=(n>=2?p[2]:"Unknown");print chrom,start,end,te"|"cls,pident,"."}' "$RES_TAB" > "$OUT_BED"
        echo "   OK $(( $(date +%s)-T0 ))s $(wc -l < "$OUT_BED") hits" | tee -a "$MASTER_LOG"; N_OK=$((N_OK+1))
    else
        echo "   FAIL (sin hits)" | tee -a "$MASTER_LOG"; N_FAIL=$((N_FAIL+1))
    fi
    rm -rf "$Q_DB"* "$RES_DB"* "$TMP_D" "$RES_TAB"
done < "$WORK/queue.txt"

echo "[$(date '+%H:%M:%S')] === PLANTAS LOCAL OK=$N_OK SKIP=$N_SKIP FAIL=$N_FAIL ===" | tee -a "$MASTER_LOG"
