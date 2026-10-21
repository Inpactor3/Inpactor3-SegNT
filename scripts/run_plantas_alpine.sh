#!/bin/bash
# Plantas en ALPINE (192.168.1.22): binario SSE2, chunks de 10k, rango 500-1500 MB.
set -u

H=/home/cami/InpactorRM_plantas
MMSEQS=$H/mmseqs_sse2/bin/mmseqs
REF_LIB=$H/libs/panteon_plantae.fasta
MANIFEST=$H/manifest.tsv
DONE=$H/done_plantas.txt
WORK=$H/work2
RESULTS=$H/results
CHUNKS=$H/libs/chunks_small

mkdir -p "$WORK"/{genomes,tmp,logs} "$RESULTS"
MASTER_LOG="$WORK/logs/plantas.log"
echo "[$(date '+%H:%M:%S')] ALPINE Plantas 500-1500MB (sse2) - inicio" | tee -a "$MASTER_LOG"

cat > "$WORK/dl.sh" <<'DL'
#!/bin/bash
URL="$1"; DEST="$2"; RETRIES="${3:-3}"
for i in $(seq 1 "$RETRIES"); do
    expected=$(curl -sILk --max-time 30 "$URL" | awk 'tolower($1)=="content-length:"{gsub("\r",""); print $2; exit}')
    [ -z "$expected" ] && expected="unknown"
    curl -sLk --retry 3 --retry-delay 10 -C - -o "$DEST" "$URL"
    if [ "$expected" != "unknown" ]; then
        actual=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
        [ "$actual" -eq "$expected" ] && { echo "OK $actual"; exit 0; }
        echo "INCOMPLETO $actual/$expected"; sleep 5
    else
        [ -s "$DEST" ] && gzip -t "$DEST" 2>/dev/null && { echo "OK gzip"; exit 0; }
        sleep 5
    fi
done
echo "FAIL"; exit 1
DL
chmod +x "$WORK/dl.sh"

# createdb de los chunks si faltan
if ! ls "$CHUNKS"/chunk_1_db.index >/dev/null 2>&1; then
    for f in "$CHUNKS"/chunk_*.fasta; do
        b=$(basename "$f" .fasta)
        "$MMSEQS" createdb "$f" "$CHUNKS/${b}_db" > /dev/null 2>&1
    done
fi

awk -F'\t' '$2=="Plantas" && $4=="found" && $8+0 >= 500 && $8+0 <= 1500 {print $8"\t"$1"\t"$3"\t"$5"\t"$9}' "$MANIFEST" | sort -g > "$WORK/queue.txt"
TOTAL=$(wc -l < "$WORK/queue.txt")
echo "[queue] $TOTAL plantas (500-1500MB)" | tee -a "$MASTER_LOG"

N_OK=0; N_SKIP=0; N_FAIL=0; IDX=0
while IFS=$'\t' read -r size_mb split especie accession url; do
    IDX=$((IDX+1))
    sp_safe=$(echo "$especie" | tr ' ' '_' | tr -cd 'A-Za-z0-9._-')
    tag="${sp_safe}__${accession}"
    OUT_BED="$RESULTS/$split/${tag}.bed"
    mkdir -p "$RESULTS/$split"
    if grep -qxF "$tag" "$DONE" 2>/dev/null || [ -s "$OUT_BED" ]; then
        echo "[$IDX/$TOTAL] SKIP $tag" >> "$MASTER_LOG"; N_SKIP=$((N_SKIP+1)); continue
    fi
    T0=$(date +%s)
    echo "[$IDX/$TOTAL] RUN $tag (${size_mb} MB)" | tee -a "$MASTER_LOG"
    GZ="$WORK/genomes/${tag}.fna.gz"
    bash "$WORK/dl.sh" "$url" "$GZ" 3 >> "$MASTER_LOG" 2>&1
    if [ ! -s "$GZ" ]; then echo "   FAIL download" | tee -a "$MASTER_LOG"; N_FAIL=$((N_FAIL+1)); rm -f "$GZ"; continue; fi
    FASTA="$WORK/tmp/${tag}.fasta"
    gunzip -c "$GZ" > "$FASTA" && rm "$GZ"
    Q_DB="$WORK/tmp/q_${tag}"
    "$MMSEQS" createdb "$FASTA" "$Q_DB" > /dev/null 2>&1
    rm -f "$FASTA"
    COMB="$WORK/tmp/comb_${tag}.m8"; : > "$COMB"
    for ci in "$CHUNKS"/chunk_*_db; do
        [ -f "${ci}.index" ] || continue
        RES="$WORK/tmp/r_${tag}_$(basename "$ci")"
        TMPD="$WORK/tmp/t_${tag}_$(basename "$ci")"
        "$MMSEQS" search "$Q_DB" "$ci" "$RES" "$TMPD" --search-type 3 --threads 4 -s 5.7 > /dev/null 2>&1
        if [ -s "${RES}.index" ]; then
            "$MMSEQS" convertalis "$Q_DB" "$ci" "$RES" "$WORK/tmp/c.m8" --format-output 'query,target,pident,alnlen,qstart,qend,evalue' > /dev/null 2>&1
            cat "$WORK/tmp/c.m8" >> "$COMB" 2>/dev/null
        fi
        rm -rf "$RES"* "$TMPD" "$WORK/tmp/c.m8"
    done
    if [ -s "$COMB" ]; then
        sort -k1,1 -k5,5n -k6,6n "$COMB" | uniq > "$WORK/tmp/s.m8"
        awk -F'\t' 'BEGIN{OFS="\t"}{c=$1;t=$2;p=$3;qs=$5;qe=$6;s=(qs<qe?qs:qe)-1;e=(qs>qe?qs:qe);n=split(t,x,"#");print c,s,e,x[1]"|"(n>=2?x[2]:"Unknown"),p,"."}' "$WORK/tmp/s.m8" > "$OUT_BED"
        echo "   OK $(( $(date +%s)-T0 ))s $(wc -l < "$OUT_BED") hits" | tee -a "$MASTER_LOG"; N_OK=$((N_OK+1))
    else
        echo "   FAIL sin_hits" | tee -a "$MASTER_LOG"; N_FAIL=$((N_FAIL+1))
    fi
    rm -rf "$Q_DB"* "$COMB" "$WORK/tmp/s.m8"
done < "$WORK/queue.txt"

echo "[$(date '+%H:%M:%S')] === ALPINE PLANTAS OK=$N_OK SKIP=$N_SKIP FAIL=$N_FAIL ===" | tee -a "$MASTER_LOG"
