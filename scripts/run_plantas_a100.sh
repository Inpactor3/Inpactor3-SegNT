#!/bin/bash
# A100: plantas >3GB con lib completa (83GB RAM, 12 hilos).
set -u
W=/content/plantas
MM=$W/mmseqs/bin/mmseqs
REF=$W/libs/panteon_plantae.fasta
REFDB=$W/work/refdb
LOG=$W/work/logs/a100.log
mkdir -p "$W"/{work/genomes,work/tmp,work/logs,results}

echo "[$(date '+%H:%M:%S')] A100 plantas >3GB inicio" >> "$LOG"
[ -f "$REFDB.index" ] || "$MM" createdb "$REF" "$REFDB" >> "$LOG" 2>&1

cat > "$W/work/dl.sh" <<'DL'
#!/bin/bash
URL="$1"; DEST="$2"
exp=$(curl -sILk --max-time 30 "$URL" | awk 'tolower($1)=="content-length:"{gsub("\r","");print $2;exit}')
curl -sLk --retry 3 --retry-delay 10 -C - -o "$DEST" "$URL"
if [ -n "${exp:-}" ]; then
  act=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  [ "$act" = "$exp" ] || { echo "INCOMPLETO $act/$exp"; exit 1; }
fi
echo OK
DL
chmod +x "$W/work/dl.sh"

cat > "$W/work/lista.tsv" <<'EOF'
3310.1	train	Capsicum_baccatum	GCA_030864225.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/030/864/225/GCA_030864225.1_ASM3086422v1/GCA_030864225.1_ASM3086422v1_genomic.fna.gz
3332.6	test	Apium_graveolens	GCF_009905375.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/905/375/GCF_009905375.1_ASM990537v1/GCF_009905375.1_ASM990537v1_genomic.fna.gz
3571.7	train	Alopecurus_myosuroides	GCA_028641055.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/028/641/055/GCA_028641055.1_Alomy_genome_v1/GCA_028641055.1_Alomy_genome_v1_genomic.fna.gz
3995.6	val	Nicotiana_tabacum	GCF_000715075.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/715/075/GCF_000715075.1_ASM71507v2/GCF_000715075.1_ASM71507v2_genomic.fna.gz
5902.7	val	Aegilops_bicornis	GCA_021605145.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/021/605/145/GCA_021605145.1_ASM2160514v1/GCA_021605145.1_ASM2160514v1_genomic.fna.gz
6702.6	train	Secale_cereale	GCA_965641915.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/965/641/915/GCA_965641915.1_lpSecCere.Lo7.IPK.v3/GCA_965641915.1_lpSecCere.Lo7.IPK.v3_genomic.fna.gz
6801.1	train	Glebionis_coronaria	GCA_025667525.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/025/667/525/GCA_025667525.1_Gcor_v1/GCA_025667525.1_Gcor_v1_genomic.fna.gz
10238.0	train	Taxus_chinensis	GCA_019776745.2	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/019/776/745/GCA_019776745.2_Ta-2021/GCA_019776745.2_Ta-2021_genomic.fna.gz
10353.3	train	Triticum_turgidum	GCA_057814425.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/057/814/425/GCA_057814425.1_Kronos_v1.1/GCA_057814425.1_Kronos_v1.1_genomic.fna.gz
10482.7	val	Cycas_panzhihuaensis	GCA_023213395.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/023/213/395/GCA_023213395.1_ASM2321339v1/GCA_023213395.1_ASM2321339v1_genomic.fna.gz
10677.1	train	Triticum_dicoccoides	GCF_002162155.2	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/002/162/155/GCF_002162155.2_WEW_v2.1/GCF_002162155.2_WEW_v2.1_genomic.fna.gz
11813.7	test	Sambucus_nigra	GCA_949130495.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/949/130/495/GCA_949130495.1_daSamNigr1.1/GCA_949130495.1_daSamNigr1.1_genomic.fna.gz
14566.5	train	Triticum_aestivum	GCF_018294505.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1_genomic.fna.gz
26572.0	train	Picea_glauca	GCA_000966675.3	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/966/675/GCA_000966675.3_WS77111_v2/GCA_000966675.3_WS77111_v2_genomic.fna.gz
EOF

TOTAL=$(wc -l < "$W/work/lista.tsv")
IDX=0
while IFS=$'\t' read -r mb split tag acc url; do
  IDX=$((IDX+1))
  mytag="${tag}__${acc}"
  OUT="$W/results/$split/${mytag}.bed"; mkdir -p "$W/results/$split"
  [ -s "$OUT" ] && { echo "[$IDX/$TOTAL] SKIP $mytag" >> "$LOG"; continue; }
  T0=$(date +%s); echo "[$IDX/$TOTAL] RUN $mytag (${mb} MB)" >> "$LOG"
  GZ="$W/work/genomes/$mytag.fna.gz"
  bash "$W/work/dl.sh" "$url" "$GZ" >> "$LOG" 2>&1 || { echo "  FAIL download" >> "$LOG"; rm -f "$GZ"; continue; }
  FA="$W/work/tmp/$mytag.fasta"; gunzip -c "$GZ" > "$FA" && rm "$GZ"
  Q="$W/work/tmp/q_$mytag"; "$MM" createdb "$FA" "$Q" >> "$LOG" 2>&1; rm -f "$FA"
  R="$W/work/tmp/r_$mytag"; T="$W/work/tmp/t_$mytag"
  "$MM" search "$Q" "$REFDB" "$R" "$T" --search-type 3 --threads 12 -s 5.7 >> "$LOG" 2>&1
  M8="$W/work/tmp/$mytag.m8"
  [ -s "$R.index" ] && "$MM" convertalis "$Q" "$REFDB" "$R" "$M8" --format-output 'query,target,pident,alnlen,qstart,qend,evalue' >> "$LOG" 2>&1
  if [ -s "$M8" ]; then
    awk -F'\t' 'BEGIN{OFS="\t"}{c=$1;t=$2;p=$3;qs=$5;qe=$6;s=(qs<qe?qs:qe)-1;e=(qs>qe?qs:qe);n=split(t,x,"#");print c,s,e,x[1]"|"(n>=2?x[2]:"Unknown"),p,"."}' "$M8" > "$OUT"
    echo "  OK $(( $(date +%s)-T0 ))s $(wc -l < "$OUT") hits" >> "$LOG"
  else
    echo "  FAIL sin_hits" >> "$LOG"
  fi
  rm -rf "$Q"* "$R"* "$T" "$M8"
done < "$W/work/lista.tsv"
echo "[$(date '+%H:%M:%S')] A100 FIN" >> "$LOG"
