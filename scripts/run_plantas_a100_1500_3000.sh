#!/bin/bash
set -u
W=/content/plantas
MM=$W/mmseqs/bin/mmseqs
REF=$W/libs/panteon_plantae.fasta
REFDB=$W/work/refdb
LOG=$W/work/logs/a100.log
mkdir -p "$W"/{work/genomes,work/tmp,work/logs,results}

echo "[$(date '+%H:%M:%S')] A100 plantas 1.5-3GB inicio" >> "$LOG"
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
1533.7	train	Musa_acuminata__GCF_036884655.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/036/884/655/GCF_036884655.1_Cavendish_Baxijiao_AAA/GCF_036884655.1_Cavendish_Baxijiao_AAA_genomic.fna.gz
1629.8	train	Chenopodium_formosanum__GCA_024500155.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/024/500/155/GCA_024500155.1_ASM2450015v1/GCA_024500155.1_ASM2450015v1_genomic.fna.gz
1694.9	train	Rhynchospora_pubera__GCA_028095005.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/028/095/005/GCA_028095005.1_MPIPZ_rhyPub2m_2.0/GCA_028095005.1_MPIPZ_rhyPub2m_2.0_genomic.fna.gz
1735.0	train	Agrostemma_githago__GCA_958450325.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/958/450/325/GCA_958450325.1_dcAgrGith1.1/GCA_958450325.1_dcAgrGith1.1_genomic.fna.gz
1745.0	train	Diphasiastrum_complanatum__GCA_029204225.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/029/204/225/GCA_029204225.1_Diphasiastrum_complanatum_v3.1/GCA_029204225.1_Diphasiastrum_complanatum_v3.1_genomic.fna.gz
1779.8	train	Poa_annua__GCA_026929855.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/026/929/855/GCA_026929855.1_FRR_PoAnn_1.0/GCA_026929855.1_FRR_PoAnn_1.0_genomic.fna.gz
1781.3	train	Dactylis_glomerata__GCA_007115705.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/007/115/705/GCA_007115705.1_ASM711570v1/GCA_007115705.1_ASM711570v1_genomic.fna.gz
1826.8	train	Hibiscus_syriacus__GCF_006381635.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/006/381/635/GCF_006381635.1_ASM638163v2/GCF_006381635.1_ASM638163v2_genomic.fna.gz
2028.9	train	Panax_japonicus__GCA_020205505.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/020/205/505/GCA_020205505.1_ASM2020550v1/GCA_020205505.1_ASM2020550v1_genomic.fna.gz
2168.7	train	Chloranthus_sessilifolius__GCA_021018995.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/021/018/995/GCA_021018995.1_ASM2101899v1/GCA_021018995.1_ASM2101899v1_genomic.fna.gz
2297.0	train	Gossypium_mustelinum__GCA_007990455.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/007/990/455/GCA_007990455.1_Gossypium_mustelinum_v1.1/GCA_007990455.1_Gossypium_mustelinum_v1.1_genomic.fna.gz
2378.0	train	Zea_mays__GCA_049266015.1	https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/049/266/015/GCA_049266015.1_ASM4926601v1/GCA_049266015.1_ASM4926601v1_genomic.fna.gz
EOF

TOTAL=$(wc -l < "$W/work/lista.tsv"); IDX=0
while IFS=$'\t' read -r mb split tag url; do
  IDX=$((IDX+1))
  OUT="$W/results/$split/${tag}.bed"; mkdir -p "$W/results/$split"
  [ -s "$OUT" ] && { echo "[$IDX/$TOTAL] SKIP $tag" >> "$LOG"; continue; }
  T0=$(date +%s); echo "[$IDX/$TOTAL] RUN $tag (${mb} MB)" >> "$LOG"
  GZ="$W/work/genomes/$tag.fna.gz"
  bash "$W/work/dl.sh" "$url" "$GZ" >> "$LOG" 2>&1 || { echo "  FAIL download" >> "$LOG"; rm -f "$GZ"; continue; }
  FA="$W/work/tmp/$tag.fasta"; gunzip -c "$GZ" > "$FA" && rm "$GZ"
  Q="$W/work/tmp/q_$tag"; "$MM" createdb "$FA" "$Q" >> "$LOG" 2>&1; rm -f "$FA"
  R="$W/work/tmp/r_$tag"; T="$W/work/tmp/t_$tag"
  "$MM" search "$Q" "$REFDB" "$R" "$T" --search-type 3 --threads 12 -s 5.7 >> "$LOG" 2>&1
  M8="$W/work/tmp/$tag.m8"
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
