#!/bin/bash
# Deduplica BED: agrupa hits solapados por cromosoma/locus y conserva el de mayor pident.
# Reduce ~33x -> 1x y arregla coverage_fraction > 1.
# Uso: dedup_bed.sh <archivo.bed> <salida.bed>
set -u
IN="$1"; OUT="$2"
TMP=$(mktemp -d /tmp/kilo/dedup_XXXX)
trap 'rm -rf "$TMP"' EXIT

# ordenar por cromosoma y start, luego fusionar solapados
LC_ALL=C sort -S 1G -T "$TMP" -k1,1 -k2,2n "$IN" | awk -F'\t' 'BEGIN{OFS="\t"}
{
  s=$2+0; e=$3+0; p=($5==""?0:$5+0); name=$4;
  if (NR==1) { ck=$1; cs=s; ce=e; cname=name; cp=p; next }
  if ($1!=ck || s>ce) {
     print ck, cs, ce, cname, cp, ".";
     ck=$1; cs=s; ce=e; cname=name; cp=p;
  } else {
     if (e>ce) ce=e;
     if (p>cp) { cp=p; cname=name }
  }
}
END { if (ck!="") print ck, cs, ce, cname, cp, "." }' > "$OUT"
