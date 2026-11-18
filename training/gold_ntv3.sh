#!/usr/bin/env bash
# Evalua el cabezal NTv3 de plantas sobre el test oro de Arabidopsis.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="${1:?pasa la carpeta gold_ds}"
HEAD="${2:-$HOME/Escritorio/Inpactor3/trabajo/ntv3_plantas/modelo/best.pt}"
OUT="${OUT:-$PWD/trabajo/gold_ntv3}"
G="$DS/gold"
primero() { for p in "$@"; do [ -f "$p" ] && { echo "$p"; return; }; done; }
TAIR=$(primero "$G/TAIR10.fasta/TAIR10.fasta" "$G/TAIR10.fasta" "$G/TAIR10.fasta.gz")
BED=$(primero "$G/tair10_te.bed")
[ -n "$TAIR" ] && [ -n "$BED" ] || { echo "ERROR: no encuentro TAIR10 o el BED en $G"; exit 1; }
[ -f "$HEAD" ] || { echo "ERROR: no existe el cabezal $HEAD"; exit 1; }
echo "genoma: $TAIR"
echo "bed:    $BED"
echo "cabezal:$HEAD"
python "$HERE/gold_ntv3.py" --genome "$TAIR" --bed "$BED" --head "$HEAD" \
  --out "$OUT" --chunks 4 --batch 2
echo
echo "===== NTv3 EN ORO (Arabidopsis) ====="
echo "Referencia v2 (24 kb, MIL): BA 0,7031 · F1 0,5953 · PR-AUC 0,8243"
