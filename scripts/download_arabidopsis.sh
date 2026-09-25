#!/usr/bin/env bash
# Baja Arabidopsis TAIR10 (los 5 cromosomas) desde Ensembl Plants.
# Deja el FASTA en data/raw/TAIR10.fasta

set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw

BASE="https://ftp.ensemblgenomes.org/pub/plants/release-58/fasta/arabidopsis_thaliana/dna"

echo "[dl] Arabidopsis TAIR10 - 5 cromosomas"
for CHR in 1 2 3 4 5; do
    NAME="Arabidopsis_thaliana.TAIR10.dna_sm.chromosome.${CHR}.fa.gz"
    if [ ! -f "data/raw/$NAME" ]; then
        wget -q --show-progress "$BASE/$NAME" -O "data/raw/$NAME"
    else
        echo "  [skip] $NAME ya existe"
    fi
done

echo "[unpack] descomprimiendo y concatenando"
cd data/raw
for CHR in 1 2 3 4 5; do
    gunzip -kf "Arabidopsis_thaliana.TAIR10.dna_sm.chromosome.${CHR}.fa.gz"
done

# Concatenar en un solo FASTA con headers simplificados a "1", "2", "3", "4", "5"
> TAIR10.fasta
for CHR in 1 2 3 4 5; do
    echo ">${CHR}" >> TAIR10.fasta
    grep -v '^>' "Arabidopsis_thaliana.TAIR10.dna_sm.chromosome.${CHR}.fa" >> TAIR10.fasta
done

echo "[ok] data/raw/TAIR10.fasta creado"
grep '^>' TAIR10.fasta
echo "Bases totales:"
grep -v '^>' TAIR10.fasta | tr -d '\n' | wc -c
