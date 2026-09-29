#!/usr/bin/env bash
# Descarga los 3 genomas de fondo para PanTEon (uno por reino).
# Total: ~5-6 GB comprimidos, ~15-20 GB descomprimidos.

set -e
cd "$(dirname "$0")/.."
mkdir -p data/common_dataset/genomes
cd data/common_dataset/genomes

# ---- Plantae: Zea mays (~2.5 GB descomprimido) ----
echo "=== Descargando Zea mays (maíz) ==="
if [ ! -f Zea_mays.fasta ]; then
    curl -Lk -o Zea_mays.fa.gz \
        "https://ftp.ensemblgenomes.org/pub/plants/release-58/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
    gunzip Zea_mays.fa.gz
    mv Zea_mays.fa Zea_mays.fasta
fi
ls -lh Zea_mays.fasta

# ---- Animalia: Homo sapiens (~3 GB descomprimido) ----
echo "=== Descargando Homo sapiens ==="
if [ ! -f Homo_sapiens.fasta ]; then
    # Solo chr21 y chr22 para no bajar 3 GB (suficiente diversidad de ERV)
    for CHR in 21 22; do
        curl -Lk -o "chr${CHR}.fa.gz" \
            "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.chromosome.${CHR}.fa.gz"
        gunzip "chr${CHR}.fa.gz"
    done
    cat chr21.fa chr22.fa > Homo_sapiens.fasta
    rm chr21.fa chr22.fa
fi
ls -lh Homo_sapiens.fasta

# ---- Fungi: Neurospora crassa (~40 MB) ----
echo "=== Descargando Neurospora crassa ==="
if [ ! -f Neurospora_crassa.fasta ]; then
    curl -Lk -o Neurospora_crassa.fa.gz \
        "https://ftp.ensemblgenomes.org/pub/fungi/release-58/fasta/neurospora_crassa/dna/Neurospora_crassa.NC12.dna_sm.toplevel.fa.gz"
    gunzip Neurospora_crassa.fa.gz
    mv Neurospora_crassa.fa Neurospora_crassa.fasta
fi
ls -lh Neurospora_crassa.fasta

# ---- Limpiar: uppercase + solo ACGTN ----
echo "=== Limpiando FASTAs (uppercase + ACGTN only) ==="
for F in Zea_mays.fasta Homo_sapiens.fasta Neurospora_crassa.fasta; do
    python3 -c "
import sys
with open('$F') as fi, open('${F}.tmp', 'w') as fo:
    for line in fi:
        if line.startswith('>'):
            fo.write(line)
        else:
            s = line.strip().upper()
            s = ''.join(c if c in 'ACGTN' else 'N' for c in s)
            fo.write(s + '\n')
"
    mv "${F}.tmp" "$F"
done

echo "=== LISTO ==="
ls -lh
