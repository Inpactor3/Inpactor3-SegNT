#!/bin/bash
# Descarga anotaciones TE curadas por la comunidad (para test "oro").
# Usa URLs alternativas si la primera falla.
set -u

PROJECT=/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
OUT=$PROJECT/data/curated_te
mkdir -p "$OUT"
cd "$OUT"

# TAIR10 TE Elements (Arabidopsis) -- URL alternativa de Phytozome
echo "[1] TAIR10 TE (Arabidopsis)..."
# Opcion A: Phytozome mirror
curl -Lk --max-time 60 -o tair10_te.txt \
  "https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_Transposable_Elements.txt" 2>/dev/null
if [ $(stat -c%s tair10_te.txt 2>/dev/null) -lt 1000 ]; then
    rm -f tair10_te.txt
    # Opcion B: GFF completo de TAIR + filtrar
    curl -Lk --max-time 120 -o tair10_full.gff.gz \
      "https://ftp.ensemblgenomes.org/pub/plants/release-58/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.58.gff3.gz" 2>/dev/null
    gunzip -f tair10_full.gff.gz 2>/dev/null
    # Ensembl Plants no incluye TE features para TAIR -- descartar
    rm -f tair10_full.gff
    # Opcion C: GitHub mirror de archivos TAIR
    curl -Lk --max-time 60 -o tair10_te.txt \
      "https://raw.githubusercontent.com/salzberglab/TE/master/TAIR10_Transposable_Elements.txt" 2>/dev/null
fi
ls -lh tair10_te.txt 2>/dev/null | head -1

echo ""
echo "[2] FlyBase Drosophila TE..."
curl -Lk --max-time 180 -o flybase_dmel.gff.gz \
  "https://ftp.flybase.net/releases/current/dmel_r6.55/gff/dmel-all-r6.55.gff.gz" 2>/dev/null
if [ ! -s flybase_dmel.gff.gz ]; then
    curl -Lk --max-time 180 -o flybase_dmel.gff.gz \
      "https://ftp.flybase.net/releases/current/dmel_r6.56/gff/dmel-all-r6.56.gff.gz" 2>/dev/null
fi
gunzip -f flybase_dmel.gff.gz 2>/dev/null
ls -lh flybase_dmel.gff 2>/dev/null | head -1

echo ""
echo "[3] WormBase C. elegans (Ensembl GFF)..."
# Ya lo tenemos en reference_annotations/celegans.gff3 -- hacer link
if [ ! -f wormbase_celegans.gff3 ] && [ -f ../reference_annotations/celegans.gff3 ]; then
    ln -s ../reference_annotations/celegans.gff3 wormbase_celegans.gff3
fi
ls -lh wormbase_celegans.gff3 2>/dev/null | head -1

echo ""
echo "[4] SGD S. cerevisiae TE Ty..."
# Ya tenemos scerevisiae.gff3 con 91 transposable_element features
if [ ! -f sgd_scerevisiae.gff3 ] && [ -f ../reference_annotations/scerevisiae.gff3 ]; then
    ln -s ../reference_annotations/scerevisiae.gff3 sgd_scerevisiae.gff3
fi
ls -lh sgd_scerevisiae.gff3 2>/dev/null | head -1

echo ""
echo "[5] Oryza MSU7 RiceTE..."
curl -Lk --max-time 180 -o rice_msu7_te.gff3 \
  "http://rice.uga.edu/annotation_dl.shtml" 2>/dev/null
if [ $(stat -c%s rice_msu7_te.gff3 2>/dev/null || echo 0) -lt 10000 ]; then
    rm -f rice_msu7_te.gff3
    # Alternativa: GitHub con anotacion de rice
    curl -Lk --max-time 180 -o rice_msu7_te.gff3 \
      "https://raw.githubusercontent.com/oushujun/riceTE/main/rice.MSU7.te.gff3" 2>/dev/null
fi
ls -lh rice_msu7_te.gff3 2>/dev/null | head -1

echo ""
echo "[6] Dfam curated (version seed, no la gigante)..."
curl -Lk --max-time 300 -o dfam_curated.embl.gz \
  "https://www.dfam.org/releases/current/families/FamDB/dfam3.6.hmm.gz" 2>/dev/null
if [ $(stat -c%s dfam_curated.embl.gz 2>/dev/null || echo 0) -lt 100000 ]; then
    rm -f dfam_curated.embl.gz
fi
ls -lh dfam_curated.embl.gz 2>/dev/null | head -1

echo ""
echo "=== DESCARGAS TERMINADAS ==="
ls -lh "$OUT"
