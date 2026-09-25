#!/usr/bin/env bash
# Pipeline completo Inpactor3-SegNT.
# Requiere: models/best.pt no existir (fase entrenamiento) o existir (fase eval).

set -e
cd "$(dirname "$0")/.."

echo "==> 1. Descargando Arabidopsis TAIR10"
bash scripts/download_arabidopsis.sh

echo "==> 2. Verificando anotaciones ground truth"
if [ ! -f data/raw/Inpactor2_predictions.tab ]; then
    echo "[WARN] data/raw/Inpactor2_predictions.tab no existe."
    echo "        Cópialo del proyecto anterior o corre Inpactor2 sobre TAIR10."
    exit 1
fi
python scripts/build_labels.py \
    --genome data/raw/TAIR10.fasta \
    --annotations data/raw/Inpactor2_predictions.tab

echo "==> 3. Entrenamiento (SegNT sobre NT-50M)"
python -m inpactor3_segnt.train --config configs/nt_50m.yaml

echo "==> 4. Inferencia sobre cromosoma 5"
python -m inpactor3_segnt.predict \
    --checkpoint models/best.pt \
    --genome data/raw/TAIR10.fasta \
    --scaffold 5 \
    --out results/chr5_predictions.tab

echo "==> 5. Evaluación contra Inpactor2"
python -m inpactor3_segnt.evaluate \
    --pred results/chr5_predictions.tab \
    --truth data/raw/Inpactor2_predictions.tab \
    --scaffold 5 \
    --genome data/raw/TAIR10.fasta \
    --report results/chr5_report.md

echo "==> LISTO. Ver results/chr5_report.md"
