#!/bin/bash
set -u
REINO="${1:?uso: make_bundle.sh <Hongos|Animales|Plantas> <dir_salida>}"
OUT="${2:?falta dir de salida}"
D="/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM"
S="/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt"
B="$OUT/bundle_$REINO"
mkdir -p "$B"

cp -r "$S/training" "$B/training"
rm -rf "$B/training/__pycache__"
mkdir -p "$B/src"
cp "$S/colab_bridge/src/model.py" "$S/colab_bridge/src/dataset.py" "$B/src/"
cp "$S/training/README_TRAINING.md" "$B/"

mkdir -p "$B/consolidated"
cp -r "$D/consolidated/$REINO" "$B/consolidated/"
if [ "$REINO" = "Plantas" ] && [ -d "$D/consolidated_balanced/Plantas" ]; then
  mkdir -p "$B/consolidated_balanced"
  cp -r "$D/consolidated_balanced/Plantas" "$B/consolidated_balanced/"
fi

cat > "$B/MANIFEST.txt" <<EOF
Bundle v2: $REINO
Generado: $(date '+%F %H:%M')

Entrada: 4 bloques x 6 kb (max pooling). Evaluacion: 8 bloques x 6 kb.

Contenido:
  training/        block_layout.py, make_sequence_bundle.py, precompute_embeddings.py,
                   train_mlp.py, evaluate_gold.py, colab_run.py, coverage_report.py
  src/             model.py y dataset.py
  consolidated/    JSONL con label y negative_type
  README_TRAINING.md

Datos pesados (generar en el PC con el disco externo):
  genomes: $D/genomes_449/$REINO
  tamaño:  $(du -sh "$D/genomes_449/$REINO" 2>/dev/null | cut -f1)

Comandos:
  python training/make_sequence_bundle.py --jsonl_dir consolidated/$REINO \\
      --genomes_dir <genomes> --out_dir seq_bundle/$REINO --chunks 4
  python training/make_sequence_bundle.py --jsonl_dir consolidated/$REINO \\
      --genomes_dir <genomes> --out_dir seq_bundle_eval8/$REINO --chunks 8 \\
      --mode eval_full --splits val,test
  python training/colab_run.py --reino $REINO --seq_dir seq_bundle/$REINO \\
      --drive_root <drive> --epochs 100 --balance global --class_weight --calibrate
EOF

tar czf "$OUT/bundle_$REINO.tar.gz" -C "$OUT" "bundle_$REINO"
echo "listo: $OUT/bundle_$REINO.tar.gz"
ls -lh "$OUT/bundle_$REINO.tar.gz"
