#!/bin/bash
set -u
REINO="${1:?uso: make_bundle.sh <Hongos|Animales|Plantas> <dir_salida>}"
OUT="${2:?falta dir de salida}"
D="/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM"
S="/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt"
B="$OUT/bundle_$REINO"
mkdir -p "$B"

cp -r "$S/training" "$B/training"
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
Bundle: $REINO
Generado: $(date '+%F %H:%M')

Contenido:
  training/        scripts (precompute_embeddings.py, train_mlp.py, make_sequence_bundle.py)
  src/             model.py y dataset.py
  consolidated/    JSONL con label y negative_type
  README_TRAINING.md

Datos pesados (copiar del disco externo o generar con make_sequence_bundle.py):
  genomes: $D/genomes_449/$REINO
  tamaño:  $(du -sh "$D/genomes_449/$REINO" 2>/dev/null | cut -f1)

Comandos:
  python training/precompute_embeddings.py --jsonl_dir consolidated/$REINO \\
      --genomes_dir <ruta_genomes> --out_dir emb/$REINO

  python training/train_mlp.py --emb_dir emb/$REINO --out_dir ckpt/$REINO \\
      --epochs 100 --patience 10 --class_weight
EOF

tar czf "$OUT/bundle_$REINO.tar.gz" -C "$OUT" "bundle_$REINO"
echo "listo: $OUT/bundle_$REINO.tar.gz"
ls -lh "$OUT/bundle_$REINO.tar.gz"
