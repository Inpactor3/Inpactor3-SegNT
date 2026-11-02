#!/bin/bash
# Piloto: ¿ayuda entrenar con 8 bloques (48 kb) en vez de 4 (24 kb)?
# Mismas especies de train y mismas de evaluación; solo cambia el nº de bloques.
set -e
cd "$HOME/Escritorio/Inpactor3"
. .venv/bin/activate

T=/tmp/pilot8; rm -rf "$T"; mkdir -p "$T"/{train,val,test}

echo "== 1/5 descargando bundle de 48 kb (12 especies)"
kaggle datasets download -d cristianosoriog/inpactor2-piloto8 -p "$T/dl" --unzip --quiet
cp "$T"/dl/*/*.npz "$T"/train/ 2>/dev/null || cp "$T"/dl/*.npz "$T"/train/
echo "   train: $(ls "$T"/train/*.npz | wc -l) especies"

echo "== 2/5 enlazando val/test de 48 kb (3 + 3 especies)"
B8=datos/bundle8_Animales/Animales
for sp in Acropora_acuminata Pelagia_noctiluca Aricia_agestis; do
  f=$(ls $B8/val/*"$sp"*.npz 2>/dev/null | head -1) && ln -sf "$(readlink -f "$f")" "$T/val/"
done
for sp in Nasonia_giraulti Xenos_peckii Branchiostoma_lanceolatum; do
  f=$(ls $B8/test/*"$sp"*.npz 2>/dev/null | head -1) && ln -sf "$(readlink -f "$f")" "$T/test/"
done
echo "   val: $(ls "$T"/val/*.npz 2>/dev/null | wc -l) · test: $(ls "$T"/test/*.npz 2>/dev/null | wc -l)"

echo "== 3/5 embeddings de 48 kb"
python code/precompute_embeddings.py --seq_dir "$T" --out_dir trabajo/piloto48/emb \
  --splits train,val,test --batch_size 32 --token_batch 128

echo "== 4/5 enlazando el mismo set en 4 bloques"
T4=/tmp/pilot4; rm -rf "$T4"; mkdir -p "$T4"/{train,val,test}
B4=datos/bundle4_Animales/Animales
for f in "$T"/train/*.npz; do
  b=$(basename "$f")
  g=$(ls $B4/train/*"${b%%.npz}"* 2>/dev/null | head -1)
  [ -n "$g" ] && ln -sf "$(readlink -f "$g")" "$T4/train/$b"
done
for f in "$T"/val/*.npz "$T"/test/*.npz; do
  b=$(basename "$f"); sp=${b%.npz}
  for split in val test; do
    g=$(ls $B4/$split/*"$sp"* 2>/dev/null | head -1)
    [ -n "$g" ] && ln -sf "$(readlink -f "$g")" "$T4/$split/$b"
  done
done
echo "   train: $(ls "$T4"/train/*.npz 2>/dev/null | wc -l) · val: $(ls "$T4"/val/*.npz 2>/dev/null | wc -l) · test: $(ls "$T4"/test/*.npz 2>/dev/null | wc -l)"
python code/precompute_embeddings.py --seq_dir "$T4" --out_dir trabajo/piloto24/emb \
  --splits train,val,test --batch_size 32 --token_batch 128

echo "== 5/5 entrenando ambos y comparando sobre los embeddings de 48 kb"
python code/entrenar_piloto.py
