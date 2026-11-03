#!/bin/bash
# Piloto 4 vs 8 bloques, consistente con el split actual de animales.
set -e
cd "$HOME/Escritorio/Inpactor3"
. .venv/bin/activate

T8=/tmp/pil2_8; T4=/tmp/pil2_4
rm -rf "$T8" "$T4"; mkdir -p "$T8" "$T4"/{train,val,test}

echo "== 1/4 descargando bundle de 48 kb (18 especies)"
kaggle datasets download -d cristianosoriog/inpactor2-piloto8 -p "$T8/dl" --unzip --quiet
for s in train val test; do
  mkdir -p "$T8/$s"
  cp "$T8"/dl/$s/*.npz "$T8/$s/" 2>/dev/null || true
done
rm -rf "$T8/dl"
echo "   train=$(ls "$T8"/train/*.npz 2>/dev/null|wc -l) val=$(ls "$T8"/val/*.npz 2>/dev/null|wc -l) test=$(ls "$T8"/test/*.npz 2>/dev/null|wc -l)"

echo "== 2/4 enlazando el mismo set en 4 bloques"
B4=datos/bundle4_Animales/Animales
for s in train val test; do
  for f in "$T8/$s"/*.npz; do
    b=$(basename "$f")
    g=$(ls $B4/*/"$b" 2>/dev/null | head -1)
    if [ -n "$g" ]; then ln -sf "$(readlink -f "$g")" "$T4/$s/$b";
    else echo "   SIN PAR 24k: $b"; fi
  done
done
echo "   train=$(ls "$T4"/train/*.npz 2>/dev/null|wc -l) val=$(ls "$T4"/val/*.npz 2>/dev/null|wc -l) test=$(ls "$T4"/test/*.npz 2>/dev/null|wc -l)"

echo "== 3/4 embeddings 48 kb"
python code/precompute_embeddings.py --seq_dir "$T8" --out_dir trabajo/pil2/emb48 \
  --splits train,val,test --batch_size 32 --token_batch 128

echo "== 4/4 embeddings 24 kb y comparación"
python code/precompute_embeddings.py --seq_dir "$T4" --out_dir trabajo/pil2/emb24 \
  --splits train,val,test --batch_size 32 --token_batch 128

python code/comparar_piloto.py --emb48 trabajo/pil2/emb48 --emb24 trabajo/pil2/emb24
