#!/bin/bash
# Integra las 10 especies nuevas de animales (descargadas de Kaggle) y reentrena.
set -e
cd "$HOME/Escritorio/Inpactor3"
. .venv/bin/activate

D=cristianosoriog/inpactor2-animales-nuevas
T=/tmp/nuevas
rm -rf "$T"; mkdir -p "$T"

echo "== 1/4 listando archivos del dataset"
kaggle datasets files -d "$D" --page-size 50 --format json > /tmp/nuevas.json
python - <<'PY' > /tmp/nuevas.txt
import json
for i in json.load(open('/tmp/nuevas.json')):
    if i['name'].endswith('.npz'):
        print(i['name'])
PY
wc -l < /tmp/nuevas.txt

echo "== 2/4 descargando npz"
while read -r f; do
  echo "   $f"
  kaggle datasets download -d "$D" -f "$f" -p "$T" --quiet
done < /tmp/nuevas.txt

mkdir -p datos/bundle4_Animales/Animales/train
cp "$T"/Animales/train/*.npz datos/bundle4_Animales/Animales/train/ 2>/dev/null || \
cp "$T"/*/*/train/*.npz datos/bundle4_Animales/Animales/train/ 2>/dev/null || true
echo "   especies en train: $(ls datos/bundle4_Animales/Animales/train/*.npz | wc -l)"
rm -rf "$T"

echo "== 3/4 submuestreo (semilla por especie: no altera las existentes)"
rm -rf trabajo/sub/Animales
python code/subsample_train.py --seq_dir datos/bundle4_Animales/Animales \
  --out_dir trabajo/sub/Animales --ratio 1 | tail -3

echo "== 4/4 embeddings de las nuevas (salta las ya calculadas)"
python code/precompute_embeddings.py --seq_dir trabajo/sub/Animales \
  --out_dir trabajo/emb/Animales --splits train --batch_size 32 --token_batch 128

echo "== entrenando (negativas dificiles + ensamble + umbral en validacion)"
python code/modelo_final.py --emb_dir trabajo/emb/Animales \
  --out_dir trabajo/final/Animales_v2 --seeds 3
