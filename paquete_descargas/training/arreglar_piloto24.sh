#!/bin/bash
# Rehace el set de 4 bloques del piloto y repite la comparación.
set -e
cd "$HOME/Escritorio/Inpactor3"
. .venv/bin/activate

T4=/tmp/pilot4
B4=datos/bundle4_Animales/Animales
rm -rf "$T4"; mkdir -p "$T4"/{train,val,test}

echo "== train (12 especies del piloto)"
for f in /tmp/pilot8/train/*.npz; do
  [ -e "$f" ] || { echo "   falta bundle 48k; ejecuta piloto48.sh primero"; exit 1; }
  b=$(basename "$f")
  sp=$(echo "$b" | cut -d_ -f1,2)
  g=$(ls $B4/train/${sp}__*.npz 2>/dev/null | head -1)
  if [ -n "$g" ]; then ln -sf "$(readlink -f "$g")" "$T4/train/$b";
  else echo "   SIN PAR 24k: $b"; fi
done

echo "== val (3 especies)"
for sp in Acropora_acuminata Pelagia_noctiluca Aricia_agestis; do
  g=$(ls $B4/val/*${sp}*.npz 2>/dev/null | head -1)
  [ -n "$g" ] && ln -sf "$(readlink -f "$g")" "$T4/val/" || echo "   SIN PAR: $sp"
done

echo "== test (3 especies)"
for sp in Nasonia_giraulti Xenos_peckii Branchiostoma_lanceolatum; do
  g=$(ls $B4/test/*${sp}*.npz 2>/dev/null | head -1)
  [ -n "$g" ] && ln -sf "$(readlink -f "$g")" "$T4/test/" || echo "   SIN PAR: $sp"
done

echo "   train=$(ls "$T4"/train/*.npz 2>/dev/null | wc -l) val=$(ls "$T4"/val/*.npz 2>/dev/null | wc -l) test=$(ls "$T4"/test/*.npz 2>/dev/null | wc -l)"
[ "$(ls "$T4"/val/*.npz 2>/dev/null | wc -l)" -ge 3 ] || { echo "FALLO: val incompleto"; exit 1; }

echo "== embeddings de 24 kb"
python code/precompute_embeddings.py --seq_dir "$T4" --out_dir trabajo/piloto24/emb \
  --splits train,val,test --batch_size 32 --token_batch 128

echo "== comparación"
python code/entrenar_piloto.py
