#!/bin/bash
# Generado por rehacer_split_animales.py — REVISAR ANTES DE EJECUTAR
set -e
cd "$HOME/Escritorio/Inpactor3"
. .venv/bin/activate

B=datos/bundle4_Animales/Animales
E=trabajo/emb/Animales
S=trabajo/sub/Animales

echo "== 1/5 moviendo npz ==="
mv "$B/val/Acanthaster_planci___GCF_054643075.1.npz" "$B/train/"
mv "$B/train/Acanthopagrus_latus__GCF_904848185.1.npz" "$B/test/"
mv "$B/val/Acropora_acuminata___GCA_014633975.1.npz" "$B/test/"
mv "$B/train/Acropora_hyacinthus__GCA_964291705.1.npz" "$B/val/"
mv "$B/train/Amphilophus_citrinellus__GCA_013435755.1.npz" "$B/val/"
mv "$B/val/Arctopsyche_grandis___GCF_051622035.1.npz" "$B/test/"
mv "$B/val/Aricia_agestis___GCF_905147365.1.npz" "$B/test/"
mv "$B/val/Athalia_rosae___GCF_917208135.1.npz" "$B/train/"
mv "$B/test/Branchiostoma_lanceolatum___GCF_035083965.1.npz" "$B/train/"
mv "$B/test/Caenorhabditis_remanei___GCF_010183535.1.npz" "$B/train/"
mv "$B/train/Capromys_pilorides__GCA_004027915.1.npz" "$B/val/"
mv "$B/train/Cariama_cristata__GCF_000690535.1.npz" "$B/test/"
mv "$B/train/Charadrius_alexandrinus__GCA_053477275.1.npz" "$B/val/"
mv "$B/train/Chrysoperla_carnea__GCF_905475395.1.npz" "$B/test/"
mv "$B/val/Drosophila_willistoni___GCF_018902025.1.npz" "$B/train/"
mv "$B/train/Ellobius_talpinus__GCA_001685095.1.npz" "$B/test/"
mv "$B/train/Emydura_subglobosa__GCA_054346155.1.npz" "$B/val/"
mv "$B/train/Gigantopelta_aegis__GCF_016097555.1.npz" "$B/test/"
mv "$B/train/Gopherus_agassizii__GCA_002896415.1.npz" "$B/test/"
mv "$B/test/Nasonia_giraulti___GCA_016647725.1.npz" "$B/train/"
mv "$B/val/Pelagia_noctiluca___GCA_963924455.1.npz" "$B/train/"
mv "$B/test/Porites_divaricata__GCA_964035745.1.npz" "$B/train/"
mv "$B/test/Xenos_peckii___GCA_040167675.1.npz" "$B/train/"

echo "== 2/5 regenerando train submuestreado (semilla por especie) ==="
rm -rf "$S"
python code/subsample_train.py --seq_dir "$B" --out_dir "$S" --ratio 1

echo "== 3/5 borrando embeddings de las especies movidas ==="
rm -f "$E"/*/Acanthaster_planci___GCF_054643075.1_*.npy
rm -f "$E"/*/Acanthopagrus_latus__GCF_904848185.1_*.npy
rm -f "$E"/*/Acropora_acuminata___GCA_014633975.1_*.npy
rm -f "$E"/*/Acropora_hyacinthus__GCA_964291705.1_*.npy
rm -f "$E"/*/Amphilophus_citrinellus__GCA_013435755.1_*.npy
rm -f "$E"/*/Arctopsyche_grandis___GCF_051622035.1_*.npy
rm -f "$E"/*/Aricia_agestis___GCF_905147365.1_*.npy
rm -f "$E"/*/Athalia_rosae___GCF_917208135.1_*.npy
rm -f "$E"/*/Branchiostoma_lanceolatum___GCF_035083965.1_*.npy
rm -f "$E"/*/Caenorhabditis_remanei___GCF_010183535.1_*.npy
rm -f "$E"/*/Capromys_pilorides__GCA_004027915.1_*.npy
rm -f "$E"/*/Cariama_cristata__GCF_000690535.1_*.npy
rm -f "$E"/*/Charadrius_alexandrinus__GCA_053477275.1_*.npy
rm -f "$E"/*/Chrysoperla_carnea__GCF_905475395.1_*.npy
rm -f "$E"/*/Drosophila_willistoni___GCF_018902025.1_*.npy
rm -f "$E"/*/Ellobius_talpinus__GCA_001685095.1_*.npy
rm -f "$E"/*/Emydura_subglobosa__GCA_054346155.1_*.npy
rm -f "$E"/*/Gigantopelta_aegis__GCF_016097555.1_*.npy
rm -f "$E"/*/Gopherus_agassizii__GCA_002896415.1_*.npy
rm -f "$E"/*/Nasonia_giraulti___GCA_016647725.1_*.npy
rm -f "$E"/*/Pelagia_noctiluca___GCA_963924455.1_*.npy
rm -f "$E"/*/Porites_divaricata__GCA_964035745.1_*.npy
rm -f "$E"/*/Xenos_peckii___GCA_040167675.1_*.npy

echo "== 4/5 recalculando embeddings (salta los que existen) ==="
python code/precompute_embeddings.py --seq_dir "$S" --out_dir "$E" --splits val,test --batch_size 32 --token_batch 128
python code/precompute_embeddings.py --seq_dir "$S" --out_dir "$E" --splits train --batch_size 32 --token_batch 128

echo "== 5/5 reentrenando ==="
python code/train_mlp.py --emb_dir "$E" --out_dir trabajo/ckpt/Animales_split2 --epochs 100 --patience 10 --batch_size 4096 --balance global --class_weight --calibrate --select balanced_accuracy

echo "== LISTO =="
