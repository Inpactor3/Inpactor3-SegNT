# Celda para Colab / Kaggle — entrenamiento v2 (4×6 kb, max pooling)

## Antes (una sola vez)

Sube a Drive (`Mi unidad/Inpactor2_gpu/`):

- `code.zip` (`bundles/inpactor2_code.zip`, regenerado con los scripts v2)
- `data/<Reino>/` → `seq_bundle` con `--chunks 4` (24 kb)
- `data_eval8/<Reino>/` → `seq_bundle` con `--chunks 8 --mode eval_full` (48 kb)
- opcional: `data/<Reino>/gold/` (genoma + BED curado) para el test oro

Se generan con (PC local, disco externo conectado):

```bash
python training/make_sequence_bundle.py --jsonl_dir consolidated/Hongos \
  --genomes_dir /ruta/genomes_449/Hongos --out_dir seq_bundle/Hongos --chunks 4

python training/make_sequence_bundle.py --jsonl_dir consolidated/Hongos \
  --genomes_dir /ruta/genomes_449/Hongos --out_dir seq_bundle_eval8/Hongos \
  --chunks 8 --mode eval_full --splits val,test
```

## Celda comun

Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/Inpactor2_gpu'
```

Kaggle (dataset privado en `/kaggle/input/inpactor2-gpu`):
```python
import shutil, os
ROOT = '/kaggle/working/Inpactor2_gpu'
os.makedirs(ROOT, exist_ok=True)
shutil.copytree('/kaggle/input/inpactor2-gpu', ROOT, dirs_exist_ok=True)
```

Instalacion y codigo:
```python
!pip install -q transformers
!unzip -oq {ROOT}/code.zip -d /content/code   # Kaggle: -d /kaggle/working/code
%cd /content/code
```

## Entrenamiento (una celda por reino)

```python
# Hongos
!python colab_run.py --reino Hongos --seq_dir {ROOT}/data/Hongos \
  --drive_root {ROOT} --epochs 100 --patience 10 --class_weight --calibrate
```

```python
# Plantas: balance 1:1 + class_weight + calibracion
!python colab_run.py --reino Plantas --seq_dir {ROOT}/data/Plantas \
  --drive_root {ROOT} --epochs 100 --patience 10 \
  --balance global --class_weight --calibrate
```

```python
# Animales: idem. Si degrada por especies sin negativas: --balance species
!python colab_run.py --reino Animales --seq_dir {ROOT}/data/Animales \
  --drive_root {ROOT} --epochs 100 --patience 10 \
  --balance global --class_weight --calibrate
```

## Evaluacion a 48 kb (8 bloques)

```python
!python precompute_embeddings.py --seq_dir {ROOT}/data_eval8/Plantas \
  --out_dir {ROOT}/Plantas/emb48 --splits val,test --batch_size 4
!python train_mlp.py --emb_dir {ROOT}/Plantas/emb48 --out_dir {ROOT}/Plantas/ckpt48 \
  --epochs 1 --patience 1        # solo reutiliza el cabezal: usar evaluate_gold
```

La ruta recomendada para comparar 24 kb vs 48 kb es correr `evaluate_gold.py`
directamente sobre el genoma curado con `--chunks 4` y `--chunks 8`.

## Test oro

```python
!python evaluate_gold.py --genome {ROOT}/Plantas/gold/TAIR10.fasta.gz \
  --bed {ROOT}/Plantas/gold/tair10_te.bed \
  --head {ROOT}/Plantas/ckpt/best.pt --out {ROOT}/Plantas/gold4 \
  --reino Plantas --chunks 4 --overlap_mode ref

!python evaluate_gold.py --genome {ROOT}/Plantas/gold/TAIR10.fasta.gz \
  --bed {ROOT}/Plantas/gold/tair10_te.bed \
  --head {ROOT}/Plantas/ckpt/best.pt --out {ROOT}/Plantas/gold8 \
  --reino Plantas --chunks 8 --mode eval_full --overlap_mode ref
```

## Donde quedan los resultados

```
<ROOT>/<Reino>/
  emb/    <split>/<especie>_emb.npy · _labels.npy · _meta.jsonl · _pool.npy
  ckpt/   best.pt (incluye umbral calibrado) · metrics.csv ·
          calibration.json · test_metrics.json
  logs/   gpu_info.txt · precompute.log · train.log
```

## Si se reinicia el runtime

- Embeddings y `best.pt` están en Drive/working persistente.
- Re-ejecuta la celda: `colab_run.py` detecta `emb/*_emb.npy` y salta el precalculo.
- El MLP se reentrena desde cero en minutos.
- Si se corta el precalculo a mitad, divide con `--num_shards 3 --shard_index 0..2`
  desde `precompute_embeddings.py` y relanza en otras sesiones.

## Tiempos en A100 40 GB

| Reino | Precalculo 24 kb | 100 ep MLP | Precalculo 48 kb (val+test) |
|---|---|---|---|
| Hongos | 25–40 min | 10–20 min | ~10 min |
| Plantas | 3–4 h | 20–40 min | ~40 min |
| Animales | 7–10 h | 30–60 min | ~15 min |

En T4 gratis multiplicar ~3×: solo hongos y plantas entran en una sesión.
