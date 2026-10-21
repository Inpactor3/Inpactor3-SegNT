# Celda para Colab — entrenamiento con logs y resultados

## Antes (una sola vez)

Sube a tu Drive, en `Mi unidad/Inpactor2_gpu/`:

- `code.zip` (está en `bundles/inpactor2_code.zip`)
- `data/Hongos/` (seq bundle de hongos, ~150 MB)
- `data/Plantas/` (~1 GB)
- `data/Animales/` (~1.5 GB)

Los `seq_bundle` se generan en el PC local con:

```bash
python training/make_sequence_bundle.py \
  --jsonl_dir consolidated/Hongos \
  --genomes_dir /ruta/genomes_449/Hongos \
  --out_dir seq_bundle/Hongos
```

(repetir para Animales y Plantas; requiere el disco externo conectado)

## Celda de Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import os
ROOT = '/content/drive/MyDrive/Inpactor2_gpu'
os.makedirs(ROOT, exist_ok=True)

!pip install -q transformers
!unzip -oq {ROOT}/code.zip -d /content/code
%cd /content/code
```

Luego, una celda por reino:

```python
!python colab_run.py \
  --reino Hongos \
  --seq_dir {ROOT}/data/Hongos \
  --drive_root {ROOT} \
  --epochs 100 --patience 10
```

```python
!python colab_run.py \
  --reino Plantas \
  --seq_dir {ROOT}/data/Plantas \
  --drive_root {ROOT} \
  --epochs 100 --patience 10 --class_weight
```

```python
!python colab_run.py \
  --reino Animales \
  --seq_dir {ROOT}/data/Animales \
  --drive_root {ROOT} \
  --epochs 100 --patience 10
```

## Dónde quedan los resultados

```
Mi unidad/Inpactor2_gpu/
  Hongos/
    emb/    *_emb.npy, *_labels.npy, *_meta.jsonl
    ckpt/   best.pt, metrics.csv, test_metrics.json
    logs/   gpu_info.txt, precompute.log, train.log
```

- `gpu_info.txt`: GPU, VRAM, CUDA, versiones.
- `precompute.log`: progreso del precalculo.
- `train.log`: salida por época.
- `metrics.csv`: loss y F1 por época.
- `test_metrics.json`: métricas finales en test.

## Si Colab se reinicia

- Los **embeddings** y el **best.pt** están en Drive.
- Re-ejecuta la misma celda: detecta `emb/*.npy` y **salta el precalculo**.
- El MLP se reentrena desde cero en minutos (es pequeño); no se pierde el trabajo caro.

## Viabilidad por reino (A100)

| Reino | Precalculo | 100 épocas | Total | ¿Una sesión? |
|---|---|---|---|---|
| Hongos | ~20 min | 10–20 min | **< 1 h** | ✅ |
| Plantas | ~2 h | 20–40 min | **~3 h** | ✅ |
| Animales | ~5 h | 30–60 min | **~6 h** | ⚠️ puede requerir 2 sesiones |

En T4 (gratis) los tiempos se multiplican ~2–3×: hongos sigue siendo viable; animales probablemente no.
