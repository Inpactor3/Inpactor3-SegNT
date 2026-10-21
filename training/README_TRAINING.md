# Entrenamiento Inpactor2 Detect — paquete listo

Este directorio contiene lo necesario para entrenar en otra máquina con GPU.
El backbone (Nucleotide Transformer 500M) está **congelado**: se precalculan
embeddings una vez y luego se entrena solo el cabezal MLP.

---

## 1. Estado de los datos

| Reino | Especies | Ventanas | % positivo | Archivo |
|---|---|---|---|---|
| Hongos | 117 | 79.472 | 46% | `consolidated/Hongos` |
| Animales | 81 | ~801.000 | 69% | `consolidated/Animales` |
| Plantas | 53+ | ~296.000 | 92% | `consolidated/Plantas` |

Los JSONL consolidados ya tienen `label` (1 = LTR, 0 = negativo) y `negative_type`.

---

## 2. Cómo llevar los datos

### Opción A — Llevar el disco externo (recomendado)
El disco tiene los genomas completos:
- `genomes_449/Hongos` (1,8 GB)
- `genomes_449/Animales` (42 GB)
- `genomes_449/Plantas` (48 GB)

En la otra máquina se monta y se usan las rutas tal cual.

### Opción B — Paquete pequeño de secuencias
Genera un `.npz` por especie con solo los 6 kb que usa el modelo:

```bash
python make_sequence_bundle.py \
  --jsonl_dir  /ruta/consolidated/Hongos \
  --genomes_dir /ruta/genomes_449/Hongos \
  --out_dir    /ruta/seq_bundle/Hongos
```

Tamaños aproximados:
- Hongos: ~0,5 GB
- Animales: ~4,8 GB (≈1,5 GB comprimido)
- Plantas: ~1,8 GB (≈0,6 GB comprimido)

---

## 3. Instalación en la máquina GPU

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy tqdm
```

---

## 4. Paso 1 — Precalcular embeddings (una sola vez)

Con genomas:
```bash
python precompute_embeddings.py \
  --jsonl_dir  /ruta/consolidated/Hongos \
  --genomes_dir /ruta/genomes_449/Hongos \
  --out_dir    /ruta/emb/Hongos \
  --batch_size 32
```

Con paquete de secuencias:
```bash
python precompute_embeddings.py \
  --seq_dir /ruta/seq_bundle/Hongos \
  --out_dir /ruta/emb/Hongos \
  --batch_size 32
```

Genera por split: `train_emb.npy`, `train_labels.npy`, `train_meta.jsonl`
(y lo mismo para val/test).

**Tiempo estimado (A100):**
- Hongos: ~15–25 min
- Animales: ~4–6 h
- Plantas: ~1,5–2,5 h

---

## 5. Paso 2 — Entrenar el MLP (100 épocas)

```bash
python train_mlp.py \
  --emb_dir /ruta/emb/Hongos \
  --out_dir /ruta/ckpt/Hongos \
  --epochs 100 --patience 10 --batch_size 4096 --class_weight
```

- Guarda `best.pt` (mejor F1 de validación) y `metrics.csv` por época.
- Hace **early stopping** (patience 10): no siempre llega a 100.
- Al final evalúa en test y escribe `test_metrics.json`.

**Tiempo por 100 épocas:** 5–30 min por reino (el MLP es diminuto).

---

## 6. Tiempos totales realistas

| Reino | Embeddings | 100 épocas MLP | Total |
|---|---|---|---|
| Hongos | 20 min | 10–20 min | **< 1 h** |
| Animales | 5 h | 30–60 min | **~6 h** |
| Plantas | 2 h | 20–40 min | **~3 h** |
| **Todo** | ~7 h | ~1,5 h | **< 1 día** |

Con 4 días hay margen para ablaciones, comparación y reintentos.

---

## 7. Advertencias

1. **No hagas fine-tuning completo del NT-500M**: 100 épocas serían semanas.
   El diseño es backbone congelado + MLP.
2. **100 épocas sobreajustan** un MLP pequeño. Usa el mejor epoch por validación,
   no el último.
3. Para plantas (92% positivo), usa `--class_weight` o consigue más negativas.
4. Mantén el mismo `--nt_model` para comparar con los otros reinos.

---

## 8. Reproducibilidad

Todo el pipeline previo está documentado en `../../CAMBIOS_PROYECTO.md`:
- dedup de hits
- reetiquetado LTR
- negativas (genic, other_te, intergenic)
- splits por especie (sin fuga entre train/val/test)

---

## 9. Colab

Ver `COLAB_CELL.md`. Resumen:

- `colab_run.py` orquesta precalculo + entrenamiento.
- Resultados persistentes en Drive:
  `Inpactor2_gpu/<reino>/{emb,ckpt,logs}`.
- Logs: `gpu_info.txt`, `precompute.log`, `train.log`, `metrics.csv`, `test_metrics.json`.
- Si el runtime se reinicia, los embeddings ya estan en Drive y el precalculo se salta.
- Paquete de codigo listo: `bundles/inpactor2_code.zip`.
