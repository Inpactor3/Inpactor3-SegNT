# Entrenamiento Inpactor2 Detect v2 — entrada de 4×6 kb con max pooling

Backbone **Nucleotide Transformer 500M congelado**. Cambios de esta version:

1. **Entrada ampliada**: cada ventana de 50 kb aporta **4 bloques de 6 kb (24 kb)**
   en offsets `0, 14667, 29333, 44000`. Los bloques se embeben y se combinan con
   **max pooling** (`emb = max_b(mean_tokens(NT(bloque_b)))`).
2. **Evaluacion a 48 kb**: el mismo modelo se evalúa con **8 bloques de 6 kb**
   contiguos (offsets `0, 6000, …, 42000`) en la GPU A100. Como el pooling es un
   `max`, el cabezal no cambia con el numero de bloques.
3. **Reentrenamiento de los tres reinos** con balance **1:1** en plantas y animales
   (`--balance global` o `--balance species`) + `--class_weight`.
4. **Calibracion de umbral en validacion** (balanced accuracy / Youden). El umbral
   queda guardado dentro de `best.pt` y lo usa la evaluacion gond.
5. **Test oro armonizado**: misma definicion LTR que `relabel_ltr.py`
   (`class` contiene `LTR`) y solapamiento configurable (`--overlap_mode ref|window`).

---

## 1. Estado de los datos (bundles locales generados en el PC)

| Reino | Especies | train | val | test | % pos train |
|---|---|---|---|---|---|
| Hongos | 116 | 58.075 | 14.067 | 7.330 | 44,3% |
| Plantas | 66 | 408.713 | 52.672 | 45.250 | 95,9% |
| Animales | 57 | 801.118 | 69.135 | 16.926 | 69,0% |

Splits disjuntos por especie (0 ventanas compartidas entre train/val/test).
El desbalance de plantas (95,9% positivo) es de origen: solo una parte de las
especies tiene negativas. Por eso **plantas y animales se entrenan balanceados
1:1** (además de `class_weight`).

Cobertura pendiente (especies con genoma pero sin ventanas):

| Reino | Genomas | Con ventanas | Sin ventanas |
|---|---|---|---|
| Hongos | 124 | 116 | 8 |
| Plantas | 141 | 66 | 75 |
| Animales | 135 | 57 | 78 |

Genera el detalle con `coverage_report.py` (ver §7).

---

## 2. Regenerar los bundles de secuencia

Bloques de entrenamiento (4×6 kb = 24 kb):
```bash
python make_sequence_bundle.py \
  --jsonl_dir  <consolidated>/<Reino> --genomes_dir <genomes_449>/<Reino> \
  --out_dir    seq_bundle/<Reino> --chunks 4 --splits train,val,test
```

Paquete de evaluacion a 48 kb (8 bloques contiguos):
```bash
python make_sequence_bundle.py \
  --jsonl_dir  <consolidated>/<Reino> --genomes_dir <genomes_449>/<Reino> \
  --out_dir    seq_bundle_eval8/<Reino> --chunks 8 --mode eval_full --splits val,test
```

Layout por ventana guardado en el npz:
`seq (N, chunks, 6000) uint8`, `label (N,) int8`, `window_id (N,)`,
`offsets (chunks,)`, `chunk_bp ()`.

Tamaño aproximado (comprimido): Hongos ~0,35 GB · Plantas ~2,5 GB · Animales ~5 GB
(crece ×4 frente al bundle de 1 bloque, pero ya no hace falta mover los genomas).

---

## 3. Instalacion (Colab / Kaggle)

```bash
pip install torch tqdm
pip install transformers
# Kaggle: transformers ya viene; torch usa la rueda CUDA de la imagen (~2.10)
```

---

## 4. Precalcular embeddings (max pooling sobre bloques)

```bash
python precompute_embeddings.py \
  --seq_dir <seq_bundle>/<Reino> --out_dir emb/<Reino> \
  --splits train,val,test --batch_size 8 --token_batch 64 --chunks 4
```

- `--batch_size` = ventanas por paso; cada ventana son `chunks` bloques, así que el
  trabajo por forward es `batch_size × token_batch` bloques.
- Reanudable: si `<tag>_emb.npy` existe se salta (colab_run ya lo hace).
- Salida por especie: `<tag>_emb.npy (N,1280) f16`, `_labels.npy`, `_meta.jsonl`,
  `_pool.npy` (fraccion de N por bloque), `_offsets.npy`.

**48 kb para evaluacion** (solo val/test, una pasada):
```bash
python precompute_embeddings.py --seq_dir <seq_bundle_eval8>/<Reino> \
  --out_dir emb48/<Reino> --splits val,test --batch_size 4 --token_batch 64
```

---

## 5. Entrenar el MLP

```bash
# Hongos (ya casi balanceado)
python train_mlp.py --emb_dir emb/Hongos --out_dir ckpt/Hongos \
  --epochs 100 --patience 10 --class_weight --calibrate

# Plantas y Animales: balance 1:1 + class_weight + calibracion
python train_mlp.py --emb_dir emb/Plantas --out_dir ckpt/Plantas \
  --epochs 100 --patience 10 --balance global --class_weight --calibrate
python train_mlp.py --emb_dir emb/Animales --out_dir ckpt/Animales \
  --epochs 100 --patience 10 --balance global --class_weight --calibrate
```

- `--balance global`: submuestrea el train a 1:1.
- `--balance species`: 1:1 por especie (evita que las especies sin negativas
  dominen el balance). Usalo si el rendimiento por especie se degrada.
- `--class_weight`: `pos_weight = n_neg/n_pos` del train ya balanceado (≈1,0 con
  `--balance global`; sigue siendo relevante con `--balance species`).
- `--calibrate`: elige el umbral por **balanced accuracy** en validacion y lo
  guarda en `calibration.json` + `best.pt["threshold"]`.
- Salidas: `best.pt`, `metrics.csv` (con `val_bal_acc`), `calibration.json`,
  `test_metrics.json` (metricas al umbral calibrado y a 0,5).

---

## 6. Test oro (anotacion curada)

Con el umbral calibrado y la definicion LTR armonizada:
```bash
python evaluate_gold.py --genome TAIR10.fasta.gz --bed tair10_te.bed \
  --head ckpt/Plantas/best.pt --out results/Gold_TAIR10 --reino Plantas \
  --chunks 4 --overlap_mode ref --min_frac 0.2
```

- **Misma regla LTR que el entrenamiento**: se acepta si el `class` del nombre del
  BED contiene `LTR` (`LTR/Gypsy`, `LTR/Copia`, `LTR`). Se escribe
  `ltr_definition.json` junto con el numero de TEs LTR y no-LTR del archivo.
- `--overlap_mode ref` (por defecto): positivo si ≥20% del **TE** solapa la ventana
  (coherente con `relabel_ltr.py`). `--overlap_mode window`: ≥20% de la **ventana**
  (canon de Inpactor/EDTA).
- Reporta `at_calibrated_threshold`, `at_050`, `always_positive` y PR-AUC.
- `--chunks 8 --mode eval_full` para la variante de 48 kb (compara el efecto de la
  entrada ampliada sobre el mismo test oro).

---

## 7. Cobertura de especies y negativas

```bash
python coverage_report.py --seq_dir <seq_bundle>/<Reino> \
  --genomes_dir <genomes_449>/<Reino> --out coverage_<Reino>.json
```
Lista las `species_tag` con genoma pero sin ventanas y cuenta las negativas por tipo
presentes (`genic`, `other_te`, `intergenic`). Para cada faltante hay que relanzar
downstream (BED → dedup → `build_windows.py` → `relabel_ltr.py` →
`negative_windows.py` → `consolidate.py`); sin disco externo (BED/GFF) no se puede
inventar. Las dos categorias de negativas sin implementar siguen siendo
`heterochromatin` y `simple_repeats` (requieren RepeatMasker / centromeros).

---

## 8. Tiempos (A100 40 GB, estimados)

| Reino | Precalculo 24 kb | Precalculo 48 kb (val+test) | 100 ep MLP |
|---|---|---|---|
| Hongos | ~25–40 min | ~10 min | 10–20 min |
| Plantas | ~3–4 h | ~40 min | 20–40 min |
| Animales | ~7–10 h | ~15 min | 30–60 min |

El coste crece ×4 (24 kb) porque cada ventana hace 4 forward del backbone.
Si se acaba el tiempo, divide el precalculo con `--num_shards/--shard_index`
(resumible: los `<tag>_emb.npy` ya hechos se saltan).

---

## 9. Advertencias

1. No hacer fine-tuning del NT-500M: el diseño es backbone congelado + MLP.
2. 100 epocas sobreajustan un MLP pequeño: usa `best.pt` (mejor F1 de validacion).
3. Usa **el mismo `--nt_model`** en entrenamiento y en evaluacion gond
   (`evaluate_gold.py` aborta si el `hidden` del checkpoint no coincide).
4. Plantas: el test sigue siendo 92% positivo; el F1 alto puede ser optimista.
   Reporta balanced accuracy y PR-AUC junto con F1.
5. El umbral calibrado en validacion no se re-ajusta en test oro: se aplica tal cual.
