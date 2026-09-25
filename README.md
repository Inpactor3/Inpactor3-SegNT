# Inpactor3-SegNT

Detección de LTR-retrotransposones en genomas vegetales usando un **Genome
Language Model** pre-entrenado (Nucleotide Transformer) con una **cabeza de
segmentación** por token, al estilo SegmentNT.

**Proyecto Integrador II · Universidad de Caldas · 2026**

## Motivación

La versión anterior de Inpactor3 (`../Inpactor3/`) usó una CNN 1D estilo YORO
entrenada desde cero. Encontró un techo de recall alrededor de 26-30% F1
sobre Arabidopsis debido a:

1. **Fondos artificiales** que el modelo aprendía a distinguir del genoma real.
2. **Entrenamiento desde cero** — sin transferencia desde un modelo pre-entrenado
   con millones de bases de contexto biológico.
3. **Composición nucleotídica** irreal en los flancos negativos.

Esta arquitectura resuelve los tres problemas:

- **Base pre-entrenada** (Nucleotide Transformer entrenado en 850 especies).
- **Segmentación por token**: cada nucleótido recibe una etiqueta
  (LTR-RT / fondo) directamente, sin ancajes ni cajas.
- **Contexto largo** (hasta 12 kb por ventana) capturado por atención transformer.

## Arquitectura

```
Ventana ADN 6 kb (~1000 tokens BPE)
       │
       ▼
┌─────────────────────────┐
│ Nucleotide Transformer  │  ← pretrained (500M o 100M params)
│ (frozen o fine-tuned)   │
└──────────┬──────────────┘
           │  embeddings (B, T, D)
           ▼
┌─────────────────────────┐
│ Cabeza de segmentación  │  ← 2-3 capas Conv1D o Linear
│ per-token classifier    │
└──────────┬──────────────┘
           ▼
    (B, T, K) logits por token
     K = {fondo, LTR-RT}  (o {fondo, Copia, Gypsy, ...} para multiclase)
           │
           ▼
    Post-proceso: mayoría por celda de 100 bp
    → cajas contiguas de LTR-RT detectadas
```

## Estructura del repositorio

```
Inpactor3_segnt/
├── src/inpactor3_segnt/
│   ├── model.py           # NTSegmentation: encoder + cabeza
│   ├── dataset.py         # ventanas + tokenización + etiquetas por token
│   ├── train.py           # bucle de entrenamiento
│   ├── predict.py         # inferencia sobre genoma
│   └── evaluate.py        # métricas por ventana (F1) + IoU
├── scripts/
│   ├── download_arabidopsis.sh   # baja TAIR10 y GFF de repeats
│   ├── build_labels.py           # convierte anotaciones a etiquetas por bp
│   └── run_all.sh                # pipeline completo
├── configs/
│   ├── nt_50m.yaml        # config para NT-v2-50M (más ligero)
│   └── nt_500m.yaml       # config para NT-v2-500M o SegmentNT
├── notebooks/
│   └── colab_train.ipynb  # notebook listo para Colab
├── docs/
│   └── methodology.md     # split por cromosoma, métricas, ground truth
└── requirements.txt
```

## Datos y split (metodología rigurosa)

Adoptamos la división por cromosoma del informe de referencia:

| Cromosoma | Papel | # ventanas | # con LTR-RT (Inpactor2) |
|---|---|---|---|
| 1 | **Entrenar** | 609 | 17 |
| 2 | **Entrenar** | 394 | 18 |
| 3 | **Entrenar** | 470 | 20 |
| 4 | **Validar** (elegir umbral) | 372 | 20 |
| 5 | **Examen** (nunca visto) | 540 | 21 |

Ground truth = `Inpactor2_predictions.tab` sobre TAIR10 (109 elementos).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline completo

```bash
# 1) Bajar Arabidopsis TAIR10 (5 cromosomas + Inpactor2 ground truth)
bash scripts/download_arabidopsis.sh

# 2) Construir etiquetas por bp
python scripts/build_labels.py \
    --genome data/raw/TAIR10.fasta \
    --annotations data/raw/Inpactor2_predictions.tab \
    --out data/processed/labels.parquet

# 3) Entrenar (Colab con GPU) — ver notebooks/colab_train.ipynb
python -m inpactor3_segnt.train --config configs/nt_50m.yaml

# 4) Predecir sobre cromosoma 5
python -m inpactor3_segnt.predict \
    --checkpoint models/best.pt \
    --genome data/raw/TAIR10.fasta \
    --scaffold chr5 \
    --out results/chr5_predictions.tab

# 5) Evaluar contra Inpactor2
python -m inpactor3_segnt.evaluate \
    --pred results/chr5_predictions.tab \
    --truth data/raw/Inpactor2_predictions.tab \
    --scaffold chr5 \
    --report results/chr5_report.md
```

## Modelo base recomendado

| Modelo | Parámetros | Contexto | RAM GPU (fine-tune) |
|---|---|---|---|
| `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | 50M | 12 kb | ~4 GB ← recomendado para Colab free |
| `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | 100M | 12 kb | ~7 GB |
| `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | 500M | 12 kb | ~12 GB (Colab Pro) |
| `InstaDeepAI/segment_nt` | 500M | 30 kb | fine-tune caro; usar zero-shot |

Empezamos con **NT-v2-50M** — cabe en Colab gratis y es un baseline honesto.

## Métricas objetivo

- **F1 por ventana** (presencia binaria): meta ≥ 40% (baseline informe = 26.3%).
- **Precision/Recall por linaje**: meta ≥ 50% top-3 linajes.
- **IoU medio de coordenadas**: meta ≥ 0.5.

Reportar todo cromosoma-a-cromosoma sobre chr5, sin data leakage.

## Licencia

Por definir — se alineará con Inpactor2 (GPL-3.0).
