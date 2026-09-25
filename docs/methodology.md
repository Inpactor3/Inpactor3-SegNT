# Metodología

## Ground truth

Se toma `Inpactor2_predictions.tab` como referencia. Es la salida oficial
de Inpactor2 sobre el mismo genoma TAIR10:

- 109 LTR-RTs anotados por Inpactor2 sobre los 5 cromosomas.
- Formato de 8 columnas: `seqid start end length lineage det_p filt_p cls_p`.

## Split por cromosoma (evita data leakage)

Del informe de referencia interno, adoptamos:

| Cromosoma | Ventanas 50 kb | Con LTR-RT | Papel |
|---|---|---|---|
| 1 | 609 | 17 | Entrenamiento |
| 2 | 394 | 18 | Entrenamiento |
| 3 | 470 | 20 | Entrenamiento |
| 4 | 372 | 20 | Validación (umbral y early stopping) |
| 5 | 540 | 21 | **Examen final — nunca visto** |

Total: 2 385 ventanas, 96 con al menos un elemento (~4% de las ventanas
son positivas — desbalance real).

## Métricas

### Nivel ventana (métrica principal del informe)

Una ventana de 50 kb cuenta como positiva si **cualquier token** dentro
de ella se predice como LTR-RT. Comparamos contra ventanas donde
Inpactor2 anotó al menos un elemento.

- **TP**: ventana con LTR-RT verdadero → predijo LTR-RT.
- **FP**: ventana vacía → predijo LTR-RT.
- **FN**: ventana con LTR-RT verdadero → predijo vacío.
- **TN**: ventana vacía → predijo vacío.

Reportamos `Precisión = TP / (TP+FP)`, `Recall = TP / (TP+FN)`, `F1`.

### Nivel coordenadas (segunda métrica)

Para las ventanas TP, adicionalmente medimos IoU entre las cajas
predichas y las verdaderas. Meta: IoU medio ≥ 0.5.

### Nivel linaje

Cuando `num_classes > 2`, la matriz de confusión de linajes se calcula
sobre los TP posicionales.

## Comparación con enfoques previos

| Enfoque | F1 en chr5 |
|---|---|
| CNN 1D YORO, fondo aleatorio uniforme (`../Inpactor3/`) | 0-7% |
| CNN 1D YORO, fondo otro TE | 7.5% |
| CNN 1D YORO, fondo genoma real (informe intento 3) | 26.3% |
| **SegNT (este proyecto) — meta** | **≥ 40%** |

El salto esperado de +13 puntos viene de:

1. **Transfer learning**: NT ya "sabe" leer ADN de 850 especies.
2. **Sin fondo sintético**: entrenamos sobre las ventanas reales de los
   cromosomas 1-3 con sus anotaciones Inpactor2, no plantamos elementos.
3. **Segmentación fina**: cada token de ~6 bp recibe etiqueta, en vez de
   una decisión global por ventana.

## Modelo base

Empezamos con `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`:

- Pre-entrenado sobre 850 especies (incluye Arabidopsis).
- BPE de longitud 6 (aprox), contexto máximo ~12 kb.
- Cabe en Colab GPU T4 (15 GB VRAM) con `batch_size=4`.

## Ampliación futura

- Escalar a `nucleotide-transformer-v2-100m-multi-species` (mejor
  capacidad, +Colab Pro).
- Multiclase por linaje (13 clases InpactorDB).
- Evaluar en arroz, sorgo, maíz.
