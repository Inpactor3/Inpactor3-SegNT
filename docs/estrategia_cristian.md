# Estrategia Cristian · MIL binario con Nucleotide Transformer

**Autor**: Cristian
**Modelo**: Inpactor3-SegNT · `NTBinaryClassifier` (NT-50M + MIL max-pool + cabeza binaria)
**Entrenamiento ya corrido**: 34 épocas en V100 32GB · F1 test = 0.67 · F1 Arabidopsis chr5 = 0.08
**Fecha**: viernes [defensa con el profesor]

---

## 1. Resumen ejecutivo

Mi estrategia aplica el **protocolo común del equipo** (RepeatMasker con biblioteca
PanTEon por especie, partición por especie, test con prevalencia natural) y
**mantiene mi arquitectura ya entrenada**: clasificación binaria a nivel de
**ventana de 50 kb** usando el **Nucleotide Transformer** como extractor de
features congelado con **Multiple Instance Learning (MIL)** sobre chunks.

Mi diferenciador frente a Jhonier y Alejandro **no es la granularidad de etiqueta**
(los tres entregan al final una etiqueta binaria por ventana), **sino el DATASET**:
las ventanas negativas se muestrean como una **mezcla curada de 5 tipos de ADN
real** con metadata explícita, para evitar el domain shift que ya detectamos.

---

## 2. Problema que ataco

El entrenamiento previo con PanTEon pre-procesado dio:

- **F1 = 0.67** en test balanceado (dentro de dominio).
- **F1 = 0.08** en Arabidopsis chr5 real (fuera de dominio).

Caída de 0.59 puntos → **domain shift severo**.

Causa raíz: las ventanas negativas de PanTEon son fondo neutro, mientras que
en genoma real son una mezcla de genes, otros TE no-LTR, heterocromatina, etc.

**No ataco la arquitectura (ya convergió a su techo con 0.13 M params entrenables).
Ataco el dataset.**

---

## 3. Mi arquitectura (ya entrenada, no cambia)

```
Ventana 50 kb
    ↓ dividir en 10 chunks de 5 kb
    ↓ tokenizar cada chunk (NT 6-mer, ~1000 tokens)
    ↓
[10 × Encoder NT-50M congelado]
    ↓ mean-pool intra-chunk sobre tokens válidos
[10 vectores de 512]
    ↓ MAX-POOL entre chunks (MIL)
[1 vector de 512]
    ↓ Linear(512→256) → GELU → Dropout → Linear(256→1)
[1 logit] → sigmoid → BCE sobre label de ventana (0 ó 1)
```

**Detalles clave**:
- Encoder congelado → solo entrenan **0.13 M params** (la cabeza).
- MIL max-pool: si cualquiera de los 10 chunks "grita TE", la ventana es positiva.
- Pérdida: **BCE sobre label de ventana** (1 número por ventana).
- Esto es lo que corrió 34 épocas en V100.

**Esta arquitectura NO cambia.** Lo que cambia es el dataset.

---

## 4. Mi diferenciador real frente al equipo

Las 3 estrategias terminan dándole al modelo **ventanas de 50 kb con etiqueta binaria**.
Lo que varía es cómo se construyen esas ventanas:

| | Jhonier | Alejandro | **Cristian (yo)** |
|---|---|---|---|
| **Modelo backbone** | CNN (YORO / Inpactor 2) | CNN (U-Net colapsado a ventana) | **Transformer (NT-50M)** |
| **Pooling / agregación** | Convolucional global | Downsampling | **MIL max-pool entre chunks** |
| **Negativas** | Sin hit + fallback de pariente | Muestreo por densidad (50/20/20/10) | **5 TIPOS de ADN real con metadata** |
| **Analiza falla por tipo** | No | No | **Sí (metadata por categoría)** |

**Mis 2 diferenciadores**:
1. **Backbone transformer** vs CNN de los otros dos.
2. **Dataset de negativas en 5 categorías** con metadata para análisis de falla.

---

## 5. Pipeline — 7 pasos

### Paso 0 · Piloto obligatorio (antes de escalar a todas las especies)

**Qué**: antes de descargar y procesar en masa, correr RepeatMasker + PanTEon
sobre **3 genomas pequeños, uno por reino**, y comparar contra una anotación
de referencia conocida (Dfam, rmsk, EDTA).

**Pilotos sugeridos**:
- 🌱 Planta: *Arabidopsis thaliana* (~120 Mb · anotación TAIR10 disponible)
- 🍄 Hongo:  *Neurospora crassa* (~40 Mb · ligero)
- 🐟 Animal: *Caenorhabditis elegans* (~100 Mb · anotación WormBase)

**Qué validar**:
- Que la salida de RepeatMasker tenga coordenadas sensatas (no ruido).
- Que el %TE total encontrado se parezca al reportado en literatura
  (ej. Arabidopsis ~15%, Neurospora ~10%, C. elegans ~17%).
- Que los principales órdenes coincidan con lo publicado.

**Por qué es obligatorio**: Simón lo dice textual —
*"Piloto obligatorio: 3 genomas pequeños (uno por reino) comparados contra
rmsk/Dfam precomputado antes de descargar en masa."* Sin esto, se puede
descubrir muy tarde que la biblioteca o los parámetros estaban mal.

---

### Paso 1 · Anotar cada genoma con RepeatMasker + biblioteca PanTEon

**Qué**: generar coordenadas (BED) de todos los TE anotados por especie.

**Por qué**: PanTEon es una biblioteca de consensos, no un conjunto de coordenadas.
Sin este paso, nadie en el equipo tiene dataset real.

**Entregable**: `annotations/{species}.bed` con columnas
`cromosoma / inicio / fin / orden / superfamilia / %divergencia`.

**Reglas de limpieza**:
- Clase Unknown → se excluye del etiquetado (ni positivo ni negativo).
- Satélite y baja complejidad → se excluyen.
- Buffer de ≥1 kb alrededor de cada hit → ventanas que caen en el buffer se descartan.

**Biblioteca a usar**:
- Si la especie tiene **≥30 consensos** en PanTEon → usar biblioteca de la especie.
- Si tiene **<30 consensos** → usar biblioteca del **reino completo**.
- Dato duro: según mi análisis, **1.667 de 2.790 especies (59.7%)** tienen <30
  consensos y van a necesitar fallback de reino.

---

### Paso 2 · Selección de especies (acuerdo con el equipo)

- Estratificado por **reino**: Animales, Plantas, Hongos.
- Estratificado por **%TE del genoma**: <10%, 10-50%, >50%.
- Partición **80/10/10** por especie (nunca por ventana).
- **Mismo conjunto de test** para los 3 miembros del equipo.
- Fuentes: NCBI (GCF > GCA), Phytozome, Ensembl.
- Filtros duros: al menos una secuencia ≥50 kb, N50 ≥100 kb idealmente.

---

### Paso 3 · Generar ventanas de 50 kb

- **Train**: ventanas sin solape.
- **Dev/Test**: teselado completo con solape 50% (paso 25 kb) para inferencia.
- Descartar scaffolds < 25 kb.
- Padding con N si el scaffold es menor.

**Regla de etiquetado (base común)**:
Ventana positiva si existe un TE con solape ≥ max(50 bp, 20% del elemento).

---

### Paso 4 · Chunking para el modelo (fijo de mi arquitectura)

Para cada ventana de 50 kb:
- Dividir en **10 chunks contiguos de 5 kb**.
- Tokenizar cada chunk con NT (k-mer=6) → ~1000 tokens por chunk.
- **No se etiqueta por token** — la etiqueta es ÚNICA por ventana (0 ó 1).

```
input_ids:      (B, 10, 1000)    → por ventana
attention_mask: (B, 10, 1000)
label:          (B,)              → por ventana (binario)
```

---

### Paso 5 · Las 5 categorías de ventana negativa [MI DIFERENCIADOR]

Las ventanas negativas (sin TE) se muestrean de **genoma real** con cuotas fijas:

| Categoría | % del total | Qué contiene |
|---|---:|---|
| Intergénico puro | 30% | ADN entre genes, sin hits de ningún TE |
| Dentro de genes | 20% | Región génica codificante o UTR |
| Otros TE (no-LTR) | 20% | SINE, LINE, DNA-transposons |
| Heterocromatina | 15% | Centrómero, pericentromérico |
| Repeticiones simples / teloméricas | 15% | ADN repetitivo no-TE |

**Clave**: todas colapsan a `label = 0` (son negativas). Pero el **tipo se guarda
como metadata** (ampliado según sugerencia de Simón):

```json
{
  "window_id": "ath_chr3_01245",
  "species": "Arabidopsis_thaliana",
  "kingdom": "plant",
  "chromosome": "chr3",
  "start": 1245000,
  "end": 1295000,
  "label": 0,
  "negative_type": "intergenic",
  "has_other_TE": false,

  // Para ventanas positivas (label=1), extras exigidos por Simón:
  "n_TE": 2,
  "coverage_fraction": 0.18,
  "orders_present": ["LTR", "LINE"],
  "te_coords": [[15000, 22000], [34000, 38500]]
}
```

**Por qué importa**: al final del entrenamiento genero una tabla
"errores por tipo de negativo" → identifica dónde falla el modelo
(ej. "confunde heterocromatina con TE 35% de las veces").

Esta ventaja analítica **no la tienen Jhonier ni Alejandro**.

---

### Paso 6 · Entrenamiento con mi arquitectura MIL

**Arquitectura** (código actual, sin cambios):
- `NTBinaryClassifier` en [binary_model.py](Inpactor3_segnt/src/inpactor3_segnt/binary_model.py)
- NT-50M congelado + mean-pool intra-chunk + max-pool entre chunks + cabeza.

**Pérdida**:
```python
# labels: (B,) — 0 ó 1 por ventana
# logits: (B,)
loss = BCEWithLogitsLoss(pos_weight=pw)(logits, labels_float)
```

**Hiperparámetros (del protocolo común)**:
- Optimizador: SGD con decaimiento.
- Lote: 64 ventanas.
- Semillas: 42, 1, 7 (tres corridas, reportar media ± DE).
- Umbral de decisión: calibrado en dev, aplicado 1 vez en test.
- Early stopping: patience=15 sobre dev_f1.

**Nota**: el entrenamiento previo que corrió 34 épocas usó AdamW (del script original).
Para la comparación con el equipo, se re-corre con SGD como pide el protocolo común.

---

### Paso 7 · Inferencia y métricas

#### Pipeline de inferencia

```
Genoma de test  →  teselado 50 kb solape 50% (paso 25 kb)
                 →  Modelo emite probabilidad POR VENTANA
                 →  Agregar probs en zonas solapadas (promedio)
                 →  Pred = (prob > umbral)
                 →  Comparar contra BED de ground truth
```

#### Métricas obligatorias (base común)

- **F1 por ventana** → comparable con Jhonier y Alejandro.
- **Recall por ventana**.
- **FP/Mb** (falsos positivos por megabase) → honesta a prevalencia natural.
- Desglose por **reino** (Animales / Plantas / Hongos).
- Desglose por **especie**.

#### Métrica adicional mía (ventaja analítica)

- **F1 por tipo de negativo** → identifica dónde falla.

---

## 6. Diferencias con las otras 2 estrategias

Las 3 estrategias terminan generando **ventanas de 50 kb con etiqueta binaria**.
Lo que varía es la construcción del dataset y la arquitectura:

| Dimensión | Jhonier | Alejandro | **Cristian (yo)** |
|---|---|---|---|
| Granularidad final | Ventana (binaria) | Ventana (colapsada desde base) | **Ventana (binaria)** |
| Backbone | CNN | CNN | **Transformer (NT-50M congelado)** |
| Agregación | Convolución global | Downsampling | **MIL max-pool entre chunks** |
| Positivos | Ventanas con TE anotado | Ventanas con TE anotado | Ventanas con TE anotado |
| Negativos | Sin hit + fallback | Muestreo por densidad | **5 TIPOS de ADN real con metadata** |
| Fuente ADN | PanTEon + pariente | Genomas ensamblados | **Genomas ensamblados (NCBI/Phytozome)** |
| Análisis de falla por tipo | No | No | **Sí (metadata)** |

---

## 7. Adaptación al protocolo común

El equipo definió que la ÚNICA variable debe ser el dataset:

| Elemento común | Mi cumplimiento |
|---|---|
| Formato uint8 (N, 5, 50000, 1) | Convertir ACGTN → uint8 one-hot al cargar |
| Mismo modelo de comparación: Inpactor 2 Detect | **Para la tabla comparativa final uso Inpactor 2 Detect sobre mi dataset** (no mi SegNT) |
| Hiperparámetros SGD + BCE + lote 64 | ✅ |
| Semillas 42, 1, 7 | ✅ |
| Mismas especies de test | ✅ (consensuado con el equipo) |
| Umbral calibrado en val, test medido 1 vez | ✅ |

**Nota importante**: la comparación "¿qué dataset es mejor?" se mide con **Inpactor 2
Detect sobre los 3 datasets** (el de Jhonier, el de Alejandro, el mío). Mi
arquitectura SegNT es un **extra**: muestra qué pasa con una red más potente
sobre mi dataset.

---

## 8. Resultados esperados

Si la estrategia funciona:

| Métrica | Dataset viejo (PanTEon limpio) | Dataset nuevo (esperado) |
|---|---:|---:|
| F1 test balanceado | 0.67 | 0.60 - 0.70 |
| F1 Arabidopsis chr5 real | **0.08** | **0.45 - 0.55** |
| Gap test → genoma real | 0.59 (crítico) | <0.20 (aceptable) |

**No el objetivo**: llegar a F1=0.90.
**El objetivo**: cerrar el **gap** entre test y genoma real.

---

## 9. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| RepeatMasker + PanTEon no genera suficientes hits en especies con <30 consensos | Usar biblioteca del reino completo como fallback (acuerdo del equipo) |
| Encoder congelado limita el techo (F1=0.67 en el run previo) | Discutir con el profe si se descongela el encoder (requiere más GPU) |
| MIL max-pool puede ser engañado por falsos positivos locales | Las 5 categorías de negativas atacan esto: el modelo verá tipos de "no-TE" variados |
| Clases desbalanceadas en genoma real | `pos_weight` en BCE para compensar |
| Fuga por homología entre especies cercanas | Agrupar por género en el split, no por especie suelta |

---

## 10. Cronograma estimado

| Tarea | Duración | Dependencia |
|---|---|---|
| 1. Anotación RepeatMasker (todas las especies) | 3-5 días | Compartida con el equipo |
| 2. Selección de especies + descarga | 2 días | Acuerdo del equipo |
| 3. Muestreo de las 5 categorías de negativas | 2 días | Pasos 1 y 2 |
| 4. Reentrenamiento SegNT (3 semillas, SGD) | 3-4 días GPU | V100 32GB |
| 5. Inferencia sobre genomas de test | 1 día GPU | Modelo entrenado |
| 6. Comparación cruzada con los 3 datasets | 1 día | Los 3 modelos |
| **TOTAL** | **~2-3 semanas** | |

---

## 11. Frase para la defensa

> *"Mi arquitectura es un clasificador binario a nivel de ventana de 50 kb,
> usando el Nucleotide Transformer como extractor congelado con Multiple
> Instance Learning sobre 10 chunks. La corrí 34 épocas en V100 y llegó a
> F1=0.67 en test balanceado — convergente, pero cae a 0.08 en Arabidopsis
> real por domain shift.*
>
> *Mi diferenciador frente al equipo NO es la granularidad de etiqueta
> (los tres entregamos binario por ventana), sino DOS cosas: (1) uso un
> backbone transformer donde Jhonier y Alejandro usan CNN, y (2) mis
> ventanas negativas no son fondo neutro — son una mezcla curada de
> 5 tipos de ADN real (intergénico, génico, otros TE, heterocromatina,
> repeticiones) con metadata explícita, lo que me permite analizar
> exactamente en qué tipo de negativo falla el modelo.*
>
> *Respeto el protocolo común del equipo: RepeatMasker + PanTEon, split
> por especie, hiperparámetros SGD-BCE-lote 64, semillas 42/1/7, test
> con prevalencia natural. La única variable es el dataset — que es
> lo que queremos comparar."*
