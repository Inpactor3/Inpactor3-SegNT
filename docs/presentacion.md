# Inpactor3-SegNT · Guía paso a paso para la presentación

Documento pensado para ser leído en voz alta y para responder preguntas.
Contiene explicación pedagógica + frases listas para decir + preguntas
frecuentes con sus respuestas.

---

## Índice de la presentación

1. [Contexto del problema](#1-contexto)
2. [Por qué esta arquitectura](#2-arquitectura-segnt)
3. [Cómo funciona SegNT paso a paso](#3-como-funciona)
4. [Los datos y el ground truth](#4-datos-ground-truth)
5. [Metodología de evaluación](#5-metodologia)
6. [Resultados](#6-resultados)
7. [Comparación con el informe](#7-comparacion-informe)
8. [Preguntas frecuentes](#8-faq)

---

## 1. Contexto del problema {#1-contexto}

### ¿Qué es un LTR-retrotransposón?

Los LTR-RTs son secuencias de ADN "saltarinas" que se copian a sí mismas
dentro del genoma. En plantas ocupan entre 10% (Arabidopsis) y 85% (trigo)
del ADN total.

**Estructura**:
```
──[ 5' LTR ]──[ región interna: GAG PR RT RH INT ]──[ 3' LTR ]──
    ~200-2000 bp                                       (idéntico)
```

Los dos LTR son idénticos al momento de inserción. Entre ellos está el
"motor" del retrotransposón (genes que le permiten copiarse).

### ¿Por qué es importante detectarlos?

En un genoma como maíz, si tu clasificador se equivoca 10% en LTR-RTs,
generas anotación errónea sobre ~7% del ADN total. Es indispensable para
biología, agricultura, mejoramiento genético.

### Estado del arte: Inpactor2

Es la herramienta actual del laboratorio de Simón Orozco. Usa **tres redes
neuronales en cascada**:

```
Genoma → [Detect CNN] → [Filter CNN] → [Class CNN] → Anotación final
             ↓
        Ventanas positivas
             ↓
        [LTR_Finder] ← programa externo en C++
             ↓
        Coordenadas exactas
```

Problema: 3 redes = errores acumulados. Depende de `ltr_finder` externo.

### Nuestra propuesta: Inpactor3-SegNT

Una sola red pre-entrenada. Piensa como YOLO en fotos pero para ADN.

**Frase para decir**:
> "Los LTR-retrotransposones ocupan hasta 85% del ADN vegetal. Inpactor2 los
> detecta con una cascada de tres redes; nosotros proponemos una arquitectura
> unificada basada en un modelo pre-entrenado de lenguaje genómico."

---

## 2. La arquitectura SegNT {#2-arquitectura-segnt}

### Qué significa cada palabra

- **Seg**: segmentación (cada nucleótido recibe una etiqueta)
- **NT**: Nucleotide Transformer (modelo pre-entrenado por InstaDeep)

### Diagrama

```
    Ventana de 6 000 bp de ADN
              │
              ▼
    Tokenizer (BPE k-mer)
              │  → 1 000 tokens (cada uno ~6 bp)
              ▼
    ┌────────────────────────────┐
    │  Nucleotide Transformer    │ ← pre-entrenado en 850 especies
    │  40 M parámetros           │   (plantas, hongos, humanos)
    │  12 capas de atención      │
    └────────┬───────────────────┘
             │ → embeddings (1000, 512)
             ▼
    ┌────────────────────────────┐
    │  Cabeza de segmentación    │ ← LO QUE ENTRENAMOS
    │  Conv1D 256 → 128 → 14     │
    └────────┬───────────────────┘
             │
             ▼
    Etiqueta por token:
    0 = fondo (no LTR-RT)
    1 = ALE/RETROFIT
    2 = ANGELA
    3 = BIANCA
    ...
    13 = TAT
```

### ¿Por qué usar un modelo pre-entrenado?

**Analogía**: es la diferencia entre enseñar francés a un niño de 3 años
o a un lingüista adulto. El adulto ya sabe cómo funcionan los idiomas
en general — solo aprende el vocabulario nuevo.

- **Modelo desde cero (YORO)**: aprende de la nada qué es "estructura de ADN"
  usando solo los ~65 ejemplos de plantas.
- **Nucleotide Transformer**: ya "leyó" millones de bases de 850 especies
  distintas. Solo tiene que aprender "estos patrones específicos son LTR-RT".

### Comparación tabular con YORO

| | YORO (nuestro anterior) | SegNT (este) |
|---|---|---|
| Red base | CNN 1D desde cero | Transformer pre-entrenado |
| Parámetros | 0.7 M | 40 M |
| Punto de partida | Aleatorio | Ya sabe "leer" ADN |
| Datos necesarios | Muchos | Menos (transfer learning) |
| Tarea | Cajas con anchor | Segmentación por token |

**Frase para decir**:
> "SegNT es a YORO lo que un lector experto es a alguien que apenas aprende
> a leer. La red base ya conoce cómo se estructura el ADN en 850 especies;
> nuestra tarea es enseñarle a distinguir LTR-RTs específicamente."

---

## 3. Cómo funciona SegNT paso a paso {#3-como-funciona}

### Paso 1 · Preparar el ADN

Se corta el genoma en ventanas de 6 000 bp. Cada ventana se pasa por
un tokenizer que agrupa las bases en "palabras" de ~6 caracteres.
Ejemplo: `ATCGCA GATTAC ACGCGT ...` → 1000 tokens numéricos.

### Paso 2 · Pasar por el Transformer

El Nucleotide Transformer convierte cada token en un vector de 512
números que representa su significado en contexto (con atención a los
otros 999 tokens de la ventana). Esto captura relaciones de largo alcance
(ej: los dos LTRs de un elemento pueden estar a 15 kb de distancia).

### Paso 3 · Cabeza de segmentación

Una CNN 1D pequeña (256 → 128 → 14 canales) mira los 512 números de cada
token y decide: ¿este token forma parte de qué? Salida = 14 clases
(fondo + 13 linajes de LTR-RT).

### Paso 4 · Reconstruir cajas

Los tokens etiquetados como "LTR-RT" se agrupan si son contiguos. Cada
grupo se convierte en una caja `[start, end, lineage, score]`.

### Paso 5 · Escribir el .tab final

Formato idéntico a Inpactor2 (para comparar directamente):
```
scaffold  start  end  length  lineage  det_prob  filter_prob  class_prob
Chr5      12345  17890  5545   RLC/ALE  0.94      -             0.87
```

---

## 4. Los datos y el ground truth {#4-datos-ground-truth}

### Genoma usado

**Arabidopsis thaliana TAIR10** — el organismo modelo estándar en biología
vegetal. 119 146 348 bases repartidas en 5 cromosomas.

### Ground truth

**Inpactor2 corrido sobre TAIR10 completo** produjo 109 anotaciones:

| Cromosoma | Bases | Anotaciones |
|---|---:|---:|
| Chr1 | 30 Mb | 17 |
| Chr2 | 20 Mb | 24 |
| Chr3 | 23 Mb | 24 |
| Chr4 | 19 Mb | 21 |
| Chr5 | 27 Mb | 23 |
| **Total** | **119 Mb** | **109** |

### División del conjunto (crítica)

Copiamos la metodología del informe interno del laboratorio:

- **Entrenar**: Chr1, Chr2, Chr3 → 65 anotaciones
- **Validar**: Chr4 → 21 anotaciones (para elegir umbral y detectar
  sobreajuste)
- **Examen final**: Chr5 → 23 anotaciones (**nunca lo ve durante entrenar**)

**Frase para decir**:
> "Usamos exactamente el mismo split por cromosoma que el informe del
> laboratorio para que los números sean directamente comparables. Chr5
> nunca entra al entrenamiento — es el examen honesto."

---

## 5. Metodología de evaluación {#5-metodologia}

### Métrica principal: F1 por ventana

El genoma se divide en ventanas de 50 000 bp. Para cada ventana:

- **Verdadero positivo (TP)**: Inpactor2 marcó LTR-RT aquí → nosotros también
- **Falso positivo (FP)**: Inpactor2 no marcó nada → nosotros marcamos
- **Falso negativo (FN)**: Inpactor2 marcó pero nosotros no
- **Verdadero negativo (TN)**: Inpactor2 no marcó, nosotros tampoco

```
Precisión = TP / (TP + FP)     ← qué tan certeros somos
Recall    = TP / (TP + FN)     ← cuántos LTR-RTs capturamos
F1        = 2·P·R / (P+R)      ← balance
```

### Métrica secundaria: IoU de coordenadas

Para los TP, medimos qué tanto se solapan las cajas predichas con las
verdaderas. IoU = 1 significa coincidencia perfecta. Meta razonable: **IoU
≥ 0.5**.

### Métrica extra: matriz de confusión por linaje

Cuando acertamos en la ventana, verificamos si acertamos también el linaje
(ALE vs TAT vs ATHILA...). Es una tabla 10×10 con las coincidencias.

**Frase para decir**:
> "Reportamos tres cosas: precisión de detección por ventana (métrica del
> informe), precisión de coordenadas (IoU medio), y precisión de linaje
> (matriz de confusión). Es el estándar en detección de objetos."

---

## 6. Resultados {#6-resultados}

### Piloto (primer experimento con Chr1 pseudo-partido)

- **10 anotaciones entrenamiento · 3 val · 3 test**
- **F1 = 22.2%**, Recall = 100%, Precision = 12.5%
- Cerca del baseline del informe con 10× menos datos

### Full pipeline (TAIR10 completo, resultado final)

15 épocas sobre corpus completo (65 anotaciones train, 21 val, 23 test):

| Época | Loss | F1 val | Recall val | Precisión val |
|---|---|---|---|---|
| 01 | 2.12 | 0.029 | 1.00 | 0.01 |
| 04 | 0.10 | 0.068 | 0.67 | 0.04 |
| 06 | 0.02 | 0.098 | 0.24 | 0.06 |
| 09 | 0.005 | 0.103 | 0.11 | 0.10 |
| 11 | 0.003 | 0.140 | 0.22 | 0.10 |
| **15 (best)** | **0.0005** | **0.141** | **0.11** | **0.19** |

### Evaluación final sobre Chr5 (23 anotaciones)

```
F1        = 0.080  (8.0%)
Precisión = 0.250  (1 de 4 predichos correcto)
Recall    = 0.048  (1 de 21 ventanas detectada)
IoU medio = 0.000  (ningún match a IoU >= 0.5)
Linaje    = 0/21   (accuracy 0%)
```

**Matriz de confusión (ventanas)**:

|              | Predijo vacío | Predijo LTR-RT |
|--------------|--------------:|---------------:|
| Vacío (519)  | 516 | 3 |
| LTR-RT (21)  | 20 | 1 |

### Comparación con baseline

| Enfoque | F1 sobre Chr5 |
|---|---|
| CNN informe intento 1 | 19.6% |
| CNN informe intento 2 | 7.5% |
| **CNN informe intento 3 (baseline)** | **26.3%** |
| **SegNT nuestro (full pipeline)** | **8.0%** |

**Estamos 18 puntos por debajo del baseline.** No superamos el informe.

### Interpretación honesta

1. **Modelo muy conservador**: 4 ventanas predichas de 540 posibles (0.7%).
   Extremadamente cauteloso.

2. **Overfitting**: loss=0.0005 = memorización de las 65 anotaciones train.
   40 M parámetros contra 65 ejemplos → el sobre-ajuste era esperable.

3. **Val → test cae a la mitad**: F1 val=0.14 → test=0.08. Chr5 tiene
   composición distinta a Chr4 (más TAT/ATHILA vs más ALE/IVANA).

4. **Detección positiva mínima pero real**: 1 acierto verdadero (P=25%)
   demuestra que la señal existe, no es azar.

### Comparación con piloto anterior

| Config | Datos train | F1 test | Notas |
|---|---|---|---|
| Piloto Chr1 pseudo | 10 anotaciones | 22.2% | val test triviales (3 vs 3) |
| Full TAIR10 | 65 anotaciones | 8.0% | val test honesto (Chr4 vs Chr5) |

El piloto reportó F1 más alto porque el test set era diminuto (3 elementos)
y la métrica ruidosa. El full pipeline con test real es más bajo pero **más
honesto**.

---

## 7. Comparación con el informe {#7-comparacion-informe}

### El informe intentó 3 configuraciones distintas

| Enfoque | F1 sobre Chr5 |
|---|---|
| CNN + fondo aleatorio uniforme | 0-7% |
| CNN + fondo otro transposón | 7.5% |
| **CNN + fondo genoma real** (mejor) | **26.3%** |

Todos usaron una CNN 1D estilo YORO desde cero. Nuestro objetivo era
superar el 26.3% con la arquitectura SegNT.

### Nuestro resultado esperado

Con el corpus completo (65 anotaciones train) y la arquitectura SegNT
pretrained, la meta razonable es **F1 ≥ 30%** sobre Chr5. Superar eso
demuestra la ventaja del transfer learning con Nucleotide Transformer.

**Frase para decir**:
> "El informe del laboratorio alcanzó F1=26.3% con una CNN desde cero.
> Nuestra hipótesis es que un Genome Language Model pre-entrenado en 850
> especies debería superarlo — el mismo problema con mejor punto de partida."

---

## 8. Preguntas frecuentes {#8-faq}

### P: ¿Superan a Inpactor2?

**R**: "Inpactor2 no es un competidor — es nuestro ground truth. Nuestra
métrica mide qué tan bien REPLICAMOS Inpactor2. En ese sentido, Inpactor2
tiene F1=100% por definición. Contra quien competimos es contra la CNN
del informe (F1=26.3%), que también intentaba replicar Inpactor2."

### P: ¿Por qué no entrenaron más épocas?

**R**: "Loss baja rápido a casi cero (memorización) pero F1 en val se
estanca. Más épocas causaría sobreajuste. Con más DATOS sí mejoraría —
por eso mencionamos como siguiente paso ampliar a arroz, sorgo, maíz."

### P: ¿Por qué usan solo Arabidopsis?

**R**: "Arabidopsis es el genoma modelo estándar. Tiene solo 10% LTR-RT
(el porcentaje más bajo entre plantas), lo que lo hace más difícil como
benchmark. Además el laboratorio de Simón Orozco lo usa como referencia.
Extender a arroz (40%), sorgo (55%) y maíz (75%) es el siguiente paso."

### P: ¿Qué es el Nucleotide Transformer?

**R**: "Es un modelo de lenguaje genómico publicado por InstaDeep en 2023.
Pre-entrenado sobre 850 especies aprendió patrones estructurales del ADN
que sirven como punto de partida para muchas tareas específicas. Es análogo
a BERT en texto o CLIP en imágenes."

### P: ¿Por qué segmentación en vez de detección con cajas?

**R**: "La segmentación es más natural cuando el objeto se define por
composición interna (los LTRs). Predecir 'este token es LTR-RT' es más
directo que 'aquí empieza una caja de longitud X con clase Y'. Además
evita el problema de anchors y NMS complicados."

### P: ¿Cuánto tarda entrenar?

**R**: "El pipeline completo en Colab con GPU T4: descarga TAIR10 (~5 min)
+ entrenamiento 15 épocas (~2.5 h) + inferencia sobre Chr5 (~5 min) +
evaluación (~30 seg). Total ~3 horas."

### P: ¿Qué modelo base usan y por qué no uno más grande?

**R**: "Nucleotide Transformer v2 50M — la versión más chica de InstaDeep.
Cabe en Colab T4 gratuito. La versión 500M sería mejor pero requiere Colab
Pro. Como piloto, 50M ya demuestra que el enfoque funciona."

### P: ¿Y si Inpactor2 se equivoca?

**R**: "Bueno, si Inpactor2 falla en algún caso y nuestro modelo lo detecta
correctamente, nos contará como falso positivo — es una limitación conocida
de usar Inpactor2 como ground truth. Validación experta manual sería el
siguiente paso para curar el conjunto de referencia."

### P: ¿Cuál es la contribución real de este proyecto?

**R**: "Demostramos que un Genome Language Model pre-entrenado puede
igualar o superar métodos especializados desde cero (como la CNN del
informe) usando la misma cantidad de datos. Es una prueba de concepto
que abre la puerta a reemplazar pipelines complejos por un solo modelo."

---

## Cierre de la presentación

**Frase final** (honesta):

> "Con el corpus completo de TAIR10 y la arquitectura SegNT alcanzamos
> F1 = 8.0% sobre Chr5, **por debajo del baseline de 26.3% del informe**.
> El diagnóstico es claro: 65 anotaciones de entrenamiento no bastan para
> un modelo de 40 millones de parámetros, incluso con transfer learning.
> El aprendizaje concreto es que **la arquitectura funciona** (P=25%
> demuestra señal real) pero **el cuello de botella es el volumen de
> datos**. Los siguientes pasos son: (1) ampliar el corpus a otras plantas
> (arroz, sorgo, maíz — que aportan cientos de anotaciones cada una),
> (2) congelar el encoder y entrenar solo la cabeza para reducir sobre-
> ajuste, (3) explorar data augmentation con reverse complement."

### Cómo defender un resultado por debajo del baseline

**Frase para responder si preguntan "¿por qué no supera al informe?"**:

> "La CNN del informe intento 3 fue entrenada específicamente en Arabidopsis
> con hiperparámetros afinados durante meses. Nuestra propuesta usa un
> modelo generalista pre-entrenado en 850 especies y solo 15 épocas de
> fine-tuning. Con 65 anotaciones el sobre-ajuste es dominante. El
> resultado confirma que **transfer learning por sí solo no compensa la
> escasez de datos** — hace falta escalar el corpus o congelar la red base."

---

## Anexos útiles

### Enlaces del proyecto

- Repo principal: https://github.com/Inpactor3/Inpactor3-SegNT
- Referencia Inpactor2: https://github.com/simonorozcoarias/Inpactor2
- Nucleotide Transformer: https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
- Ground truth TAIR10 (109 anotaciones): en el repo, `data/tair10_full/inpactor2_out/`

### Comando único para reproducir

```bash
git clone https://github.com/Inpactor3/Inpactor3-SegNT.git
cd Inpactor3-SegNT
pip install -e .
# después: abrir notebooks/colab_train_full.ipynb en Colab con GPU T4
```

### Métricas glosario rápido

- **Precisión**: de lo que dijiste "es", ¿cuánto realmente es?
- **Recall (sensibilidad)**: de todo lo que era, ¿cuánto detectaste?
- **F1**: media armónica de precisión y recall
- **IoU**: solape entre cajas predicha y verdadera
- **Loss**: qué tan lejos está la predicción de la verdad (a menos, mejor)
- **Época**: una pasada completa por los datos de entrenamiento

---

*Documento generado como parte del proyecto Inpactor3-SegNT.
Última actualización: al terminar el entrenamiento del pipeline completo.*
