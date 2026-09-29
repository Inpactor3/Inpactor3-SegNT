# Guion — explicando el proyecto desde cero

Un texto para leer despacio, sin saltarse pasos. Está escrito como si te
lo estuviera contando en voz alta. No usa jerga sin explicarla.

---

## Parte 1 · El problema biológico

### Qué es un genoma

Un **genoma** es todo el ADN de un organismo. En una planta como *Arabidopsis
thaliana* (una hierba que se usa como modelo en laboratorios de todo el mundo),
el genoma son 119 millones de letras: A, C, G y T, una tras otra, sin espacios.

Podríamos imaginar el genoma como un libro gigantesco escrito con solo cuatro
letras. Ese libro está dividido en 5 capítulos, que llamamos **cromosomas**.

En Arabidopsis:
- Cromosoma 1: 30 millones de letras
- Cromosoma 2: 20 millones
- Cromosoma 3: 23 millones
- Cromosoma 4: 19 millones
- Cromosoma 5: 27 millones

### Qué es un LTR-retrotransposón

Dentro de ese libro hay **pedazos que se copian solos**. Se los llama
**elementos transponibles**. Uno de los tipos más importantes en plantas se
llama **LTR-retrotransposón** (LTR-RT).

Un LTR-RT es una secuencia de ADN de entre 1 000 y 15 000 letras que tiene
una estructura muy particular:

```
──[ LTR izquierdo ]──[ región interna ]──[ LTR derecho ]──
     ~500 letras         ~5 000 letras       ~500 letras
```

Los dos "LTR" (repeticiones terminales largas) son **iguales entre sí** en
el momento en que el elemento se copia. Entre ellos está el "motor" que le
permite copiarse.

En una planta como maíz, **el 75% del ADN** son LTR-RTs. Ocupan tres cuartas
partes del genoma. En Arabidopsis ocupan alrededor del 10%.

### Por qué importan

Anotar dónde están los LTR-RTs en un genoma es crítico para biólogos y
mejoradores agrícolas. Si te equivocas en un 10% al detectarlos, en maíz
estás anotando mal el 7% de todo el ADN de la planta. Es mucho.

---

## Parte 2 · La herramienta que ya existe: Inpactor 2

Un investigador de la Universidad de Caldas, Simón Orozco, publicó una
herramienta llamada **Inpactor 2**. Recibe el genoma de una planta y
devuelve una tabla con las coordenadas de todos los LTR-RTs.

### Cómo lo hace

Inpactor 2 usa **tres redes neuronales encadenadas**, más un programa
externo:

1. **Detect** — una red neuronal recibe pedazos del genoma de 50 000 letras
   ("ventanas") y para cada pedazo dice si hay o no un LTR-RT ahí adentro.
2. **LTR_FINDER** — un programa clásico en C++ que busca las dos
   repeticiones LTR en los pedazos que la Detect marcó como positivos, y
   escribe las coordenadas exactas.
3. **Filter** — otra red que verifica si el elemento está completo (con
   sus dos LTR y su región interna).
4. **Classify** — la última red decide de qué familia es el elemento (ALE,
   TAT, ATHILA, etc. — hay unas 13 familias comunes).

Al final produce un archivo `.tab` con 8 columnas:
```
cromosoma   inicio    fin     largo   linaje         det   filt  class
Chr1        3780764   3785720 4956    RLC/IVANA      0.99  0.99  0.99
Chr1        7717355   7722547 5192    RLC/IVANA      0.99  0.99  0.99
...
```

**En Arabidopsis, Inpactor 2 encontró 109 elementos** repartidos así:
Chr1=17, Chr2=24, Chr3=24, Chr4=21, Chr5=23.

### Lo que asumimos

Para nuestro proyecto tomamos esos **109 elementos como si fueran la
verdad absoluta**. En terminología de ciencia de datos, esto se llama
"ground truth" (verdad de referencia). Vamos a tratar de que nuestro
modelo dé exactamente los mismos 109.

---

## Parte 3 · Qué queríamos mejorar

Inpactor 2 funciona pero tiene tres problemas:
1. **Tres redes en cascada**: si la primera se equivoca, arrastra el
   error a las siguientes.
2. **Depende de un programa externo** (LTR_FINDER) que hay que compilar
   en C++ e instalar aparte.
3. **Aprende desde cero**: cada red se entrena solo con datos de plantas,
   sin aprovechar conocimiento previo del ADN en general.

**Nuestra idea**: reemplazar todo eso con **una sola red neuronal** que
haga todo a la vez, y que **arranque con conocimiento pre-existente**.

---

## Parte 4 · La arquitectura que propusimos: SegNT

### El nombre

**Seg** viene de "segmentación por token", y **NT** viene de "Nucleotide
Transformer" — el nombre del modelo pre-entrenado que usamos como base.

### Nucleotide Transformer

Es un modelo publicado por una empresa francesa llamada **InstaDeep** en 2023.
Ellos lo entrenaron sobre **el ADN de 850 especies distintas** (plantas,
hongos, humanos, insectos). El modelo aprendió patrones básicos: cómo se
comportan las bases en un exón, en un intrón, en zonas repetitivas, etc.

Piénsalo como un **niño que aprendió a leer español**. Ya sabe cómo funcionan
las palabras y las frases en general. Solo le falta aprender vocabulario
específico de un dominio.

Nosotros tomamos este modelo y le enseñamos una tarea nueva: **decir cuáles
letras del ADN son parte de un LTR-retrotransposón**.

### Los dos pedazos de la arquitectura

Nuestra red tiene dos pedazos:

**El cuerpo (Transformer)**: 40 millones de parámetros, ya entrenados por
InstaDeep. Nosotros lo dejamos casi como estaba, solo lo ajustamos un poco.
Su trabajo es convertir el ADN en "significado".

**La cabeza (una CNN pequeña)**: 200 000 parámetros que empezamos desde cero.
Su trabajo es tomar el "significado" que produce el cuerpo y clasificar
cada pedacito como: fondo (no LTR-RT), o alguna de 13 familias.

```
ADN → [Transformer grande, congelado casi] → significado por token → [CNN chiquita] → etiqueta
```

---

## Parte 5 · Las ventanas — el truco de trocear

Un cromosoma tiene 30 millones de letras. Ningún modelo neural puede leer
30 millones de letras de golpe (ni la GPU aguanta, ni tiene sentido).

La solución es **cortar el cromosoma en ventanas**.

### Dos tipos de ventanas

**Ventanas de 50 000 letras** (50 kb): son las que usa Inpactor 2 para
tomar decisiones grandes. Cada ventana se marca como "tiene LTR-RT" o "no
tiene". Un cromosoma de 30 millones de letras se divide en 600 ventanas de
50 kb.

**Ventanas de 6 000 letras** (6 kb): son las que nuestro modelo lee cada
vez. El Transformer solo aguanta ~12 kb como máximo, así que usamos 6 kb
para tener margen.

Para el examen final, medimos F1 sobre ventanas de 50 kb (para poder
comparar con Inpactor 2). Pero durante el entrenamiento, el modelo lee de
6 kb en 6 kb.

### Qué es un token

Dentro de una ventana de 6 000 letras, agrupamos las bases en "tokens".
Cada token es un pedazo de 6 letras. Por ejemplo:

```
ADN:   ACGTATGCATAT...
Tokens: [ACGTAT][GCATAT]...
Total: 1 000 tokens por ventana
```

Este agrupamiento se llama **BPE** (Byte-Pair Encoding) y lo hace un
"tokenizer" que viene junto con el Transformer.

### Cómo la red mira una ventana

Aquí viene una cosa importante que quizás no se entiende del todo:

**El Transformer NO lee de izquierda a derecha ni del centro hacia los
lados.** Todos los 1 000 tokens de la ventana se miran entre sí **al mismo
tiempo**. Cada token puede prestar atención a cualquier otro. Esto se
llama **atención** y es lo que hace que los Transformers sean tan potentes.

Esto es útil porque un LTR-RT tiene dos LTRs que pueden estar separados
por miles de bases. Con atención, el modelo puede conectar el LTR izquierdo
con el derecho aunque haya mucha secuencia intermedia.

---

## Parte 6 · Cómo dividimos los datos

**Ley básica del aprendizaje automático**: nunca evalúes un modelo con los
mismos datos con los que lo entrenaste. Si lo haces, el modelo va a
"engañarte" — parece que sabe pero solo memorizó.

Dividimos los 5 cromosomas así:

| Cromosoma | Papel | Anotaciones |
|---|---|---|
| Chr1 | Entrenar | 17 |
| Chr2 | Entrenar | 24 |
| Chr3 | Entrenar | 24 |
| Chr4 | Validar | 21 |
| Chr5 | **Examen** | 23 |

**Entrenar** (65 anotaciones total): la red ve estos cromosomas millones
de veces y ajusta sus parámetros.

**Validar** (21 anotaciones): al terminar cada época, medimos F1 aquí para
decidir si el modelo mejoró. No ajustamos parámetros con esto — solo
decidimos cuándo parar.

**Examen** (23 anotaciones): esto solo se usa **UNA VEZ**, al final de todo.
Es la prueba honesta.

---

## Parte 7 · Cómo aprende la red

Todo el aprendizaje se basa en un proceso llamado **descenso del gradiente**.
Suena complicado pero la idea es simple: el modelo hace una predicción, ve
qué tan lejos está de la verdad, y ajusta sus parámetros un pelito para la
próxima vez estar un poquito más cerca.

### El problema del desbalance

De cada 1 000 tokens de una ventana, solo unos 10-20 son LTR-RT. El resto
es fondo (98%).

Si dejáramos al modelo aprender libremente, encontraría un atajo obvio:
**decir "fondo" a todo**. Con eso tendría 98% de aciertos. Pero sería
totalmente inútil.

Para evitar ese atajo hicimos dos cosas:

**1. Peso de la clase positiva (`pos_weight=100`)**. En la función de
pérdida, si el modelo dice "fondo" cuando la verdad era "LTR-RT", esa
equivocación pesa 100 veces más que la contraria. Así lo obligamos a
prestar atención a los positivos.

**2. Muestreo balanceado (`WeightedRandomSampler`)**. En vez de mostrarle
al modelo ventanas aleatorias del genoma (donde 99% serían negativas), le
forzamos que **el 50% de las ventanas de cada batch tengan al menos un
LTR-RT**. Esto se logra "sobre-muestreando" las pocas ventanas positivas
que tenemos.

### Qué es una época

Una **época** es cuando el modelo ve todos los datos de entrenamiento una
vez. Un batch de 8 ventanas, luego otro de 8, luego otro... hasta pasar
por todo. Entrenamos 15 épocas.

En cada época, la loss (el error) debería bajar. Y el F1 debería subir.

---

## Parte 8 · Qué pasó de verdad en las 15 épocas

Aquí es donde se pone honesto y hay que contar la realidad:

| Época | Loss | F1 val | Recall | Precisión |
|---|---|---|---|---|
| 01 | 2.12 | 0.03 | 1.00 | 0.01 |
| 04 | 0.10 | 0.07 | 0.67 | 0.04 |
| 06 | 0.02 | 0.10 | 0.24 | 0.06 |
| 09 | 0.005 | 0.10 | 0.11 | 0.10 |
| 11 | 0.003 | 0.14 | 0.22 | 0.10 |
| **15** | **0.0005** | **0.14** | 0.11 | 0.19 |

Cómo leer estos números:

- **Loss** empezó en 2.12 y terminó en 0.0005 — cayó 4 000 veces. Suena
  bueno, pero significa que el modelo **memorizó** los 65 ejemplos.
- **F1 val** empezó en 0.03 (recall=1.00 pero precisión 0.01 — "digo sí a
  todo") y terminó en 0.14 (recall 0.11, precisión 0.19 — "digo sí muy
  poco pero cuando lo digo suele estar bien").
- El mejor F1 en validación fue 0.14 (14%).

### La curva

En las primeras épocas el modelo aprende rápido. Alrededor de la época 6
encuentra un buen balance. Luego se estanca — la loss sigue bajando (memoriza
más) pero el F1 no sube (no generaliza más).

---

## Parte 9 · El examen final

Con el mejor modelo guardado, lo aplicamos a **Chr5**, que nunca vio.

Chr5 tiene:
- 540 ventanas de 50 000 letras cada una
- 23 elementos verdaderos (según Inpactor 2)
- Esos 23 elementos caen en 21 ventanas (algunas tienen 2 elementos)

**Nuestro modelo predijo 7 anotaciones** que caen en solo **4 ventanas**.

De esas 4 ventanas:
- 1 coincidió con una ventana verdadera de Inpactor 2 (**TP** — verdadero positivo)
- 3 fueron ventanas donde Inpactor 2 no marcó nada (**FP** — falso positivo)

Ventanas verdaderas que se nos escaparon: 20 (**FN** — falso negativo).

### Las métricas

```
Recall    = 1/21 = 0.048 (4.8%)   ← detectamos 5 de cada 100 verdaderos
Precisión = 1/4  = 0.250 (25%)    ← 1 de cada 4 nuestros es correcto
F1        = 0.080 (8%)            ← promedio armónico de recall y precisión
```

También medimos si acertamos las coordenadas exactas: **IoU = 0**. Ninguna
de nuestras cajas se solapa más del 50% con las cajas verdaderas. Las
posiciones están mal.

Y medimos si acertamos el linaje (ALE, TAT, etc.): **0 aciertos de linaje**.

---

## Parte 10 · Qué dice esto — lectura honesta

### Comparación con lo que existía antes

Un informe interno del laboratorio de Simón intentó hacer algo parecido con
una CNN estilo YORO desde cero. Sus mejores números sobre Chr5:

| Enfoque | F1 sobre Chr5 |
|---|---|
| Fondo aleatorio uniforme | 0-7% |
| Fondo otro transposón | 7.5% |
| **Fondo genoma real (baseline)** | **26.3%** |
| **Nuestro SegNT** | **8.0%** |

Estamos por debajo del baseline por 18 puntos. Superamos los intentos
malos, pero no el bueno.

### Por qué no ganamos

**El cuello de botella es la cantidad de datos**. Un modelo de 40 millones
de parámetros necesita miles o decenas de miles de ejemplos positivos para
generalizar bien. Nosotros le dimos 65.

**El modelo se sobre-ajustó**: aprendió los 65 ejemplos perfectamente
(loss=0.0005) pero no aprendió el patrón general de "cómo se ve un LTR-RT
en Arabidopsis".

**Chr4 no es igual a Chr5**: los cromosomas tienen mezclas distintas de
familias de LTR-RTs. El modelo aprendió las de Chr4 y falla en las de Chr5.

### Qué SÍ demostramos

Aunque el número final no es bueno, el trabajo dejó tres cosas:

**1. Un pipeline completo y reproducible**. Cualquiera puede clonar el
repositorio, abrir el notebook en Google Colab, y en 1 hora tener el mismo
resultado. Con las mismas 109 anotaciones de Inpactor 2 y el mismo split.

**2. Una comparación justa con el informe del laboratorio**. Usamos el
mismo genoma, el mismo split (Chr1-3 train, Chr4 val, Chr5 test), y las
mismas métricas (F1 por ventana, IoU, matriz de confusión). Los números
son directamente comparables.

**3. Un diagnóstico claro**. Sabemos exactamente por qué no funcionó y
qué habría que hacer diferente. La arquitectura tiene sentido; falta
material.

### Qué haría falta hacer distinto

- **Más datos**: incluir arroz (~450 anotaciones), sorgo (~1 000), maíz
  (~15 000). Con miles de anotaciones el modelo tendría material real.
- **Congelar el cuerpo**: entrenar solo la cabeza pequeña. Eso baja los
  parámetros de 40 M a 200 K y reduce el sobre-ajuste.
- **Data augmentation**: usar el complemento reverso de cada secuencia
  (biológicamente equivalente) para duplicar el corpus gratis.
- **Grid search**: probar varias configuraciones de learning rate,
  pos_weight, umbral, etc.

---

## Parte 11 · Resumen en cinco frases

1. Cortamos el genoma de Arabidopsis en ventanas de 6 000 letras.
2. Un Transformer pre-entrenado en 850 especies convierte cada ventana en
   embeddings de significado, y una cabeza pequeña clasifica cada token
   como fondo o como una de 13 familias de LTR-RT.
3. Comparamos contra Inpactor 2 (109 anotaciones sobre TAIR10) usando el
   mismo split por cromosoma que el informe interno.
4. Alcanzamos F1 = 8% sobre Chr5, por debajo del baseline del informe
   (26.3%), porque 65 anotaciones de entrenamiento son insuficientes para
   40 millones de parámetros.
5. La contribución del trabajo no es un número mejor sino un pipeline
   reproducible y el diagnóstico claro de que el cuello de botella son
   los datos, no la arquitectura.

---

## Parte 12 · Glosario mínimo

- **ADN**: cadena de bases A, C, G, T que codifica la información
  genética.
- **Genoma**: todo el ADN de un organismo.
- **Cromosoma**: un pedazo grande de ADN — el genoma se divide en varios.
- **LTR-retrotransposón (LTR-RT)**: un tipo de elemento móvil de ADN con
  dos repeticiones terminales.
- **Ventana**: un pedazo contiguo del genoma de tamaño fijo (nosotros
  usamos 50 kb para métricas y 6 kb para input).
- **Token**: la unidad mínima que ve el modelo. En nuestro caso, un
  6-mero (6 bases).
- **Transformer**: tipo de red neuronal que usa atención — cada token
  puede "mirar" a cualquier otro.
- **Embedding**: un vector de números que representa el significado de
  algo. Cada token se convierte en un embedding de 512 números.
- **Batch**: un grupo de ejemplos que el modelo procesa a la vez.
- **Época**: una pasada completa por todos los datos de entrenamiento.
- **Loss**: qué tan mal se equivocó el modelo en una predicción.
- **Precision**: de las cosas que dijiste que sí, cuántas son sí.
- **Recall**: de las cosas que en verdad son sí, cuántas dijiste que sí.
- **F1**: promedio armónico de precision y recall.
- **Ground truth**: la verdad de referencia con la que comparas tus
  predicciones. En nuestro caso, la salida de Inpactor 2.
- **Overfitting (sobre-ajuste)**: cuando el modelo memoriza train pero
  falla en datos nuevos.
