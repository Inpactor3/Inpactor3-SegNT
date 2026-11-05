# Informe final — Inpactor 2 Detect con entrada de 24 kb

Detección de LTR-retrotransposones en plantas, hongos y animales con un backbone
congelado y max pooling sobre bloques.

Fecha: 3 de noviembre de 2026

---

## 1 · Resumen

Se rediseñó la entrada del detector: en vez de un bloque de 6 kb por ventana de
50 kb, cada ventana aporta **cuatro bloques de 6 kb (24 kb)** cuyos embeddings se
combinan con **max pooling**. El backbone (NT-500M) permanece congelado y solo se
entrena un MLP pequeño.

Resultados principales:

| Reino | Especies | Balanced accuracy | PR-AUC | Recall |
|---|---|---|---|---|
| Hongos | 116 | **0,770** | 0,835 | 0,856 |
| Plantas | 66 | **0,735** | 0,980 | 0,778 |
| Animales | 69 | 0,621 | 0,757 | 0,517 |

En el test oro independiente (TAIR10), el mismo modelo mejora al pasar de 24 kb a
48 kb en inferencia: F1 de 0,514 a 0,604, sin reentrenar.

Dos hallazgos metodológicos, que valen tanto como los números:

1. **El F1 engaña con clases desbalanceadas.** En plantas, el 92 % del test es
   positivo; un clasificador trivial sacaría 0,956.
2. **Elegir la época por F1 puede arruinar un modelo que sí discrimina.** En
   plantas se guardó la época con mejor F1 (0,975 en validación) y daba 0,556 de
   balanced accuracy en test. Al elegir por balanced accuracy, el mismo
   entrenamiento sube a 0,707.

Y una conclusión sobre el techo: todas las mejoras probadas mueven el punto de
operación pero **no el PR-AUC**. El límite es el backbone congelado.

---

## 2 · Qué se propuso

1. Ampliar la entrada a **4 bloques de 6 kb (24 kb)** con max pooling.
2. Evaluar con **8 bloques (48 kb)** usando el mismo modelo.
3. Reentrenar los tres reinos con **balance 1:1** y `class_weight`.
4. **Calibrar el umbral** en validación.
5. Repetir el **test oro** con definición de LTR armonizada.
6. Completar cobertura de especies y negativas.

---

## 3 · Datos

| Reino | Especies | Ventanas | Train | % positivo (train) |
|---|---|---|---|---|
| Hongos | 116 | 79.472 | 58.075 | 44 % |
| Plantas | 66 | 506.635 | 408.713 | 96 % |
| Animales | 69 | 950.663 | 860.591 | 68 % |

- Splits **disjuntos por especie**: ninguna especie en dos conjuntos.
- 0 ventanas compartidas entre train, val y test (verificado).
- Ventana de 50 kb, bloques de 6 kb.
- Taxonomía de animales obtenida de GBIF.

---

## 4 · Método

**Entrada.** Cada ventana aporta 4 bloques en los offsets 0, 14.667, 29.333 y
44.000. Cada bloque pasa por el backbone congelado; se obtiene un embedding de
1.280 dimensiones por bloque y se combinan con max pooling elemento a elemento.

**Por qué max pooling y no concatenar.** Concatenar fija la dimensión de entrada
(4 × 1.280 = 5.120) e impide evaluar con otro número de bloques. Con max pooling
la entrada al cabezal es siempre 1.280, así que el mismo modelo se evalúa con 4 u
8 bloques.

**Entrenamiento.** Solo el cabezal (MLP de tres capas). Balance 1:1 con
`class_weight`. El precalculo submuestrea el train conservando todas las negativas
y el mismo número de positivas; validación y prueba quedan completos.

**Umbral.** Se usa 0,5. Como el entrenamiento es 1:1, el modelo ya está calibrado
alrededor de ese punto; calibrar sobre validación desbalanceada empeora.

**Métricas.** Balanced accuracy y PR-AUC como principales. F1 solo cuando la
prevalencia está cerca del 50 %.

---

## 5 · Resultados por reino

### Hongos

| Métrica (test, umbral 0,50) | Modelo anterior (6 kb) | Nuevo (24 kb) |
|---|---|---|
| F1 | 0,661 | **0,729** |
| Recall | 0,702 | **0,856** |
| Precision | 0,624 | 0,635 |
| Balanced accuracy | — | **0,770** |
| PR-AUC | — | **0,835** |

32 épocas, mejor época 22. La curva muestra que aprende en las primeras seis
épocas y después se estabiliza.

### Plantas

36 épocas, mejor época 26 (por balanced accuracy).

| Métrica | Valor |
|---|---|
| Balanced accuracy | **0,735** |
| F1 | 0,862 |
| Precision | 0,965 |
| Recall | 0,778 |
| PR-AUC | 0,980 |

Advertencia: el test es 92 % positivo. El F1 de 0,862 no es comparable con el de
los otros reinos; la métrica válida es balanced accuracy.

### Animales

69 especies, 14 épocas, mejor época 12.

| Métrica | Valor | Azar |
|---|---|---|
| Balanced accuracy | **0,621** | 0,500 |
| PR-AUC | **0,757** | ~0,590 |
| F1 | 0,606 | — |
| Recall | 0,517 | — |

Es el reino más difícil: 44 especies en entrenamiento, desde corales hasta
mamíferos, y el test es disjunto por especie.

---

## 6 · Test oro (TAIR10)

Genoma de *Arabidopsis thaliana* con anotación curada: 5 cromosomas, 5.962 LTR y
25.227 elementos descartados por no ser LTR. 2.380 ventanas, 49,7 % positivas.
Definición de LTR armonizada con la del entrenamiento.

| Entrada | Precision | Recall | F1 | Balanced acc | PR-AUC |
|---|---|---|---|---|---|
| 24 kb (4 bloques) | 0,895 | 0,361 | 0,514 | 0,660 | 0,781 |
| **48 kb (8 bloques)** | 0,862 | **0,465** | **0,604** | **0,696** | **0,802** |

Mismo modelo, mismo umbral: solo cambia cuántos bloques ve. Duplicar la entrada
sube el F1 nueve puntos sin reentrenar.

El modelo es conservador: precisión alta (0,86-0,89), recall moderado (0,36-0,47).

---

## 7 · Los ocho experimentos

| # | Estrategia | Resultado |
|---|---|---|
| 1 | Entrada de 24 kb con max pooling | **Funciona**: +0,07 de F1 en hongos |
| 2 | Evaluar con 48 kb sin reentrenar | **Funciona**: +0,09 de F1 en el test oro |
| 3 | Negativas difíciles (reemplazo) | Ayuda en plantas (+0,03 BA); neutro en el ensamble de animales |
| 4 | Ensamble de semillas | +0,01 estable en plantas y animales |
| 5 | 10 especies nuevas (44.589 negativas) | Mejora el punto de operación; PR-AUC igual |
| 6 | Umbral por grupo taxonómico | **Falla**: 1-3 especies por grupo en validación |
| 7 | Calibrar el umbral en general | **No aporta** con entrenamiento 1:1 |
| 8 | Entrenar con 48 kb | Marginal (+0,006 BA, PR-AUC +0,001): no compensa el coste |

Detalle del experimento 8 (piloto con 18 especies, evaluación común en 8 bloques):

| Entrenado con | BA | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| 48 kb (8 bloques) | 0,568 | 0,326 | 0,662 | 0,216 | 0,566 |
| 24 kb (4 bloques) | 0,562 | 0,285 | 0,705 | 0,179 | 0,565 |

---

## 8 · Hallazgos metodológicos

1. **El F1 no sirve como criterio único con prevalencia alta.** Un clasificador
   trivial supera al modelo en plantas (0,956 frente a 0,733 en la primera
   corrida). Hay que reportar balanced accuracy y PR-AUC.

2. **La selección de época importa tanto como el modelo.** Con validación al 96 %
   positivo, el F1 premia al modelo que predice casi todo positivo. Cambiar el
   criterio a balanced accuracy llevó plantas de 0,556 a 0,707 sin tocar nada más.

3. **Con entrenamiento 1:1, el umbral correcto es 0,5.** Calibrarlo en validación
   empeoró el test en los tres reinos (hongos 0,770 → 0,750).

4. **El umbral por grupo necesita muchas más especies en validación.** Con 1 a 3
   especies por grupo, los cortes se ajustan al ruido y el resultado se derrumba
   (animales 0,621 → 0,538).

5. **Más datos no siempre mejoran la separación.** Añadir 44.589 negativas mejoró
   F1 y recall, pero el PR-AUC quedó igual. Lo mismo con entrenar a 48 kb.

---

## 9 · Limitaciones

- **Plantas**: solo 16.703 negativas en 35 especies. Dos especies se excluyeron
  porque el genoma disponible era otro ensamblado y no coincidían las coordenadas.
- **Animales**: validación con 8 especies y prueba con 9. La estimación tiene
  mucha varianza: la brecha validación-test es de 9 puntos.
- **Cobertura**: faltan ventanas de 8 hongos, 78 plantas y ~90 animales por falta
  de anotaciones (BED) o de genomas con el ensamblado correcto.
- **Backbone congelado**: no se hizo fine-tuning, que es la mejora de fondo
  pendiente.
- **Definición de LTR**: se usa solapamiento del 20 % del elemento; otros criterios
  darían cifras distintas.

---

## 10 · Trabajo futuro

1. **Descongelar parte del NT-500M** y entrenar de punta a punta. Es el único
   cambio que puede mover el PR-AUC; implica abandonar los embeddings en caché y
   días de GPU.
2. **Más negativas con anotación válida**: conseguir GFF para generar negativas
   génicas y completar las especies sin BED.
3. **Validación más grande por grupo**, para poder calibrar umbrales por grupo
   con garantías.
4. **Evaluar sobre más genomas curados** además de TAIR10, para tener un test oro
   cross-kingdom.

---

## Anexo · Reproducibilidad

- Código: `Inpactor3_segnt/training/` (pipeline v2) y `Inpactor3_segnt/scripts/`
  (dataset).
- El repositorio no incluye presentaciones ni guiones.
- Los bundles de secuencia y los embeddings no se versionan por tamaño; se
  regeneran con `make_sequence_bundle.py` y `precompute_embeddings.py`.

---

## 11 · Cambio de cuerpo (NT-100M multi-especie)

Ultimo experimento: sustituir el backbone `nucleotide-transformer-500m-human-ref`
por `nucleotide-transformer-v2-100m-multi-species`, preentrenado con muchas
especies en lugar de genoma humano. El resto del pipeline no cambia: 4 bloques de
6 kb, max pooling, cabezal MLP, balance 1:1 y umbral elegido en validacion.

Detalles tecnicos:
- El modelo v2 expone **512 dimensiones** por bloque (no 1.280) y no publica
  `AutoModel`: hay que cargarlo con `modeling_esm.EsmModel` (`--trust_remote_code`).
- Requiere **transformers 4.46** (con la 5.x el codigo remoto falla).
- Es **5 veces mas pequeno** (95,8 M parametros), asi que el precalculo es mas rapido.

### Resultados

| Reino | PR-AUC 500M | PR-AUC v2 | Diferencia |
|---|---|---|---|
| **Animales** | 0,760 | **0,812** | **+0,052** |
| Plantas | 0,980 | 0,985 | +0,005 |
| Hongos | 0,835 | 0,843 | +0,008 |

En animales, con el test completo (105.584 ventanas, prevalencia 0,59):

| Modelo | BA @0,5 | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| 500M human-ref | 0,6154 | 0,509 | 0,728 | 0,376 | 0,760 |
| **v2-100M** | **0,6632** | **0,717** | 0,728 | **0,705** | **0,812** |

El recall pasa de 0,376 a 0,705 manteniendo la precision: el modelo anterior era
conservador y el v2 encuentra casi el doble de transposones.

### Interpretacion

El cambio de cuerpo **solo mejora donde el lector anterior encajaba mal**. En
hongos y plantas el 500M ya representaba bien y el v2 empata (+0,008 y +0,005,
dentro del ruido). En animales, cuyo dominio (corales, insectos, medusas) esta
lejisimos del genoma humano, el preentrenamiento multi-especie aporta +0,052 de
PR-AUC, la unica mejora de capacidad —no solo de punto de operacion— de todo el
trabajo.

### Incidencias resueltas

1. El dataset de animales en Kaggle era anterior al arreglo de secuencias vacias:
   tenia 14 especies con ventanas todo `N`, lo que dejo la validacion degenerada
   (`val_f1 = 0`) y produjo un PR-AUC falso de 0,79. Se corrigieron esas especies.
2. El precalculo se salta una especie si ya existen sus embeddings: hubo que
   borrarlos antes de recalcular.
3. `Function.map` de Modal devuelve un iterador; sin recorrerlo las tareas no se
   ejecutan.

Las tablas completas, epoca por epoca, estan en `docs/tablas_epocas_v2.md`.

### Test oro con el cuerpo v2

Mismo genoma curado, mismo umbral calibrado (0,505), solo cambia el numero de bloques:

| Entrada | Precision | Recall | F1 | Balanced acc | PR-AUC |
|---|---|---|---|---|---|
| 24 kb (4 bloques) | 0,909 | 0,386 | 0,542 | 0,674 | 0,799 |
| **48 kb (8 bloques)** | 0,898 | **0,461** | **0,609** | **0,704** | **0,810** |

Comparado con el cuerpo de 500M (F1 0,514 y 0,604), el v2 mejora en las dos
configuraciones y **mantiene el efecto del contexto**: +0,067 de F1 y +0,075 de
recall al duplicar la entrada, sin reentrenar.

### Segundo test oro: Drosophila melanogaster

Genoma dm6 (UCSC) con la anotacion RepeatMasker del mismo ensamblado:
19.029 elementos LTR (Gypsy 14.237, Pao 3.746, Copia 1.046) frente a 118.526 no-LTR.
2.674 ventanas de 50 kb sobre los cromosomas principales. Modelo: animales v2.

| Entrada | Precision | Recall | F1 | Balanced acc | PR-AUC |
|---|---|---|---|---|---|
| 24 kb (4 bloques) | 0,933 | 0,434 | 0,592 | 0,705 | 0,788 |
| **48 kb (8 bloques)** | 0,934 | **0,466** | **0,622** | **0,720** | **0,812** |

Mismo patron que en Arabidopsis: el contexto adicional sube el F1 (+0,030) y el
recall (+0,032) sin reentrenar. Con dos especies independientes el efecto queda
confirmado.

### Set plata: Schizosaccharomyces pombe

Especie de hongo fuera del entrenamiento. Anotacion: los 13 elementos Tf2
(retrotransposon LTR/Gypsy) que da PomBase, sobre el genoma ASM294v2 de Ensembl.

| Entrada | Precision | Recall | F1 | Balanced acc | PR-AUC |
|---|---|---|---|---|---|
| 24 kb (4 bloques) | 0,333 | 0,750 | 0,462 | 0,837 | 0,607 |
| **48 kb (8 bloques)** | 0,110 | **0,917** | 0,196 | 0,771 | **0,657** |

Solo 12 ventanas positivas de 250 (prevalencia 4,8%), asi que **es una
comprobacion de cordura, no un test comparable al oro**. La direccion se repite:
mas contexto sube el recall (0,75 -> 0,92) y el PR-AUC (0,607 -> 0,657), frente a
un azar de 0,048. La precision cae porque, al ser una especie nueva, el umbral
calibrado en hongos (0,65) marca 100 de las 250 ventanas.

Se descarto el test plata original de *Aspergillus nidulans*: ninguna fuente
publica accesible anota sus transposones (el GFF de Ensembl Fungi solo trae
genes, y la particion curada de Dfam 4.0 no contiene familias fungicas). La via
alternativa, la particion no curada de Dfam (4,1 GB) mas RepeatMasker, se
descarto por coste (2-4 h de descarga y anotacion) y porque, con 13 elementos,
el resultado no aportaria mas que este.

### Conclusion del cambio de cuerpo

| Reino | PR-AUC 500M | PR-AUC v2 | Diferencia |
|---|---|---|---|
| Animales | 0,760 | **0,812** | **+0,052** |
| Plantas | 0,980 | 0,985 | +0,005 |
| Hongos | 0,835 | 0,843 | +0,008 |
| Test oro (48 kb, F1) | 0,604 | **0,609** | +0,005 |

El backbone de 100M multi-especie es **igual o mejor en los tres reinos**, cinco
veces mas pequeno y mas rapido de precalcular. La mejora grande esta en animales,
donde el preentrenamiento humano encajaba mal; en hongos y plantas empata.
