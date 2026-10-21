# Guion de presentación — 19 de octubre de 2026

Archivo de apoyo para exponer el avance del dataset (Inpactor2 Detect).
Presentación: `Inpactor3_segnt/docs/presentacion_avance_19oct.pptx`
Duración estimada: 6–7 minutos.

> Números actualizados al 19-oct 06:58. Si el procesamiento sigue corriendo,
> revisar los conteos antes de exponer (comando al final).

---

## Antes de empezar (1 frase)

> "Les muestro en tres slides qué se definió, qué llevamos y cómo se genera el dataset."

---

## Slide 1 — La idea (~1,5 min)

**Decir:**

> "El muestreo que se definió fueron **449 especies**: 149 hongos, 149 animales y 151 plantas, con split 70/15/15.
>
> Al buscarlas en NCBI, **54 no tienen genoma publicado**, así que el universo real quedó en **395 especies**.
>
> Hoy llevamos **231 procesadas** y **188 ya cortadas en ventanas etiquetadas**.
>
> ¿Qué significa procesar? Que ya sabemos en qué coordenadas del genoma están sus transposones. Y 'ventanas' significa que partimos el genoma en pedazos de 50.000 letras y le pusimos a cada pedazo una etiqueta: tiene transposón o no tiene.
>
> Con esos pedazos etiquetados se entrena el modelo."

**Transición:** "Les explico cómo se hace en cuatro pasos."

---

## Slide 2 — Cuatro pasos (~2,5 min)

**Paso 1 — Buscar**

> "Un programa llamado MMseqs2 compara el genoma contra **PanTEon**, el catálogo de transposones conocidos. El genoma es la consulta; PanTEon es la referencia. El programa marca dónde se parece."

**Paso 2 — Limpiar (dedup)**

> "Aquí apareció un problema. PanTEon tiene familias muy parecidas entre sí. Entonces un mismo transposón se reportaba contra varios modelos: el mismo lugar aparecía marcado 3, 5, hasta 33 veces.
>
> Es como mostrarle la misma foto de un auto a tres personas: una dice 'Toyota', otra 'Corolla', otra 'Toyota Corolla'. Es un solo auto, no tres.
>
> El **dedup** se queda con una sola marca por lugar: la de mayor identidad.
>
> Esto importa porque después contamos cuántos transposones hay por ventana. Sin dedup contábamos el mismo 30 veces y la cobertura salía en 388%, algo imposible. Con dedup da un valor real, por debajo del 100%."

**Paso 3 — Cortar**

> "Partimos el genoma en ventanas de 50.000 letras."

**Paso 4 — Etiquetar**

> "Si la ventana tiene un transposón de tipo LTR, va como **1**. Si no lo tiene, va como **0**."

---

## Slide 3 — Progreso (~1,5 min)

**Señalando la tabla:**

> "Por reino: **hongos 117 de 117** —está completo—, **animales 81 de 135** y **plantas 33 de 143**.
>
> En total: **231 procesadas** y **188 con ventanas**."

**Detalle de hongos (si preguntan):**

> "Hongos había quedado en 110 porque en 7 casos NCBI devolvió el **virus asociado** a la especie en vez del genoma del hongo: genomas de 2 a 4 KB. Identificamos los ensamblajes correctos y ya quedó en 117."

**Qué falta:**

> "Terminar 54 animales y 110 plantas. Y completar las negativas donde NCBI tenga anotación de genes, porque no todas las especies la tienen."

**Cierre:**

> "El límite ya no es el cómputo. Es la información disponible y la memoria para genomas muy grandes."

---

## Cierre general (20 s)

> "En resumen: el flujo completo está funcionando, llevamos 231 especies procesadas y 188 listas en ventanas. Hongos está terminado; animales a mitad y plantas avanzando. El siguiente hito es terminar animales y plantas y entrenar el modelo para comparar con Jhonier y Alejandro."

---

## Preguntas probables y respuestas cortas

**¿Qué es un BED?**
> "Una lista de coordenadas: en qué cromosoma, desde y hasta qué posición está un transposón."

**¿Qué es el dedup?**
> "Es colapsar las marcas repetidas de un mismo lugar. El buscador marcaba el mismo locus contra varios modelos parecidos; nos quedamos con la mejor marca."

**¿Qué es un locus?**
> "Un lugar concreto del genoma: cromosoma más posición inicial y final."

**¿Entonces MMseqs2 no comparó con PanTEon?**
> "Sí comparó: PanTEon es la referencia. Precisamente por comparar contra todo el catálogo, un mismo elemento aparecía contra varios modelos. Eso es lo que corrige el dedup."

**¿Qué son las negativas?**
> "Ventanas donde no hay un transposón LTR. No las tomamos al azar: las clasificamos en intergenic, genic y other_te, para que el modelo aprenda a distinguir casos parecidos. Implementamos 3 de 5 categorías; las otras dos requieren RepeatMasker y datos de centrómeros."

**¿Por qué el positivo es solo LTR?**
> "Porque las categorías negativas asumen que positivo es LTR. Con 'cualquier transposón' el dataset salía 98% positivo y no era entrenable."

**¿Por qué plantas tiene un porcentaje alto de positivos?**
> "Porque solo una parte de las especies tiene negativas: faltan anotaciones de genes en NCBI."

**¿Qué es NCBI?**
> "La base pública de genomas. De ahí descargamos el ADN y la anotación de genes."

**¿Por qué no usaron Colab?**
> "Funcionaba para cómputo corto, pero el runtime se reinicia y borra el trabajo. Perdimos horas; usamos los PCs locales."

**¿Cuánto falta?**
> "54 animales y 110 plantas. El límite es la anotación disponible y la memoria, no el tiempo de cómputo."

---

## Verificar números antes de exponer

```bash
D=/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM
for r in Hongos Animales Plantas; do
  echo "$r BED=$(find $D/annotations/$r -name '*.bed'|wc -l) WIN=$(find $D/windows/$r -name '*.jsonl'|wc -l)"
done
```

Si cambian, actualizar la tabla de la diapositiva 3 y los totales de la diapositiva 1.
