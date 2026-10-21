# Guión de exposición · HOY (16/10/2026)

**Presentación**: `docs/presentacion_simple.pptx` (10 slides)
**Duración**: 10-12 minutos
**Público**: equipo o profesor (ajustar tono)

---

## 🎙️ Guión slide por slide

### 🎯 Slide 1 · Portada (20 seg)

> *"Buenos días. Les voy a presentar el avance en la construcción del nuevo dataset de entrenamiento. Lo nuevo desde la última reunión: muestreo estratificado de 449 especies, migración técnica a MMseqs2 y pipeline distribuido en 2 PCs. Son 10 minutos."*

👉 Slide 2

---

### 🚨 Slide 2 · El problema (recap · 1 min)

> *"Rápido recap para los que no estuvieron la reunión anterior. Entrenamos el modelo con PanTEon pre-procesado. En el laboratorio sacó F1=0.67, decente. Pero en genoma real colapsó a 0.08. El modelo nunca vio el ruido natural del genoma — fenómeno llamado 'domain shift'.*
>
> *La solución, que ya aprobó Simón: construir el dataset desde genomas reales con 4 cambios · genomas reales, 5 tipos de negativas, split por especie, nada sintético."*

👉 Slide 3

---

### 📊 Slide 3 · Muestreo estratificado 449 especies (2 min)

> *"Primera cosa nueva: implementé el muestreo que pidió Simón.*
>
> ***Metodología**:*
> *- Solo tomo los 3 reinos relevantes · Plantas, Animales, Hongos.*
> *- Estratifico por longitud en 4 cuartiles dentro de cada reino.*
> *- Muestreo proporcional desde cada estrato.*
> *- Split 70/15/15 preservando la distribución de longitud entre train, val, test.*
> *- Semilla fija 42 · todo reproducible.*
>
> ***Resultado**: 449 especies finales · 151 plantas + 149 animales + 149 hongos. La tabla muestra el detalle.*
>
> *Están guardadas en `species_split_proposal.csv` · versionado en GitHub."*

👉 Slide 4

---

### 🚀 Slide 4 · Cambio técnico · MMseqs2 (2 min)

> *"Segundo gran cambio · tuve que reemplazar RepeatMasker. Déjenme explicar por qué.*
>
> *Para una sola especie (S. pombe · 12 MB), RepeatMasker tardaba **42 minutos**. Para 449 especies ese ritmo era imposible · se iban 2 o 3 semanas de CPU.*
>
> *Descubrí **MMseqs2**, una herramienta moderna (2017) que hace **lo mismo** pero con algoritmos modernos de indexado. La misma especie que RepeatMasker tardaba 42 min, MMseqs2 la procesa en **1 minuto 25 segundos**. **30 veces más rápido**.*
>
> *Lo validé comparando resultados · misma biblioteca PanTEon, mismo criterio de match · ambas herramientas detectan los mismos TEs. El cambio hace viable el proyecto completo.*
>
> *En vez de 2 semanas → 1-2 días para los 3 reinos completos."*

👉 Slide 5

---

### 📊 Slide 5 · Lo que hice (números) (1 min)

> *"Resumiendo mi avance en números:*
> *- **449** especies muestreadas con estratificación.*
> *- **395** URLs resueltas desde NCBI (RefSeq + GenBank).*
> *- **348** genomas descargados · 190 GB en disco externo dedicado.*
> *- **165 BEDs anotados** · 110 de hongos + 55 de animales.*
>
> ***Además**:*
> *- Pipeline completo de 10 scripts versionados en GitHub.*
> *- Validación oficial contra TAIR10 (ahora les cuento).*
> *- Pipeline distribuido · 2 PCs en paralelo."*

👉 Slide 6

---

### 🏆 Slide 6 · Avance real por reino (2 min)

> *"Esta tabla es la foto actual del dataset.*
>
> ***Hongos**: **COMPLETO** end-to-end.*
> *- 117 genomas descargados.*
> *- 110 anotados con MMseqs2 + biblioteca PanTEon Fungi (71 MB).*
> *- 110 archivos .jsonl con ventanas de 50 kb etiquetadas.*
> *- 7 FAIL en hongos muy divergentes · documentados · es información biológica útil.*
>
> ***Animales**: 41% en progreso.*
> *- 135 descargados.*
> *- 55 anotados hasta ahora.*
> *- Los vertebrados grandes toman tiempo (horas por genoma) pero están corriendo.*
>
> ***Plantas**: encoladas.*
> *- 141 de 143 descargadas (dos obsoletas).*
> *- Anotación empieza cuando animales termine · ~1 día.*
>
> ***Hongos es la prueba que el pipeline funciona end-to-end** · el flujo es exactamente el mismo para los otros reinos."*

👉 Slide 7

---

### 💻 Slide 7 · Pipeline distribuido (1.5 min)

> *"Al ver que los vertebrados grandes eran lentos, aproveché una PC auxiliar en modo demo con 15 GB de RAM.*
>
> ***Estrategia**: División del trabajo por tamaño de genoma.*
> *- Mi PC principal procesa los vertebrados grandes en orden A→Z (lento pero sin riesgo).*
> *- La PC auxiliar procesa invertebrados pequeños (<500 MB) en orden Z→A (reverso) · no colisionan.*
> *- Sincronización automática · un script rsync baja los BEDs del auxiliar a mi disco externo cada 60 segundos.*
>
> ***Resultado**: 2 veces más rápido. La PC auxiliar completó 37 BEDs en 5 horas · habría tomado 10 horas en una sola máquina.*
>
> *Mostré que el pipeline es paralelizable sin cambios · se puede escalar fácilmente."*

👉 Slide 8

---

### 🏆 Slide 8 · Validación contra TAIR10 (1.5 min)

> *"Como pidió Simón al final · test ORO contra verdad externa curada.*
>
> *Comparé mi anotación de Arabidopsis (hecha con PanTEon) contra los **31,189 TEs curados a mano** por el proyecto TAIR durante décadas.*
>
> ***Resultados con IoU ≥ 0.5**:*
> *- Precisión: 29%.*
> *- Recall: 20%.*
> *- F1 global: 0.234.*
>
> ***Interpretación honesta**: el número bajo tiene sentido · PanTEon-Arabidopsis tiene 217 consensos vs décadas de curación humana en TAIR. Lo valioso es **el desglose por familia**:*
> *- Capturamos muy bien **Gypsy y Mariner** (recall 84%).*
> *- Perdemos casi todos los **Helitron** (recall 9%).*
>
> *Esto es **información accionable** · nos dice que la biblioteca PanTEon tiene sesgo en Helitron. Decisión posible: enriquecer la biblioteca antes de entrenar.*"

👉 Slide 9

---

### 🚧 Slide 9 · Lo que falta (1 min)

> *"Lo que viene:*
>
> ***Primero**: terminar MMseqs2 sobre los 80 vertebrados grandes restantes · 3-4 días en local.*
>
> ***Segundo**: lanzar MMseqs2 sobre las 141 plantas descargadas · 1 día cuando animales termine.*
>
> ***Tercero**: aplicar el downstream pipeline (ventanas 50 kb + 5 categorías de negativas · mi diferenciador) sobre animales y plantas · 1 día más.*
>
> ***Cuarto**: entrenar Inpactor 2 Detect sobre los 3 datasets del equipo (Jhonier, Alejandro, mío) · comparación justa.*
>
> *Para la próxima reunión con el profe · dataset completo y listo para evaluación comparativa."*

👉 Slide 10

---

### 🎁 Slide 10 · Resumen final (30 seg)

> *"Resumiendo:*
>
> ***HECHO**: muestreo 449 especies, 395 URLs, 348 genomas, 165 BEDs, hongos end-to-end, MMseqs2 adoptado, pipeline distribuido, validación TAIR10.*
>
> ***FALTA**: terminar animales (3 días), plantas (1 día), downstream, entrenamiento comparativo.*
>
> *La estrategia nueva **ya tiene evidencia funcional** · hongos lo demuestran. Solo falta tiempo de cómputo para los otros dos reinos."*

**Fin · gracias.**

---

## 📋 Datos clave para memorizar

```
449 especies muestreadas (151 plantas + 149 animales + 149 hongos)
395 URLs resueltas (RefSeq + GenBank)
348 genomas descargados · 190 GB
165 BEDs totales (110 hongos + 55 animales)
30x speedup con MMseqs2 vs RepeatMasker
F1 = 0.234 vs TAIR10 (test ORO Arabidopsis)
Hongos COMPLETO end-to-end
2 PCs en paralelo (local + auxiliar)
```

---

## 💪 Frase ganadora (para abrir o cerrar)

> *"**El pipeline está completo y validado** · Hongos ya está listo end-to-end · Animales y plantas siguen el mismo flujo probado · solo falta tiempo de cómputo. Para la próxima reunión los 3 reinos están completos."*

---

## 🤔 Preguntas probables + respuestas

### P: *"¿Por qué cambiar de RepeatMasker a MMseqs2?"*
> *"RepeatMasker toma 42 min por especie con la biblioteca Fungi. Para 449 especies son 2-3 semanas. MMseqs2 hace lo mismo en 1 min 25 s · 30x más rápido. Ambos usan la misma biblioteca PanTEon y producen BEDs equivalentes."*

### P: *"¿F1=0.234 contra TAIR10 no es muy bajo?"*
> *"En términos absolutos sí, pero PanTEon-Arabidopsis tiene 217 consensos · TAIR tiene décadas de curación manual. El dato más valioso es el desglose por familia · capturamos bien Gypsy, perdemos Helitron. Esto es información para mejorar la biblioteca."*

### P: *"¿Qué especies van en el test ORO?"*
> *"Arabidopsis ya está validada con TAIR. Para los otros reinos propongo Oryza (RiceTE), Drosophila (FlyBase), S. cerevisiae (SGD). Hay que acordarlo con el equipo."*

### P: *"¿Van a terminar a tiempo para la reunión con el profe?"*
> *"Sí · animales terminan en 3-4 días, plantas en 1 día. Para la próxima semana el dataset completo está listo."*

### P: *"¿Cómo garantizan reproducibilidad?"*
> *"Semilla fija 42 en todo el muestreo. Scripts versionados en GitHub. Manifest con URLs específicas de NCBI (RefSeq + GenBank). Cualquier equipo puede regenerar el dataset exacto."*

---

## 📎 Material a tener a mano

- `docs/presentacion_simple.pptx` · 10 slides (abrir en pantalla)
- `docs/species_split_proposal.csv` · 449 especies
- `docs/species_download_manifest.tsv` · URLs resueltas
- `docs/piloto_rm_reporte.md` · reporte hongos
- `docs/test_oro_arabidopsis.md` · F1 vs TAIR10

---

## ⚠️ Si te preguntan algo que no sabes

> *"Buena pregunta. No lo tengo claro ahora mismo · lo verifico después y te mando la respuesta."*

**Nunca inventes datos.**
