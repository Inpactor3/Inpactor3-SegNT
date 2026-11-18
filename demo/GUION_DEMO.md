# Guion de la demo y del video

**Duración:** 5-6 minutos en vivo · 3 minutos en video.
**Objetivo:** que el panel vea el producto funcionando y entienda qué se midió.

---

## Antes de empezar (10 minutos)

| # | Acción | Comando / verificación |
|---|---|---|
| 1 | Actualizar el repo | `git pull` |
| 2 | Dependencias | `pip install -r requirements.txt` |
| 3 | Comprobar que está sano | `python -m unittest discover -s tests -q` → 6 pruebas OK |
| 4 | Precalentar el visor | `python3 visor/app.py --port 8000` y abrir `http://127.0.0.1:8000` |
| 5 | Pestañas abiertas | GitHub (README), visor en `/`, terminal en la raíz del repo |
| 6 | Limpieza | Cerrar notificaciones; no tener abierta ninguna terminal con tokens |

> Si vas a usar `--real`, ejecútalo una vez antes: la primera vez descarga el modelo (~400 MB).

---

## Escena 1 · El producto (30 s)

Muestra el repositorio en GitHub.

> "Es un detector de retrotransposones LTR. El lector de ADN está congelado; lo único que
> entrenamos es un cabezal pequeño. El repositorio tiene licencia GPLv3, manifiesto de datos,
> pruebas y una guía que cualquiera puede seguir."

Señala: `README.md`, `LICENSE`, `tests/`, `visor/`, `inpactor3/`.

---

## Escena 2 · Salud del repositorio (1 min)

```bash
python -m unittest discover -s tests -q
```

> "Seis pruebas en verde. Cubren el cálculo de offsets contra el usado en entrenamiento, la
> limpieza de secuencia y el generador de ventanas. Es el mismo comando que puede correr el
> jurado."

---

## Escena 3 · Cómo detecta (1 min 30 s)

```bash
bash demo/demo.sh
```

Salida esperada:

```
genoma: 1 secuencias · ventanas: 2 · bloques por ventana: 4 · ventana: 50000 pb
[dry-run] se evaluarian 2 ventanas y se escribiria /tmp/demo.bed
```

> "Este genoma de prueba tiene 120 kb, así que salen dos ventanas de 50 kb. De cada ventana
> tomamos cuatro bloques de 6 kb, porque el lector solo puede procesar ese tamaño. Son los
> bloques que ve el modelo, no la ventana completa."

El comando termina arrancando el visor.

---

## Escena 4 · El visor (2 min)

**`/`** — resumen:

> "Veintiuna corridas agrupadas por reino. Arriba, la de mayor PR-AUC y la de mayor test.
> La advertencia importa: cada reino tiene su propio test y su prevalencia, así que los
> números no se comparan entre filas."

**`/comparar`** — lectores:

> "Aquí está la comparación limpia: mismo test, mismo cabezal, distinto lector. NTv3 gana en
> hongos, plantas y en el oro; en animales empata usando el 11 % de los datos."

**`/corrida/Plantas_w24_logits`** — detalle:

> "Una corrida concreta: umbral calibrado en validación, época 10 guardada, y las cinco
> métricas del test."

**`/epocas/Hongos`** — entrenamiento:

> "Las veinte épocas. La fila resaltada es la que se guarda, elegida por balanced accuracy."

---

## Escena 5 · El detector real (1 min, opcional)

```bash
bash demo/demo.sh --real
```

> "Corre el lector sobre una ventana. En GPU tarda segundos; en CPU, un par de minutos. La
> salida es un BED con la probabilidad de cada región."

Si no hay GPU en el equipo, **no lo ejecutes en vivo**: muestra un BED preparado antes.

---

## Escena 6 · Cierre (30 s)

> "En resumen: el lector marca el techo, el contexto y la agregación por bloque mueven la
> métrica, y el modelo funciona fuera de su dominio con precisión cerca de 0,90 y recall
> cercano a 0,46. Está validado por el director científico y por Simón Orozco."

---

## Plan B (fallos y qué hacer)

| Si pasa esto | Qué haces |
|---|---|
| No hay internet | El visor y el dry-run no lo necesitan; omite `--real` |
| El puerto 8000 está ocupado | `python3 visor/app.py --port 8010` |
| La descarga del modelo falla | Muestra el BED preparado y explica que el modelo ya está en caché en el servidor |
| El proyector recorta | Zoom del navegador a 110 % o ventana maximizada |
| El visor tarda | Ya está precalentado desde la preparación |
| Preguntan por el clasificador de superfamilia | "Está documentado como etapa 2 en el anexo de desviaciones" |

---

## Guion del video (3 minutos)

| Tiempo | Plano | Voz |
|---|---|---|
| 0:00–0:25 | Repositorio en GitHub | "Inpactor3 detecta retrotransposones LTR con un lector de ADN congelado y un cabezal entrenable." |
| 0:25–0:50 | `tests` en verde | "El repositorio es reproducible: pruebas, licencia, manifiesto de datos y guía." |
| 0:50–1:20 | `demo.sh` (dry-run) | "Cada ventana de 50 kb aporta cuatro bloques de 6 kb, que son los que ve el modelo." |
| 1:20–2:05 | Visor `/` y `/comparar` | "Estas son las 21 corridas y la comparación entre lectores: NTv3 gana en tres de cuatro evaluaciones." |
| 2:05–2:35 | Visor `/corrida` y `/epocas` | "Cada corrida conserva su umbral, su época y sus métricas; y aquí está la curva de entrenamiento." |
| 2:35–3:00 | Cierre | "Precisión cercana a 0,90, recall cercano a 0,46, validado por los directores. El clasificador de superfamilia queda como etapa siguiente." |
