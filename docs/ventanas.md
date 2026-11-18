# Ventanas 24 kb / 50 kb / 100 kb — resultados completos (12 corridas)

Mismo entrenamiento salvo el tamaño de la ventana etiquetada: 4 bloques de 6 kb (24 kb de entrada),
submuestreo 1:1, MIL y pooled sobre la misma caché.

**Aviso:** cada tamaño tiene su propio test, así que las filas de 24 kb y 100 kb **no** son
comparables en términos absolutos (cambia la prevalencia y el n). La comparación MIL vs pooled
dentro de una fila sí es válida. La comparación entre tamaños necesita el test oro.

## Test por corrida

| Corrida | Umbral | Época | Bal acc | F1 | Precisión | Recall | PR-AUC | n | Prev. | PR-AUC/prev |
|---|---|---|---|---|---|---|---|---|---|---|
| Animales_w100_embeddings | 0.45 | 3 | 0.5902 | 0.5853 | 0.6570 | 0.5276 | 0.6917 | 13,619 | 0.56 | 1.24 |
| Animales_w100_logits | 0.44 | 2 | 0.5826 | 0.6174 | 0.6350 | 0.6007 | 0.6954 | 13,619 | 0.56 | 1.25 |
| Animales_w24_embeddings | 0.55 | 13 | 0.6193 | 0.3999 | 0.5062 | 0.3304 | 0.4765 | 58,893 | 0.22 | 2.15 |
| Animales_w24_logits | 0.45 | 11 | 0.6319 | 0.4239 | 0.5267 | 0.3547 | 0.5012 | 58,893 | 0.22 | 2.26 |
| Hongos_w100_embeddings | 0.62 | 16 | 0.7370 | 0.6641 | 0.8348 | 0.5513 | 0.8216 | 4,441 | 0.41 | 1.98 |
| Hongos_w100_logits | 0.57 | 7 | 0.7454 | 0.6760 | 0.8491 | 0.5616 | 0.8178 | 4,441 | 0.41 | 1.97 |
| Hongos_w24_embeddings | 0.62 | 8 | 0.8195 | 0.7202 | 0.7445 | 0.6975 | 0.7982 | 22,371 | 0.20 | 4.07 |
| Hongos_w24_logits | 0.62 | 8 | 0.8261 | 0.7365 | 0.7746 | 0.7021 | 0.8094 | 22,371 | 0.20 | 4.12 |
| Plantas_w100_embeddings | 0.47 | 8 | 0.7608 | 0.9505 | 0.9857 | 0.9178 | 0.9957 | 22,528 | 0.97 | 1.03 |
| Plantas_w100_logits | 0.43 | 8 | 0.7573 | 0.9489 | 0.9855 | 0.9148 | 0.9943 | 22,528 | 0.97 | 1.03 |
| Plantas_w24_embeddings | 0.51 | 19 | 0.7584 | 0.8546 | 0.9138 | 0.8026 | 0.9566 | 94,980 | 0.79 | 1.21 |
| Plantas_w24_logits | 0.49 | 10 | 0.7815 | 0.8551 | 0.9288 | 0.7923 | 0.9633 | 94,980 | 0.79 | 1.22 |

## MIL vs pooled (misma fila = mismo test)

| Reino | Ventana | BA pooled | BA MIL | Δ BA | Δ PR-AUC |
|---|---|---|---|---|---|
| Hongos | 24 kb | 0,8195 | **0,8261** | +0,0066 | +0,0112 |
| Hongos | 100 kb | 0,7370 | **0,7454** | +0,0084 | −0,0038 |
| Plantas | 24 kb | 0,7584 | **0,7815** | +0,0231 | +0,0067 |
| Plantas | 100 kb | **0,7608** | 0,7573 | −0,0035 | −0,0014 |
| Animales | 24 kb | 0,6193 | **0,6319** | +0,0126 | +0,0247 |
| Animales | 100 kb | **0,5902** | 0,5826 | −0,0076 | +0,0037 |

MIL gana la balanced accuracy en **4 de 6**; las dos derrotas son de −0,004 y −0,008.

## Bal acc y PR-AUC normalizado por tamaño

| Reino | BA 24 kb | BA 100 kb | Δ BA | PR-AUC/prev 24 kb | PR-AUC/prev 100 kb |
|---|---|---|---|---|---|
| Hongos | **0,8261** | 0,7454 | +0,0807 | **4,13** | 1,97 |
| Plantas | **0,7815** | 0,7573 | +0,0242 | **1,22** | 1,02 |
| Animales | **0,6319** | 0,5826 | +0,0493 | **2,26** | 1,24 |

La balanced accuracy favorece a 24 kb en los tres reinos. El PR-AUC crudo sube con la
ventana, pero eso es un artefacto de la prevalencia: normalizado (PR-AUC / prevalencia),
24 kb es entre 1,2 y 2,1 veces mejor en los tres casos.

## Nota sobre el volumen de entrenamiento

Animales 24 kb se entrenó con un **tope de 6.000 ventanas por especie** (287.814 en total)
para que el precompute fuera viable; animales 100 kb se entrenó con el train completo
(442.000 ventanas). Hongos y plantas usan el 1:1 íntegro. Es una diferencia de diseño que
hay que tener en cuenta al leer las cifras de animales.

## Prevalencia de test

| Reino | 24 kb | 50 kb (corpus) | 100 kb |
|---|---|---|---|
| Hongos | 19,6 % | — | 41,5 % |
| Plantas | 79,1 % | 92 % | 97,0 % |
| Animales | 22,2 % | 59 % | 56,0 % |
