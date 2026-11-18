# Hongos · MIL por bloque (max de logits) — métricas

Cache: `emb_mil/Hongos` (N,4,512) · 47.050 ventanas train · 14.067 val · 7.330 test.
Entrenamiento: 15 épocas, balance 1:1, class_weight. Todas las cifras por época son de **validación**.

| Época | Loss | Val loss | Precisión | Recall | F1 | Accuracy | Bal acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.5419 | 0.4326 | 0.7906 | 0.8824 | 0.8340 | 0.8003 | 0.7873 |
| 2 | 0.4477 | 0.4094 | 0.7901 | 0.8924 | 0.8382 | 0.8042 | 0.7902 |
| 3 **←** | 0.4307 | 0.3930 | 0.8240 | 0.8813 | 0.8517 | 0.8255 | 0.8167 |
| 4 | 0.4201 | 0.4078 | 0.7902 | 0.8918 | 0.8379 | 0.8039 | 0.7900 |
| 5 | 0.4111 | 0.4031 | 0.7892 | 0.8904 | 0.8368 | 0.8026 | 0.7887 |
| 6 | 0.4044 | 0.4181 | 0.7629 | 0.9022 | 0.8267 | 0.7850 | 0.7665 |
| 7 | 0.3972 | 0.4173 | 0.7559 | 0.9039 | 0.8233 | 0.7795 | 0.7598 |
| 8 | 0.3932 | 0.4136 | 0.7612 | 0.9011 | 0.8252 | 0.7831 | 0.7644 |
| 9 | 0.3905 | 0.4148 | 0.7634 | 0.9009 | 0.8265 | 0.7850 | 0.7666 |
| 10 | 0.3860 | 0.3988 | 0.7754 | 0.8957 | 0.8312 | 0.7933 | 0.7771 |
| 11 | 0.3836 | 0.3924 | 0.7837 | 0.8962 | 0.8362 | 0.8004 | 0.7852 |
| 12 | 0.3773 | 0.4160 | 0.7574 | 0.9032 | 0.8239 | 0.7806 | 0.7611 |
| 13 | 0.3755 | 0.4428 | 0.7486 | 0.9103 | 0.8216 | 0.7753 | 0.7539 |
| 14 | 0.3720 | 0.4291 | 0.7646 | 0.8961 | 0.8252 | 0.7842 | 0.7665 |
| 15 | 0.3685 | 0.4073 | 0.7714 | 0.8968 | 0.8294 | 0.7903 | 0.7734 |

Mejor época por balanced accuracy: **3** (val BA 0.8167).

## Test (7.330 ventanas)

| Umbral | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| 0.68 (calibrado) | 0.7847 | 0.7303 | 0.8758 | 0.6262 | 0.8515 |
| 0,50 | 0.7866 | 0.7387 | 0.7566 | 0.7217 | 0.8515 |

## Comparación con el baseline (max de embeddings)

Comparación al **umbral calibrado de cada modelo** (el baseline usa 0,65; el MIL 0,68):

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| Baseline v2 (época 10) | 0,7766 | 0,7254 | 0,7564 | 0,6969 | 0,8434 |
| **MIL por bloque** (época 3) | 0.7847 | 0.7303 | 0.8758 | 0.6262 | 0.8515 |
| Diferencia | +0,008 | +0,005 | **+0,119** | **−0,071** | +0,008 |

Lectura correcta: el PR-AUC (libre de umbral) sube **+0,008**, dentro del ruido entre
semillas (±0,01). Lo que sí cambia es el **punto de operación**: el MIL se vuelve mucho
más preciso (0,876 vs 0,756) y más conservador en recall (0,626 vs 0,697).

Es decir, la hipótesis de partida —que puntuar por bloque detectaría mejor los elementos
pequeños, subiendo el recall— **no se cumple**. La agregación por bloque no cambia el
techo; solo desplaza el equilibrio precisión/recall y acelera la convergencia (época 3
frente a época 10).

Conclusión: **empate técnico**. No compensa rehacer las cachés de plantas y animales
por este resultado; se documenta como experimento negativo con la evidencia.
