# Plantas · MIL por bloque vs pooled (mismos datos)

Cache `emb_mil/Plantas` · 40 épocas · test 45.250 ventanas, prevalencia **91,6 %** (ojo: el F1 no es comparable).

## Test (umbral 0,50)

| Modelo | Época | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline v2 (datos Kaggle) | 34 | 0,7450 | 0,9084 | 0,9622 | 0,8604 | 0,9851 |
| Pooled (misma caché) | 7 | 0,7696 | 0,9059 | 0,9676 | 0,8517 | 0,9860 |
| **MIL (misma caché)** | 4 | 0,7643 | 0,9018 | 0,9670 | 0,8448 | 0,9838 |

MIL − pooled: BA −0,005 · F1 −0,004 · PR-AUC −0,002 → **empate** (todo dentro del ruido ±0,01).

## Resumen de los tres reinos (MIL vs pooled, mismos datos)

| Reino | PR-AUC pooled | PR-AUC MIL | Diferencia | Veredicto |
|---|---|---|---|---|
| Hongos | 0,8434 | 0,8515 | +0,008 | empate |
| Animales | 0,8194 | 0,8544 | **+0,035** | **mejora real** |
| Plantas | 0,9860 | 0,9838 | −0,002 | empate |

El MIL por bloque solo aporta en animales. En hongos y plantas el max pooling de
embeddings ya capturaba la señal; reordenar la agregación no cambia nada.
El mecanismo en animales: +precisión y −recall en el umbral calibrado, pero con una
mejora real del ordenamiento (PR-AUC) y del área bajo la curva.
