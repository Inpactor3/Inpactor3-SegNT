# Animales · MIL por bloque (max de logits) — métricas

Cache: `emb_mil/Animales` (N,4,512) · 52 archivos de especie en train.
15 épocas, balance 1:1, class_weight. Cifras por época = **validación**.

| Época | Loss | Val loss | Precisión | Recall | F1 | Accuracy | Bal acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.5788 | 0.6024 | 0.8113 | 0.7223 | 0.7642 | 0.7203 | 0.7197 |
| 2 | 0.5348 | 0.6480 | 0.8389 | 0.6128 | 0.7083 | 0.6832 | 0.7073 |
| 3 | 0.5214 | 0.6634 | 0.8186 | 0.6148 | 0.7022 | 0.6728 | 0.6927 |
| 4 | 0.5146 | 0.6604 | 0.8355 | 0.6530 | 0.7331 | 0.7016 | 0.7183 |
| 5 **←** | 0.5091 | 0.6730 | 0.8480 | 0.6536 | 0.7382 | 0.7091 | 0.7281 |
| 6 | 0.5054 | 0.6719 | 0.8589 | 0.5994 | 0.7060 | 0.6868 | 0.7168 |
| 7 | 0.5014 | 0.7234 | 0.8520 | 0.5956 | 0.7011 | 0.6813 | 0.7106 |
| 8 | 0.4996 | 0.6783 | 0.8227 | 0.6682 | 0.7374 | 0.7015 | 0.7128 |
| 9 | 0.4970 | 0.6850 | 0.8282 | 0.6359 | 0.7194 | 0.6887 | 0.7068 |
| 10 | 0.4942 | 0.7034 | 0.8702 | 0.5659 | 0.6858 | 0.6747 | 0.7119 |
| 11 | 0.4922 | 0.7212 | 0.8453 | 0.5711 | 0.6817 | 0.6653 | 0.6975 |
| 12 | 0.4906 | 0.7014 | 0.8132 | 0.6037 | 0.6930 | 0.6644 | 0.6851 |
| 13 | 0.4894 | 0.6894 | 0.8235 | 0.6227 | 0.7091 | 0.6795 | 0.6990 |
| 14 | 0.4868 | 0.7482 | 0.8421 | 0.6051 | 0.7042 | 0.6810 | 0.7070 |
| 15 | 0.4852 | 0.7325 | 0.8457 | 0.5945 | 0.6982 | 0.6775 | 0.7059 |

Mejor época: **5** · val BA 0.7281 · val F1 0.7382.

## Test (105.584 ventanas, prevalencia 0,59)

| Umbral | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| 0,55 (calibrado) | 0,7004 | 0,6872 | 0,8159 | 0,5936 | **0,8544** |
| 0,50 | 0,6890 | 0,7085 | 0,7733 | 0,6538 | 0,8544 |

## Comparación con el baseline v2

| Modelo | Época | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (max de embeddings) | 2 | 0,6632 | 0,7166 | 0,7282 | 0,7054 | 0,8116 |
| **MIL (max de logits)** | 5 | **0,7004** | 0,6872 | **0,8159** | 0,5936 | **0,8544** |
| Diferencia | | +0,037 | −0,029 | +0,088 | −0,112 | **+0,043** |

Lectura: aquí el PR-AUC sube **+0,043**, cuatro veces el ruido entre semillas (±0,01).
No es empate como en hongos. El efecto es el mismo que allí —más precisión, menos recall—
pero de mucha mayor magnitud, y en el reino donde el modelo era más débil.

## Comparación limpia (mismos datos)

Se repitió el baseline pooled sobre **esta misma caché** (52 especies, 424.831 ventanas
de train, mismo balance 1:1), cambiando solo el orden del max:

| Modelo (misma caché) | Época | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Pooled (max de embeddings) | 2 | 0,6752 | 0,6948 | 0,7610 | 0,6392 | 0,8194 |
| **MIL (max de logits)** | 5 | **0,7004** | 0,6872 | **0,8159** | 0,5936 | **0,8544** |
| Diferencia | | +0,025 | −0,008 | +0,055 | −0,046 | **+0,035** |

Descomposición:

| Cambio | PR-AUC |
|---|---|
| Baseline original (44 especies del dataset Kaggle) | 0,8116 |
| + datos nuevos del bundle (52 especies) | 0,8194 (+0,008) |
| **+ agregación MIL por bloque** | **0,8544 (+0,035)** |

Conclusión: **el MIL aporta +0,035 de PR-AUC con datos idénticos**, tres veces el ruido
entre semillas. No es el empate de hongos. El mecanismo es el mismo en los dos reinos
(más precisión, menos recall en el umbral calibrado), pero en animales la magnitud es real.

Nota: los 19 archivos del servidor que no están en el dataset de Kaggle explican solo
+0,008 del cambio, no el efecto principal.
