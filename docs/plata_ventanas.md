# Test de plata · S. pombe (13 Tf2 de PomBase)

Los cuatro modelos de hongos evaluados sobre *Schizosaccharomyces pombe*, cromosomas I, II y III.
250 ventanas, **12 positivas** (prevalencia 4,8 %). Azar ≈ 0,05 de PR-AUC.

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| **Hongos 24 kb · MIL** | **0,8729** | **0,8182** | 0,8333 | 0,4167 | **0,8048** |
| Hongos 24 kb · pooled | 0,6229 | 0,3750 | 0,6000 | 0,2500 | 0,5740 |
| Hongos 100 kb · MIL | 0,8666 | 0,7200 | 0,8333 | 0,4167 | 0,7752 |
| Hongos 100 kb · pooled | 0,7062 | 0,5556 | 0,8333 | 0,4167 | 0,7101 |

## Lectura

1. **MIL es decisivo en los dos tamaños.** A 24 kb, MIL saca +0,25 de balanced accuracy y
   +0,23 de PR-AUC sobre pooled. Es exactamente el escenario para el que se diseñó: un
   elemento pequeño (Tf2 mide ~4,5 kb) dentro de una ventana grande, donde el max de
   embeddings lo diluye pero el max de logits lo detecta.
2. **El tamaño de ventana queda indefinido.** Con MIL, 24 kb gana (PR-AUC 0,805 vs 0,775);
   con pooled, gana 100 kb. Con 12 positivos la diferencia no es interpretable.
3. **Comparación con el modelo de 50 kb del corpus** (pooled): PR-AUC 0,607-0,657. El MIL
   a 24 kb llega a 0,805.

## Limitación

n = 250 ventanas, 12 positivas y una sola especie. Es una comprobación de cordura, no una
prueba estadística. Sirve para confirmar la dirección del efecto MIL, no para decidir el
tamaño de ventana en hongos.
