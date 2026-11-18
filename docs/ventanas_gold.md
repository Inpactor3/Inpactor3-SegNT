# Test oro de los modelos de ventana (referencia común)

Los ocho modelos se evaluaron sobre los mismos genomas, la misma rejilla de 50 kb y
**4 bloques (24 kb de entrada)**. Aquí sí es válido comparar 24 kb contra 100 kb, porque
el conjunto de evaluación es idéntico.

## Arabidopsis (TAIR10) · 2.380 ventanas

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| **Plantas 24 kb · MIL** | 0,7031 | 0,5953 | — | — | **0,8243** |
| Plantas 24 kb · pooled | **0,7125** | **0,6215** | — | — | 0,8211 |
| Plantas 100 kb · MIL | 0,6454 | 0,6874 | — | — | 0,7860 |
| Plantas 100 kb · pooled | 0,6340 | 0,6768 | — | — | 0,7799 |

**24 kb gana claramente**: +0,038 de PR-AUC y +0,058 de balanced accuracy (MIL).

## Drosophila (dm6) · 2.674 ventanas

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| Animales 24 kb · MIL | 0,7385 | 0,6549 | — | — | 0,8151 |
| Animales 24 kb · pooled | 0,7308 | 0,6479 | — | — | 0,8016 |
| **Animales 100 kb · MIL** | **0,7487** | **0,6940** | — | — | **0,8182** |
| Animales 100 kb · pooled | 0,7320 | 0,6670 | — | — | 0,8059 |

**Empate técnico**, con ventaja mínima de 100 kb (+0,003 de PR-AUC, +0,010 de BA).

## MIL contra pooled sobre la misma referencia

| Modelo | Δ BA (MIL − pooled) | Δ PR-AUC |
|---|---|---|
| Plantas 24 kb | −0,0094 | **+0,0032** |
| Plantas 100 kb | +0,0114 | **+0,0061** |
| Animales 24 kb | +0,0077 | **+0,0135** |
| Animales 100 kb | +0,0167 | **+0,0123** |

MIL mejora el PR-AUC en **las cuatro** comparaciones y la balanced accuracy en tres de cuatro.

## Conclusiones

1. **En plantas, 24 kb es mejor que 100 kb** sobre la referencia común. Coincide con lo que
   la prevalencia anunciaba: con 100 kb el 97 % del test es positivo y la etiqueta se diluye.
2. **En animales hay empate**, con una ventaja mínima de 100 kb. La gran diferencia que se
   veía en el test interno (PR-AUC 0,69 vs 0,50) era un artefacto de prevalencia.
3. **MIL por bloque es mejor agregador**: gana el PR-AUC en las cuatro comparaciones.
4. El 50 kb heredado queda en una zona razonable: hay evidencia de que 24 kb mejora en
   plantas, pero no de que 100 kb sea mejor en ningún reino.
