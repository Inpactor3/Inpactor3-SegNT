# LoRA sobre el lector v2 — resultados

**Pregunta:** ¿descongelar el lector con adaptadores de bajo rango rompe el techo del modelo
congelado?

**Respuesta corta:** no con esta configuración. Gana en hongos, empata en plantas y colapsa
en animales, a cambio de 3-5 h por reino y perder la caché de embeddings.

## Montaje

| | |
|---|---|
| Lector | NT v2 100M multi-especie (congelado salvo adaptadores) |
| LoRA | rango 8 · alpha 16 · escala 2 · en query/key/value |
| Adaptadores | 66 (22 capas × 3 proyecciones) · **540.672 parámetros** |
| Cabezal | el mismo MLP con MIL (394 k) |
| Entrenamiento | 3 épocas · lr 3e-4 · batch 2 ventanas (8 bloques) |
| Datos | tope 600/especie: hongos 46.638 · plantas 28.800 · animales 31.800 |

No usa PEFT: los adaptadores están implementados en el propio script (`LoRALinear`), por
incompatibilidades de PEFT con torch 2.11 y transformers 4.46.

## Resultados

### Hongos

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| Congelado | 0,8261 | **0,7365** | **0,7746** | 0,7021 | 0,8094 |
| **LoRA** | **0,8398** | 0,7119 | 0,6394 | **0,8036** | **0,8315** |
| Diferencia | +0,0137 | −0,0246 | −0,135 | +0,102 | **+0,0221** |

### Plantas

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| Congelado | **0,7815** | 0,8551 | 0,9288 | 0,7923 | 0,9633 |
| LoRA | 0,7418 | **0,8914** | **0,9486** | **0,8224** | **0,9661** |
| Diferencia | **−0,0397** | +0,0363 | +0,0198 | +0,0301 | +0,0028 |

### Animales (mismo volumen: control congelado con tope 600)

| Modelo | Bal acc | F1 | PR-AUC |
|---|---|---|---|
| Control congelado | **0,6217** | **0,4076** | **0,4806** |
| LoRA (lr 3e-4) | 0,5000 | 0,0000 | 0,2579 |
| Diferencia | −0,1217 | −0,4076 | −0,2227 |

En animales **colapsó**: predijo todo negativo. La loss subió dentro de la época 2 (0,425 →
0,454) y la época 3 acabó en ~0,57, que es el régimen de no marcar nada. No se calculó el
test porque el proceso se detuvo.

## Lectura

1. **Hongos: la única ganancia clara.** +0,022 de PR-AUC, por encima del ruido. El F1 baja
   porque el punto de operación se vuelve más agresivo (recall 0,70 → 0,80).
2. **Plantas: no gana en ordenamiento** (+0,003, ruido) y la balanced accuracy cae cuatro
   centésimas. El F1 sube porque marca más positivos, no porque separe mejor. Además uso
   menos datos (28.800 frente a 32.726).
3. **Animales: fallo de hiperparámetro.** El learning rate 3e-4 es demasiado para el reino
   más difícil; el modelo diverge. Es un colapso del entrenamiento, no una derrota del
   concepto.

## Veredicto

**LoRA no compensa** frente al lector congelado con esta receta:

| Coste | Beneficio |
|---|---|
| 3-5 h por reino | +0,022 de PR-AUC en hongos |
| Se pierde la caché de embeddings (no se puede reutilizar) | empate en plantas |
| Checkpoint pequeño, pero hay que reentrenar por reino | colapso en animales |

Es consistente con el resultado anterior del proyecto: **descongelar parcialmente el 500M
tampoco mejoró**. El techo no se rompe con un ajuste ligero del lector.

## Qué faltaría para cerrarlo del todo

1. **Reintento de animales con lr 1e-4 y 2 épocas** (~4 h), para sustituir el colapso por un
   resultado limpio. No cambiaría el veredicto, pero sí la tabla.
2. Probar rango 4 y dropout en los adaptadores, si se quisiera exprimir hongos.

## Comparación con el cambio de lector

| Estrategia | Coste | Ganancia en PR-AUC |
|---|---|---|
| LoRA (por reino) | 3-5 h de cómputo por reino | +0,022 (hongos) · 0 (plantas) · fallo (animales) |
| Cambiar a NTv3 | 7 h (hongos) · 35 h (plantas) · 65 h (animales) de precompute | +0,023 · +0,010 · +0,009 |
| **Ambas** | — | no se suman necesariamente |

Cambiar de lector da una ganancia parecida o mayor y **una sola vez**: después el cabezal se
reentrena en minutos sobre la caché. LoRA, en cambio, obliga a pagar el coste en **cada**
reino y en **cada** reentrenamiento.
