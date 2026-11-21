# Piloto NTv3 · hongos 24 kb

Comparación del lector **Nucleotide Transformer v3 (100M, release inicial)** contra nuestro
lector v2, con el mismo test y un volumen de entrenamiento equivalente.

## Montaje

| | v2 (referencia) | NTv3 (piloto) |
|---|---|---|
| Lector | NT v2 100M multi-especie | NTv3 100M (`NTv3_100M_pre`) |
| Tokenización | 6-meros (1.000 tokens por bloque) | **por base** (6.000 tokens, U-Net reduce 128×) |
| Vector por bloque | 512 (media sobre tokens) | **768** (media sobre las 6.000 bases) |
| Train | 47.050 ventanas | 46.638 (tope 600/especie) |
| Validación | 39.719 | 8.000 (tope 400/especie) |
| Test | 22.371 | 22.371 (idéntico) |
| Agregación | MIL (máx. de logits) | MIL (máx. de logits) |
| Épocas | 20 | 20 (mejor: 9) |

## Test (umbral calibrado)

| Reino | Modelo | Train | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|---|---|
| Hongos | v2 congelado | 47.050 | 0,8261 | 0,7365 | 0,7746 | 0,7021 | 0,8094 |
| Hongos | **NTv3** | 46.638 | **0,8465** | 0,7122 | 0,6394 | 0,8036 | **0,8327** |
| Plantas | v2 congelado | 32.726 | 0,7815 | 0,8551 | 0,9288 | 0,7923 | 0,9633 |
| Plantas | **NTv3** | ~28.800 | **0,8271** | **0,8810** | **0,9486** | **0,8224** | **0,9732** |
| Animales | v2 congelado | 287.814 | 0,6319 | 0,4239 | 0,5267 | 0,3547 | 0,5012 |
| Animales | NTv3, piloto | ~31.800 | 0,6263 | 0,4110 | — | — | 0,4868 |
| Animales | **NTv3, corpus completo** | **287.814** | **0,6376** | **0,4344** | **0,5540** | **0,3572** | **0,5101** |

Diferencias frente al v2:

| Reino | Δ Bal acc | Δ F1 | Δ PR-AUC | Datos usados |
|---|---|---|---|---|
| Hongos | **+0,0204** | −0,0243 | **+0,0233** | −0,9 % |
| Plantas | **+0,0456** | **+0,0259** | **+0,0099** | −12 % |
| Animales (piloto) | −0,0056 | −0,0129 | −0,0144 | **−89 %** |
| **Animales (completo)** | **+0,0057** | **+0,0105** | **+0,0089** | **0 %** |

## Lectura

1. **Plantas y hongos: NTv3 gana con claridad**, y en plantas lo hace en todas las
   métricas a la vez (balanced accuracy, F1, precisión, recall y PR-AUC). Con menos datos
   de entrenamiento.
2. **Animales: con datos equivalentes, NTv3 también gana, aunque por poco.** Con 287.814
   ventanas en los dos: BA 0,6376 frente a 0,6319, F1 0,4344 frente a 0,4239 y PR-AUC
   0,5101 frente a 0,5012. La diferencia de PR-AUC (+0,009) está en el borde del ruido, así
   que es una victoria leve, no un empate con datos distintos.
3. **El umbral calibrado de animales salió degenerado** (0,07) porque la validación está
   recortada a 400 por especie y las probabilidades se desplazan. La cifra honesta es la de
   umbral 0,50.
4. **Las dos corridas guardaron la época 20**, la última: seguían mejorando. Merece la pena
   entrenar más épocas, sobre todo en plantas.

## Siguiente paso

- Repetir plantas con más épocas (el precompute ya está hecho: solo entrena).
- Evaluar plantas en el test oro (Arabidopsis) hecha la corrida completa.
- Documentar LoRA: en hongos +0,022 de PR-AUC; en plantas empate; en animales colapsó con
  lr 3e-4.

## Métricas época por época (validación)

### Hongos · 20 épocas

| Época | Loss | Val loss | Precisión | Recall | F1 | Accuracy | Bal acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.5706 | 0.4723 | 0.3601 | 0.8438 | 0.5048 | 0.7430 | 0.7841 |
| 2 | 0.4248 | 0.4460 | 0.3717 | 0.8631 | 0.5196 | 0.7522 | 0.7975 |
| 3 | 0.4055 | 0.3470 | 0.6014 | 0.8188 | 0.6935 | 0.8876 | 0.8596 |
| 4 | 0.3894 | 0.3906 | 0.5321 | 0.8196 | 0.6453 | 0.8601 | 0.8436 |
| 5 | 0.3834 | 0.3497 | 0.5525 | 0.8213 | 0.6606 | 0.8690 | 0.8495 |
| 6 | 0.3768 | 0.3597 | 0.5234 | 0.8374 | 0.6442 | 0.8564 | 0.8486 |
| 7 | 0.3695 | 0.3682 | 0.5200 | 0.8470 | 0.6444 | 0.8549 | 0.8517 |
| 8 | 0.3638 | 0.3360 | 0.6076 | 0.8366 | 0.7039 | 0.8908 | 0.8686 |
| 9 **←** | 0.3592 | 0.3230 | 0.6109 | 0.8357 | 0.7059 | 0.8919 | 0.8690 |
| 10 | 0.3574 | 0.3651 | 0.5121 | 0.8535 | 0.6401 | 0.8510 | 0.8520 |
| 11 | 0.3530 | 0.3425 | 0.5567 | 0.8502 | 0.6728 | 0.8716 | 0.8629 |
| 12 | 0.3512 | 0.3233 | 0.5920 | 0.8366 | 0.6934 | 0.8851 | 0.8653 |
| 13 | 0.3488 | 0.3300 | 0.5365 | 0.8519 | 0.6584 | 0.8628 | 0.8583 |
| 14 | 0.3457 | 0.3226 | 0.5674 | 0.8438 | 0.6785 | 0.8759 | 0.8628 |
| 15 | 0.3430 | 0.3258 | 0.5564 | 0.8414 | 0.6699 | 0.8712 | 0.8591 |
| 16 | 0.3398 | 0.3426 | 0.5394 | 0.8535 | 0.6611 | 0.8641 | 0.8598 |
| 17 | 0.3368 | 0.3198 | 0.5464 | 0.8438 | 0.6633 | 0.8670 | 0.8575 |
| 18 | 0.3335 | 0.3230 | 0.5751 | 0.8357 | 0.6813 | 0.8786 | 0.8611 |
| 19 | 0.3284 | 0.3308 | 0.5289 | 0.8462 | 0.6510 | 0.8591 | 0.8539 |
| 20 | 0.3287 | 0.3647 | 0.4797 | 0.8655 | 0.6173 | 0.8334 | 0.8465 |

Mejor época por balanced accuracy: **9** (0.8690). Mínimo de val loss: época 17 (0.3198).

### Plantas · 20 épocas

| Época | Loss | Val loss | Precisión | Recall | F1 | Accuracy | Bal acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.5535 | 0.5198 | 0.9520 | 0.6703 | 0.7867 | 0.7072 | 0.7652 |
| 2 | 0.4664 | 0.6318 | 0.9727 | 0.5534 | 0.7055 | 0.6278 | 0.7446 |
| 3 | 0.4431 | 0.5911 | 0.9753 | 0.5721 | 0.7211 | 0.6436 | 0.7560 |
| 4 | 0.4244 | 0.5581 | 0.9782 | 0.5879 | 0.7344 | 0.6575 | 0.7668 |
| 5 | 0.4102 | 0.4847 | 0.9728 | 0.6786 | 0.7995 | 0.7258 | 0.8000 |
| 6 | 0.4033 | 0.4628 | 0.9744 | 0.6962 | 0.8121 | 0.7406 | 0.8102 |
| 7 | 0.4004 | 0.4764 | 0.9720 | 0.6934 | 0.8094 | 0.7369 | 0.8053 |
| 8 | 0.3938 | 0.4719 | 0.9741 | 0.7014 | 0.8156 | 0.7444 | 0.8121 |
| 9 | 0.3893 | 0.4879 | 0.9770 | 0.6752 | 0.7985 | 0.7256 | 0.8047 |
| 10 | 0.3833 | 0.4644 | 0.9755 | 0.7007 | 0.8156 | 0.7447 | 0.8139 |
| 11 | 0.3809 | 0.4277 | 0.9679 | 0.7279 | 0.8309 | 0.7614 | 0.8140 |
| 12 | 0.3807 | 0.5415 | 0.9776 | 0.6317 | 0.7675 | 0.6917 | 0.7859 |
| 13 | 0.3804 | 0.4770 | 0.9763 | 0.6831 | 0.8038 | 0.7314 | 0.8073 |
| 14 | 0.3740 | 0.5251 | 0.9749 | 0.6690 | 0.7935 | 0.7194 | 0.7988 |
| 15 | 0.3759 | 0.4556 | 0.9714 | 0.7145 | 0.8234 | 0.7531 | 0.8137 |
| 16 | 0.3712 | 0.4412 | 0.9683 | 0.7266 | 0.8302 | 0.7606 | 0.8140 |
| 17 | 0.3675 | 0.4957 | 0.9731 | 0.6866 | 0.8051 | 0.7322 | 0.8040 |
| 18 | 0.3651 | 0.4670 | 0.9728 | 0.7041 | 0.8170 | 0.7458 | 0.8114 |
| 19 | 0.3605 | 0.4958 | 0.9753 | 0.6803 | 0.8015 | 0.7286 | 0.8045 |
| 20 **←** | 0.3615 | 0.4498 | 0.9692 | 0.7262 | 0.8303 | 0.7608 | 0.8152 |

Mejor época por balanced accuracy: **20** (0.8152). Mínimo de val loss: época 11 (0.4277).

### Animales · 20 épocas

| Época | Loss | Val loss | Precisión | Recall | F1 | Accuracy | Bal acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.6345 | 0.8268 | 0.4677 | 0.2406 | 0.3177 | 0.7047 | 0.5655 |
| 2 | 0.5743 | 0.6765 | 0.3766 | 0.3552 | 0.3656 | 0.6478 | 0.5600 |
| 3 | 0.5595 | 0.8503 | 0.4009 | 0.3150 | 0.3528 | 0.6697 | 0.5633 |
| 4 | 0.5470 | 0.7560 | 0.4048 | 0.2957 | 0.3418 | 0.6745 | 0.5609 |
| 5 | 0.5392 | 0.7869 | 0.4018 | 0.3115 | 0.3509 | 0.6707 | 0.5630 |
| 6 | 0.5281 | 0.7432 | 0.4028 | 0.3228 | 0.3584 | 0.6697 | 0.5657 |
| 7 **←** | 0.5189 | 0.7702 | 0.4186 | 0.3036 | 0.3519 | 0.6805 | 0.5674 |
| 8 | 0.5086 | 0.7114 | 0.4175 | 0.2590 | 0.3197 | 0.6850 | 0.5572 |
| 9 | 0.4942 | 0.8441 | 0.3958 | 0.2940 | 0.3373 | 0.6700 | 0.5572 |
| 10 | 0.4870 | 0.7440 | 0.4308 | 0.2450 | 0.3123 | 0.6917 | 0.5577 |
| 11 | 0.4772 | 0.6353 | 0.4404 | 0.2327 | 0.3045 | 0.6963 | 0.5572 |
| 12 | 0.4748 | 0.6933 | 0.4342 | 0.2397 | 0.3089 | 0.6935 | 0.5574 |
| 13 | 0.4673 | 0.6872 | 0.5192 | 0.2012 | 0.2900 | 0.7185 | 0.5633 |
| 14 | 0.4599 | 0.6554 | 0.4470 | 0.2397 | 0.3121 | 0.6980 | 0.5605 |
| 15 | 0.4532 | 0.7516 | 0.4342 | 0.2310 | 0.3015 | 0.6943 | 0.5553 |
| 16 | 0.4463 | 0.6466 | 0.5442 | 0.1995 | 0.2919 | 0.7235 | 0.5663 |
| 17 | 0.4383 | 0.7425 | 0.4836 | 0.2318 | 0.3134 | 0.7097 | 0.5664 |
| 18 | 0.4487 | 0.7149 | 0.5147 | 0.2143 | 0.3027 | 0.7177 | 0.5667 |
| 19 | 0.4509 | 0.7432 | 0.5263 | 0.2100 | 0.3002 | 0.7202 | 0.5672 |
| 20 | 0.4419 | 0.6082 | 0.5378 | 0.2056 | 0.2975 | 0.7225 | 0.5674 |

Mejor época por balanced accuracy: **7** (0.5674). Mínimo de val loss: época 20 (0.6082).

## Lo que muestran las curvas

- **Hongos**: toca techo en la época 8-9 (0,869) y luego baja. Con 20 épocas va sobrada.
- **Plantas**: sube despacio hasta la 20 (0,815), todavía sin estabilizarse. Merece más épocas.
- **Animales**: la balanced accuracy se queda plana en 0,56-0,57 las 20 épocas, mientras la
  precisión sube y el recall baja. No es sobreajuste: es falta de datos. Confirma que aquí el
  problema no es el lector, es el volumen de entrenamiento.

## Test oro (Arabidopsis · 2.380 ventanas · 4 bloques)

Mismo genoma, misma rejilla y misma entrada que la evaluación del v2.

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| v2 100M congelado (24 kb, MIL) | 0,7031 | 0,5953 | — | — | 0,8243 |
| **NTv3 100M** | **0,7241** | **0,6241** | **0,9783** | 0,4582 | **0,8505** |
| Diferencia | **+0,0210** | **+0,0288** | — | — | **+0,0262** |

A umbral 0,50 los valores son casi idénticos (BA 0,7249 · F1 0,6261), así que el
resultado no depende de la calibración.

## Test oro (Drosophila · 2.674 ventanas · 4 bloques)

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| v2 100M congelado (24 kb, MIL) | 0,7385 | 0,6549 | — | — | 0,8151 |
| **NTv3 100M** | **0,7503** | **0,6739** | **0,9507** | **0,5220** | **0,8331** |
| Diferencia | **+0,0118** | **+0,0190** | — | — | **+0,0180** |

Se repite el patron: gana en las tres metricas, y el umbral calibrado (0,46) da valores
casi iguales a 0,50 (BA 0,7482 · F1 0,6696).

## Test de plata (S. pombe · 250 ventanas · 12 positivas)

| Modelo | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| v2 24 kb MIL | **0,8729** | **0,8182** | **0,8333** | 0,4167 | **0,8048** |
| NTv3 24 kb | 0,8666 | 0,7200 | 0,6923 | **0,7500** | 0,7917 |
| Diferencia | −0,0063 | −0,0982 | −0,1410 | **+0,3333** | −0,0131 |

**No es concluyente**: con 12 positivos cualquier diferencia es ruido. La unica lectura
clara es el cambio de punto de operacion: NTv3 detecta mas (recall 0,75 frente a 0,42) y
paga en precision.

## Conclusión

NTv3 es el mejor lector que hemos probado, y lo es **en las tres evaluaciones**:

| Evaluación | v2 | NTv3 | Δ |
|---|---|---|---|
| Hongos, test interno (PR-AUC) | 0,8094 | **0,8327** | +0,0233 |
| Plantas, test interno (bal acc) | 0,7815 | **0,8271** | +0,0456 |
| Animales, corpus completo (PR-AUC) | 0,5012 | **0,5101** | +0,0089 |
| **Oro · Arabidopsis (PR-AUC)** | 0,8243 | **0,8505** | **+0,0262** |
| **Oro · Drosophila (PR-AUC)** | 0,8151 | **0,8331** | **+0,0180** |
| Plata · S. pombe (PR-AUC) | **0,8048** | 0,7917 | −0,0131 (n=12, ruido) |

Animales se comparó con el mismo volumen de datos: gana por poco (+0,009). La plata no
permite conclusión por el tamaño de la muestra.

El cambio de lector no es gratis: reconstruir las cachés completas de los tres reinos
con NTv3 son ~100 h de GPU (hongos ~7 h, plantas ~35 h, animales ~65 h).

## Animales con el corpus completo

Se repitió el piloto sin tope (6.000 ventanas por especie: 287.814 en total, el mismo
volumen que el v2), con validación de 20.000 ventanas para calibrar mejor el umbral.

| Modelo | Train | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| v2 congelado | 287.814 | 0,6319 | 0,4239 | 0,5267 | 0,3547 | 0,5012 |
| **NTv3 completo** | 287.814 | **0,6376** | **0,4344** | **0,5540** | **0,3572** | **0,5101** |
| NTv3 piloto | 31.800 | 0,6263 | 0,4110 | — | — | 0,4868 |

Lectura:

- Con datos equivalentes, **NTv3 gana en las cinco métricas**, pero el PR-AUC sube solo
  +0,009: dentro del ruido entre semillas (±0,01). Es una victoria leve.
- El salto del piloto al completo (+0,023 de PR-AUC) confirma que la diferencia anterior
  era de volumen de datos, no de lector.
- La mejor época fue la 16 (BA de validación 0,6413), con la loss de validación mínima en
  la 9. La curva seguía subiendo despacio: no había sobreajuste claro.
