# Propuesta de muestreo estratificado por longitud

Siguiendo la instrucción de Simón (chat 14/10): muestreo aleatorio estratificado por longitud (pb) para seleccionar especies de los 3 reinos, conservando la distribución de longitud entre splits.

## Metodología

1. **Excluir reinos no relevantes**: Ambiguo, No asignado, Otros eucariotas, Virus (según indicación previa de Simón).
2. **Filtrar especies con <1 kb** (demasiado pequeñas para generar ventanas).
3. **Objetivo por reino**: 150 especies cada uno (ajustable).
4. **Estratificar por cuartiles de longitud** dentro de cada reino (Q1=pequeñas, Q4=grandes).
5. **Muestreo proporcional** desde cada estrato.
6. **Split 70/15/15** (train/val/test) por estrato, para conservar distribución de longitud.
7. **Semilla fija 42** para reproducibilidad.

## Resumen por reino

| Reino | Total disponible | Muestreadas | Train | Val | Test |
|---|---:|---:|---:|---:|---:|
| Plantas | 563 | 151 | 107 | 24 | 20 |
| Animales | 1,805 | 149 | 105 | 24 | 20 |
| Hongos | 1,541 | 149 | 105 | 24 | 20 |

## Mediana de longitud (pb) por split (debe ser similar)

| Reino | Train | Val | Test |
|---|---:|---:|---:|
| Plantas | 856,922 | 1,065,489 | 1,044,708 |
| Animales | 33,622 | 55,288 | 63,510 |
| Hongos | 18,752 | 23,990 | 19,366 |

## Observaciones

- Si las medianas por split son similares (±20%), la estratificación por longitud fue correcta.
- El CSV completo con las especies seleccionadas está en `species_split_proposal.csv`.
- El número por reino (150) es ajustable: puede subir a 200-300 por reino si computacionalmente aguantamos.
