# Test "ORO" · Arabidopsis thaliana

**Verdad externa**: `TAIR10_Transposable_Elements.txt` (anotación curada TAIR, 31,189 TEs).

**Predicción evaluada**: `annotations/arabidopsis.bed` (RM + biblioteca PanTEon de Arabidopsis).

**Criterio de match**: IoU ≥ 0.5 con el elemento verdadero.

## Métricas globales

| Métrica | Valor |
|---|---:|
| **Precisión** | **0.2868** |
| **Recall** | **0.1973** |
| **F1** | **0.2338** |
| TP | 6,153 |
| FP | 15,301 |
| FN | 25,036 |
| TEs verdad (TAIR10) | 31,189 |
| TEs predicción (PanTEon) | 21,409 |

## Recall por superfamilia TAIR10

| Superfamilia | Verdad | Encontrados | Recall |
|---|---:|---:|---:|
| RC/Helitron | 12,945 | 1,221 | 0.094 |
| DNA/MuDR | 5,410 | 595 | 0.110 |
| LTR/Gypsy | 4,181 | 1,782 | 0.426 |
| DNA | 1,829 | 678 | 0.371 |
| LTR/Copia | 1,781 | 227 | 0.127 |
| LINE/L1 | 1,366 | 714 | 0.523 |
| DNA/HAT | 1,035 | 275 | 0.266 |
| DNA/En-Spm | 941 | 157 | 0.167 |
| DNA/Harbinger | 379 | 100 | 0.264 |
| DNA/Pogo | 344 | 64 | 0.186 |
| RathE1_cons | 213 | 0 | 0.000 |
| DNA/Mariner | 151 | 127 | 0.841 |
| SINE | 131 | 83 | 0.634 |
| Unassigned | 129 | 12 | 0.093 |
| RathE3_cons | 104 | 0 | 0.000 |
| DNA/Tc1 | 95 | 64 | 0.674 |
| LINE? | 81 | 0 | 0.000 |
| RathE2_cons | 74 | 54 | 0.730 |

## Interpretación

- **Recall de 19.7%**: de los TEs curados TAIR10, encontramos 6,153 de 31,189.
- **Precisión de 28.7%**: de lo que marcamos, 6,153 de 21,409 son TE reales (según TAIR).
- **F1 global: 0.2338**.

## Comparación con el modelo previo

| Comparación | F1 |
|---|---:|
| **ANOTACIÓN** (RM+PanTEon vs TAIR10) | **0.234** |
| MODELO SegNT (previo) en test balanceado PanTEon | 0.670 |
| MODELO SegNT (previo) en Arabidopsis chr5 real | 0.080 |

Esto nos dice qué tan buena es la **ANOTACIÓN** sobre la que entrenaremos.
Si F1-anotación es alto, el ground truth es sólido y el modelo nuevo deberá acercarse.
