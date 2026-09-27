# Evaluación cromosoma Chr5

- Longitud: **26,975,502** bp
- Ventanas de 50,000 bp: **540**
- Ground truth (Inpactor2): 23 elementos, 21 ventanas con LTR-RT
- Predicciones (Inpactor3-SegNT): 7 elementos, 4 ventanas marcadas

## Matriz de confusión (ventanas)

|              | Predijo vacío | Predijo LTR-RT |
|--------------|--------------:|---------------:|
| **Vacío (519)**    | 516 | 3 |
| **LTR-RT (21)**   | 20 | 1 |

## Métricas de presencia (ventana)

- **Recall**     = 0.048  (1/21)
- **Precisión**  = 0.250  (1/4)
- **F1**         = 0.080

## Precisión de coordenadas (IoU 1D, umbral=0.5)

- IoU medio (por elemento verdadero): **0.000**
- Elementos con IoU ≥ 0.5: **0/23** (0.0%)

## Confusión de linaje por ventana positiva

| ↓verdad \ pred → | background | RLC/ALE/RETROFIT |
|---|---|---|
| **RLC/ALE/RETROFIT** | 3 | 0 |
| **RLC/BIANCA** | 3 | 0 |
| **RLC/IVANA/ORYCO** | 3 | 0 |
| **RLC/TAR/TORK** | 1 | 0 |
| **RLG/REINA** | 1 | 0 |
| **RLG/TEKAY/DEL** | 1 | 0 |
| **RLG/ATHILA** | 6 | 1 |
| **RLG/TAT** | 2 | 0 |

- **Accuracy de linaje** (sobre ventanas TP): **0.000** (0/21)

## Referencia (informe interno, chr5 Arabidopsis)

| Enfoque | F1 |
|---|---|
| Fondo aleatorio uniforme | 0-7% |
| Fondo otro transposón | 7.5% |
| Fondo genoma real (informe intento 3) | **26.3%** |
| **Este modelo (SegNT)** | **8.0%** |
