# Evaluación cromosoma Chr1_test

- Longitud: **3,000,000** bp
- Ventanas de 50,000 bp: **60**
- Ground truth (Inpactor2): 3 elementos, 3 ventanas con LTR-RT
- Predicciones (Inpactor3-SegNT): 30 elementos, 24 ventanas marcadas

## Matriz de confusión (ventanas)

|              | Predijo vacío | Predijo LTR-RT |
|--------------|--------------:|---------------:|
| **Vacío (57)**    | 36 | 21 |
| **LTR-RT (3)**   | 0 | 3 |

## Métricas

- **Recall**     = 1.000  (3/3)
- **Precisión**  = 0.125  (3/24)
- **F1**         = 0.222

## Referencia (informe interno, chr5 Arabidopsis)

| Enfoque | F1 |
|---|---|
| Fondo aleatorio uniforme | 0-7% |
| Fondo otro transposón | 7.5% |
| Fondo genoma real (informe intento 3) | **26.3%** |
| **Este modelo (SegNT)** | **22.2%** |
