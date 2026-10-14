# Estado · Test "oro" con anotación curada

Reporte del intento de descarga de anotaciones TE curadas por la comunidad
(punto 1 del cierre de Simón en el mensaje del 12/10/2026).

## ✅ Anotaciones curadas que SÍ pudimos obtener

| Especie | Fuente | Features TE | Estado |
|---|---|---:|---|
| 🍄 **S. cerevisiae** | Ensembl Fungi (SGD) | **91 transposable_element** | ✅ **LISTO para test oro** |

Esta es la única anotación curada 100% funcional ahora mismo. Suficiente para
tener **1 test "oro" en el reino Fungi**.

## ⚠️ Anotaciones curadas que NO se pudieron descargar

| Especie | Fuente intentada | Problema |
|---|---|---|
| 🌱 Arabidopsis | TAIR10_Transposable_Elements.txt | 403 Forbidden (requiere cuenta) |
| 🌱 Oryza | rice.uga.edu MSU7 | URL discontinuada / 14 bytes |
| 🌱 Zea mays | MaizeGDB B73 | URL no probada |
| 🐟 Drosophila | FlyBase r6.55/r6.56 | SSL handshake failure |
| 🐟 C. elegans | WormBase | GFF Ensembl no tiene feature `transposable_element` |
| 🐟 Humano/Mus | Dfam full | Archivo HDF5 enorme, intento con hmm.gz dio archivo truncado |

## 🎯 Qué hacer para el miércoles

**Opción A** (recomendada): llevar el mensaje de que las URLs específicas de
cada curaduría tienen distintos problemas (login, SSL, rate-limit) y
**coordinar con el equipo quién baja cada una**.

**Opción B**: usar RM + **biblioteca Dfam estándar** como "ground truth
alternativo" (no es curación comunitaria, pero es más independiente que
RM + PanTEon). Requiere correr RM otra vez con `-species` en vez de `-lib`.

**Opción C**: aceptar que **solo S. cerevisiae tiene test "oro"** y los
demás se evalúan con test "plata" (RM + PanTEon mismo del train).

## 📋 División de trabajo propuesta para el miércoles

```
Jhonier    -> conseguir TAIR10_Transposable_Elements (TAIR login o mirror)
Alejandro  -> conseguir FlyBase TE curado (SSL alternativo)
Yo         -> ya tengo SGD listo + puedo coordinar Dfam
Todos      -> si no se logra, acordar la Opción C
```

## 🧰 Scripts listos para disparar

Con lo que ya tengo preparado, apenas se decidan las especies de test oro:

```bash
# Separar en splits ORO/PLATA/VAL/TRAIN
python3 scripts/prepare_test_gold.py \
    --oro "scerevisiae,drosophila,arabidopsis" \
    --plata "aspergillus,spombe" \
    --val "neurospora,celegans"

# Marcar regiones ambiguas como "ignorar" en el test
python3 scripts/generate_ignore_mask.py all
```

Ambos **YA están probados** y funcionando.
