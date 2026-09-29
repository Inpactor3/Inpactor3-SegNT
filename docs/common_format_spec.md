# Formato común de dataset — spec v1

Formato para entrenar tanto **Inpactor 2** como **Inpactor 3-SegNT** (y
cualquier otra arquitectura futura) sobre el mismo corpus derivado de
PanTEon, cubriendo **Plantae + Animalia + Fungi** simultáneamente.

## Diseño perezoso (lazy)

Con >100 000 LTR-RTs planteables, almacenar ventanas de 50 kb "materializadas"
en disco daría **cientos de GB** — inviable.

Solución: guardar un **manifiesto ligero** (JSONL con recetas por ventana),
y **materializar cada ventana bajo demanda** en el `Dataset` de PyTorch
durante el entrenamiento.

## Estructura de archivos

```
data/common_dataset/
├── manifest.jsonl              # una línea por ventana (~120 MB para 100k ventanas)
├── panteon_ltr_subset.fasta   # solo las secuencias LTR-RT usadas (~500 MB)
├── genomes/
│   ├── Zea_mays.fasta         # fondo plantas (~2.5 GB)
│   ├── Homo_sapiens.fasta     # fondo animales (~3 GB)
│   └── Neurospora_crassa.fa   # fondo hongos (~40 MB)
├── kingdom_index.tsv           # panteon_id → kingdom → genome_bg
└── lineage_map.tsv             # PanTEon superfamily → canonical id
```

## Formato del manifiesto (`manifest.jsonl`)

Una línea = una ventana. Cada línea es un JSON:

```json
{
  "window_id": "win_00000042",
  "kingdom": "Plantae",
  "background_genome": "Zea_mays",
  "background_scaffold": "chr3",
  "background_start": 15234000,
  "background_end": 15284000,
  "has_ltr_rt": true,
  "insertions": [
    {
      "panteon_id": "PDB00001234",
      "lineage": "GYPSY",
      "insert_position": 12000,
      "insert_length": 5432,
      "species_origin": "Zea_mays"
    }
  ],
  "split": "train"
}
```

Para ventanas negativas: `has_ltr_rt: false`, `insertions: []`.

## División del dataset (train / val / test)

División **por especie** (no aleatoria) para evitar data leakage:

- **train** (80% de PanTEon LTR): TEs de la mayoría de especies.
- **val** (10%): 45 especies apartadas.
- **test** (10%): otras 45 especies apartadas.
- **Test genomas reales** (afuera del manifiesto): Arabidopsis, arroz, piña, maíz completos.

## Formato per ventana (materializada)

Cuando el `Dataset` construye una ventana:

```
Ventana:  [ 50 000 bases uppercase A/C/G/T/N ]
Labels:   [ 50 000 enteros: 0 = fondo, 1..K = clase de LTR-RT ]
Metadata: kingdom, species_origin, panteon_id
```

## Etiquetas de clase

Al combinar los 3 reinos, tenemos superfamilias LTR de PanTEon:

| Class ID | Superfamilia | Prevalencia |
|---|---|---|
| 0 | fondo (no LTR) | mayoría |
| 1 | COPIA | plantas dominante |
| 2 | GYPSY | plantas dominante |
| 3 | ERV | animales dominante |
| 4 | BELPAO | animales |
| 5 | LARD | plantas |
| 6 | TRIM | plantas |

Total: **7 clases** (fondo + 6 superfamilias LTR).

## Consumidores del manifiesto

### Inpactor 2 (Detect binario)

Lee cada línea del manifiesto, devuelve:
- x: ventana one-hot (5, 50000)
- y: 0 o 1 según `has_ltr_rt`

### Inpactor 2 (Class multi-clase)

Solo lee líneas donde `has_ltr_rt=true`, extrae la subsecuencia del TE
insertado:
- x: subsecuencia one-hot
- y: clase de superfamilia

### Inpactor 3-SegNT

Lee cada línea, devuelve:
- x: ventana tokenizada (para NT)
- y: labels por token (0..6) según insertions

## Validación externa

Los 4 genomas de referencia se procesan por separado y se compara:

| Genoma | % LTR-RT esperado | Fuente |
|---|---|---|
| *Arabidopsis thaliana* | ~10% | literatura |
| *Oryza sativa* | ~22.5% | paper Inpactor 2 |
| *Ananas comosus* | ~23% | Orozco-Arias 2018 |
| *Zea mays* | ~75% | literatura |

El modelo se evalúa midiendo:
1. **% del genoma anotado como LTR-RT** vs valor esperado (± 5% aceptable).
2. **F1 por ventana** contra Inpactor 2 corrido sobre el mismo genoma.

## Ventajas del formato lazy

- Almacenamiento: **<4 GB** en disco (genomas + manifiesto + PanTEon subset)
  vs cientos de GB si se materializaran las ventanas.
- Reproducible: mismo manifiesto → mismas ventanas exactas (semilla fija).
- Flexible: puedes cambiar el tamaño de ventana sin regenerar todo.
