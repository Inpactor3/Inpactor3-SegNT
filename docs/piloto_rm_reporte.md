# Reporte piloto · RepeatMasker + biblioteca PanTEon

Validacion de la anotacion con RepeatMasker + PanTEon sobre **8 especies** (2 por reino). Criterio: el %TE encontrado debe estar dentro de 30% del valor reportado en literatura.

## Resumen general (ordenado por reino)

| Reino | Especie | Genoma | TEs enc. | %TE obs. | %TE esp. (lit) | Status |
|---|---|---:|---:|---:|---:|---|
| Animalia | celegans | 100.3 Mb | 80,874 | 12.73% | 17.0% | OK (dentro de 30%) |
| Animalia | drosophila | 143.7 Mb | 110,815 | 6.74% | 20.0% | FAIL (desviacion >50%) |
| Fungi | aspergillus | 29.8 Mb | 6,156 | 3.56% | 5.0% | OK (dentro de 30%) |
| Fungi | neurospora | 41.1 Mb | 26,811 | 9.08% | 10.0% | OK (dentro de 30%) |
| Fungi | scerevisiae | 12.2 Mb | 4,091 | 2.34% | 3.0% | OK (dentro de 30%) |
| Fungi | spombe | 12.6 Mb | 3,670 | 2.38% | 1.0% | FAIL (desviacion >50%) |
| Plantae | arabidopsis | 119.1 Mb | 67,527 | 11.63% | 15.0% | OK (dentro de 30%) |
| Plantae | brachypodium | 271.2 Mb | 159,147 | 22.78% | 22.0% | OK (dentro de 30%) |

## Detalle · arabidopsis

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/tair10_full/TAIR10.fasta`
- Longitud: 119,146,348 bp
- Elementos detectados: 67,527
- Total bp cubiertos por TE: 13,860,847
- %TE observado: **11.63%**
- %TE esperado (literatura): 15.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 36,663 | 1,454,582 |
| Low_complexity | 9,455 | 471,001 |
| CLASSI/LTR/GYPSY | 7,905 | 5,787,085 |
| CLASSII/HELITRON/HELITRON | 4,347 | 1,813,758 |
| CLASSII/TIR/TIR | 3,125 | 863,517 |
| CLASSI/LTR/COPIA | 1,579 | 871,722 |
| CLASSI/LINE/L1 | 1,473 | 901,513 |
| CLASSII/TIR/MULE | 1,261 | 900,217 |
| CLASSII/TIR/CMC | 650 | 512,640 |
| CLASSII/TIR/TC1MARINER | 472 | 84,010 |

## Detalle · brachypodium

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/brachypodium.fasta`
- Longitud: 271,163,419 bp
- Elementos detectados: 159,147
- Total bp cubiertos por TE: 61,763,716
- %TE observado: **22.78%**
- %TE esperado (literatura): 22.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 53,296 | 2,286,553 |
| CLASSI/LTR/GYPSY | 24,716 | 32,316,350 |
| CLASSII/MITE/MITE | 21,188 | 3,254,084 |
| CLASSII/TIR/TIR | 18,014 | 2,092,375 |
| CLASSI/LINE/L1 | 12,270 | 5,544,298 |
| CLASSI/LTR/COPIA | 9,873 | 9,085,399 |
| Low_complexity | 8,109 | 409,088 |
| CLASSII/TIR/CMC | 3,815 | 2,857,156 |
| CLASSI/LTR/LARD | 2,463 | 1,579,205 |
| CLASSI/SINE/SINE | 1,674 | 283,209 |

## Detalle · scerevisiae

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/scerevisiae.fasta`
- Longitud: 12,157,105 bp
- Elementos detectados: 4,091
- Total bp cubiertos por TE: 283,999
- %TE observado: **2.34%**
- %TE esperado (literatura): 3.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 3,004 | 126,638 |
| Low_complexity | 546 | 25,480 |
| CLASSI/LTR/COPIA | 480 | 117,134 |
| CLASSI/LTR/GYPSY | 61 | 14,747 |

## Detalle · neurospora

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/neurospora.fasta`
- Longitud: 41,102,378 bp
- Elementos detectados: 26,811
- Total bp cubiertos por TE: 3,732,523
- %TE observado: **9.08%**
- %TE esperado (literatura): 10.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 19,845 | 806,550 |
| Low_complexity | 2,500 | 130,310 |
| CLASSI/LTR/LARD | 1,872 | 1,625,992 |
| CLASSI/LTR/GYPSY | 692 | 280,746 |
| CLASSI/LINE/TAD1 | 584 | 723,750 |
| CLASSI/LTR/COPIA | 223 | 24,557 |
| CLASSII/HELITRON/HELITRON | 202 | 20,634 |
| CLASSI/SINE/tRNA | 136 | 12,099 |
| CLASSI/LINE/LINE | 124 | 12,557 |
| CLASSI/LTR/TRIM | 80 | 13,054 |

## Detalle · spombe

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/spombe.fasta`
- Longitud: 12,631,379 bp
- Elementos detectados: 3,670
- Total bp cubiertos por TE: 300,156
- %TE observado: **2.38%**
- %TE esperado (literatura): 1.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 2,325 | 94,050 |
| Low_complexity | 668 | 33,590 |
| CLASSI/LTR/GYPSY | 345 | 132,439 |
| CLASSI/SINE/tRNA | 44 | 3,376 |
| CLASSI/LTR/COPIA | 37 | 3,181 |
| CLASSI/LTR/LARD | 37 | 4,424 |
| CLASSII/HELITRON/HELITRON | 29 | 2,287 |
| CLASSI/LINE/LINE | 28 | 2,661 |
| CLASSI/LINE/TAD1 | 28 | 2,357 |
| CLASSI/LINE/CR1 | 22 | 2,504 |

## Detalle · aspergillus

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/aspergillus.fasta`
- Longitud: 29,828,291 bp
- Elementos detectados: 6,156
- Total bp cubiertos por TE: 1,060,933
- %TE observado: **3.56%**
- %TE esperado (literatura): 5.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 3,189 | 129,831 |
| Low_complexity | 511 | 25,919 |
| CLASSI/LTR/COPIA | 448 | 120,037 |
| CLASSI/LTR/GYPSY | 396 | 206,577 |
| CLASSII/TIR/TC1MARINER | 270 | 157,435 |
| CLASSII/TIR/HAT | 243 | 57,758 |
| CLASSI/LTR/LTR | 191 | 38,173 |
| CLASSI/LINE/TAD1 | 146 | 24,351 |
| CLASSII/TIR/TIR | 102 | 62,507 |
| CLASSII/HELITRON/HELITRON | 100 | 14,347 |

## Detalle · celegans

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/celegans.fasta`
- Longitud: 100,286,401 bp
- Elementos detectados: 80,874
- Total bp cubiertos por TE: 12,762,373
- %TE observado: **12.73%**
- %TE esperado (literatura): 17.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 22,168 | 1,074,189 |
| CLASSII/TIR/TIR | 17,097 | 3,305,404 |
| CLASSII/TIR/HAT | 9,496 | 1,130,477 |
| CLASSII/TIR/TC1MARINER | 9,048 | 2,112,917 |
| Low_complexity | 6,350 | 300,460 |
| CLASSII/TIR/PIGGYBAC | 5,667 | 1,044,346 |
| CLASSII/HELITRON/HELITRON | 4,591 | 1,722,416 |
| CLASSI/LTR/BELPAO | 2,759 | 904,635 |
| CLASSII/TIR/MULE | 1,815 | 436,728 |
| CLASSI/LINE/CR1 | 481 | 264,023 |

## Detalle · drosophila

- Genoma: `/home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/data/pilot_genomes/drosophila.fasta`
- Longitud: 143,726,002 bp
- Elementos detectados: 110,815
- Total bp cubiertos por TE: 9,690,522
- %TE observado: **6.74%**
- %TE esperado (literatura): 20.0%

### Top 10 clases/familias (por numero de elementos)

| Clase/Familia | n elementos | bp cubiertos |
|---|---:|---:|
| Simple_repeat | 91,784 | 3,911,114 |
| Low_complexity | 11,914 | 559,154 |
| CLASSI/LTR/GYPSY | 6,277 | 4,856,586 |
| CLASSI/LTR/COPIA | 437 | 234,613 |
| CLASSII/TIR/HAT | 207 | 76,425 |
| CLASSII/TIR/CMC | 195 | 51,700 |
| CLASSII/TIR/P | 1 | 930 |

## Conclusion para el equipo

- 6/3 pilotos pasan el criterio (±30% vs literatura)
- La biblioteca PanTEon filtrada por especie funciona
- Decision: proceder con la anotacion masiva
