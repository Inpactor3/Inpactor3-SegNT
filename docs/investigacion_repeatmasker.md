# Investigación · RepeatMasker + biblioteca PanTEon

Documento de preparación para la reunión del miércoles. Resume lo que necesito
saber para correr RepeatMasker con la biblioteca PanTEon sobre los genomas de
test.

---

## 1. ¿Qué es RepeatMasker?

Software estándar en bioinformática para **anotar elementos repetitivos** en un
genoma. Dado un FASTA de genoma y una biblioteca de secuencias de referencia
(consensos de TEs), produce coordenadas donde cada secuencia de la biblioteca
aparece en el genoma.

**Entrada**: `genome.fasta` + `library.fasta` (los consensos).
**Salida**: `.out` (texto tabular), `.gff` (coordenadas estándar), `.masked`
(FASTA con regiones TE enmascaradas con N o lowercase).

**Página oficial**: https://www.repeatmasker.org

---

## 2. Instalación

Es software unix pesado. Dependencias:

- **BLAST+ o RMBlast** (motor de búsqueda).
- **TRF** (Tandem Repeat Finder).
- **Perl ≥ 5.8** con los módulos adecuados.
- Opcionalmente **HMMER** para búsquedas más sensibles.

**Opción fácil**: usar Docker o Conda.

```bash
# Via conda (bioconda)
conda install -c bioconda repeatmasker

# Verificar
RepeatMasker -h
```

**En el servidor sin sudo**: instalar en conda env del usuario.

---

## 3. Comando básico con biblioteca custom (PanTEon)

```bash
RepeatMasker \
    -lib /path/to/PanTEon_Database_v1.6.2.fasta \
    -pa 4 \
    -gff \
    -no_is \
    -dir /path/to/output \
    genome.fasta
```

**Flags clave**:
- `-lib`: ruta a la biblioteca custom (nuestra PanTEon).
- `-pa 4`: paralelismo (4 CPUs).
- `-gff`: genera también salida GFF (fácil de parsear).
- `-no_is`: no buscar Insertion Sequences bacterianas.
- `-dir`: directorio de salida.

---

## 4. Formato de salida

### Archivo `.out`

```
SW   perc perc perc  query     position in query    matching       repeat        position in repeat
score  div. del. ins. sequence    begin  end  (left) repeat         class/family  begin  end (left) ID
---------------------------------------------------------------------------------------------------
  312  18.5  0.0  0.0 chr1        1234   1532  (999)  PDB00001234    LTR/Gypsy     12    310  (5)  1
```

### Archivo `.out.gff`

```
chr1  RepeatMasker  similarity  1234  1532  312  +  .  Target "Motif:PDB00001234" 12 310
```

---

## 5. Convertir salida de RM a BED

```python
# Script simple
with open('genome.fasta.out.gff') as fi, open('annotations.bed', 'w') as fo:
    for line in fi:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split('\t')
        chrom = parts[0]
        start = int(parts[3]) - 1   # GFF is 1-based, BED is 0-based
        end = int(parts[4])
        name = parts[8].split('"')[1]  # Motif:PDB00001234
        score = parts[5]
        strand = parts[6]
        fo.write(f'{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n')
```

**Después** se cruza con metadata de PanTEon para añadir `orden` y `superfamilia`.

---

## 6. Tiempos esperados

Según documentación y experiencia reportada:

| Genoma | Tamaño | Tiempo RM (4 cpu) |
|---|---|---|
| *S. cerevisiae* | 12 Mb | ~5 min |
| *N. crassa* | 40 Mb | ~30 min |
| *A. thaliana* | 120 Mb | ~1-2 h |
| *C. elegans* | 100 Mb | ~1 h |
| *D. melanogaster* | 180 Mb | ~2-3 h |
| *O. sativa* | 370 Mb | ~5-8 h |
| *Z. mays* | 2.5 Gb | **~24-48 h** |

Con una biblioteca grande como PanTEon (240k secuencias), el tiempo **puede
multiplicarse**. Hay que medirlo en los pilotos.

---

## 7. Problema con bibliotecas gigantes

PanTEon tiene **240 k consensos**. RepeatMasker con `-lib` espera algo más
pequeño (típicamente <10 k).

**Opciones**:
1. **Filtrar la biblioteca por reino antes de correr** → 1 librería por reino.
2. **Filtrar por especie si tiene ≥30 consensos** → 1 librería por especie.

Según mi conteo del CSV de metadata:

| Reino | Especies | Consensos |
|---|---:|---:|
| Animalia | 780 | 101.144 |
| Fungi | 1.559 | 19.797 |
| Plantae | 451 | 119.005 |

**59.7% (1.667) de las especies tienen <30 consensos** → van a usar biblioteca
de reino.

---

## 8. Validación del piloto · comparar vs Dfam/rmsk

Para los 3 pilotos (Arabidopsis, Neurospora, C. elegans):

1. Correr `RepeatMasker -lib PanTEon ...` sobre cada genoma.
2. Descargar la anotación TE pre-computada:
   - Arabidopsis: TAIR10 TE annotation (https://www.arabidopsis.org/download/)
   - Neurospora: FungiDB o anotación del genoma de referencia.
   - C. elegans: WormBase TE annotation.
3. Comparar:
   - % total del genoma cubierto por TEs.
   - Top 10 familias por abundancia.
   - Overlap de coordenadas (bedtools intersect).

**Criterio de éxito del piloto**: overlap ≥70% con la anotación pre-computada
en los top 5 órdenes.

---

## 9. Alternativas si RepeatMasker es muy lento

Si en los pilotos vemos que RM con PanTEon toma >1 día por genoma grande:

- **EDTA** (https://github.com/oushujun/EDTA): más rápido, orientado a plantas.
- **RepeatModeler + RepeatMasker**: generar primero biblioteca específica.
- **Dfam + nhmmer**: motor HMM, más sensible pero más lento aún.

**No desviarse** del plan sin autorización del equipo + Simón.

---

## 10. Lo que necesito pedirle al equipo/Simón

- [ ] Confirmar que la biblioteca PanTEon que tenemos
      (`PanTEon_Database_v1.6.2.fasta`, 992 MB) es la versión correcta a usar.
- [ ] Saber si alguno del equipo ya corrió RepeatMasker con PanTEon antes
      (ahorrar aprendizaje).
- [ ] Confirmar que el servidor donde se corrió el entrenamiento SegNT tiene
      RM instalado (o permiso para instalarlo vía conda).
- [ ] Pedir recomendación: ¿bibliotecas por reino o por especie?

---

## 11. Script esqueleto para arrancar

Cuando se apruebe la estrategia, este será el primer script que escribo
(no toca el modelo, solo genera los BED):

```bash
#!/bin/bash
# scripts/annotate_with_repeatmasker.sh
# Uso: bash annotate_with_repeatmasker.sh plantae

KINGDOM=$1
LIB=data/panteon_libraries/${KINGDOM}.fasta
OUTDIR=annotations/${KINGDOM}

mkdir -p ${OUTDIR}

for GENOME in data/genomes/${KINGDOM}/*.fasta; do
    SPECIES=$(basename ${GENOME} .fasta)
    echo "=== Annotating ${SPECIES} ==="
    RepeatMasker -lib ${LIB} -pa 4 -gff -no_is \
                 -dir ${OUTDIR} ${GENOME}
    # Convertir .out.gff a BED
    python scripts/gff_to_bed.py \
        ${OUTDIR}/${SPECIES}.fasta.out.gff \
        ${OUTDIR}/${SPECIES}.bed
done
```

---

## 12. Resumen ejecutivo para el miércoles

1. **RepeatMasker con biblioteca PanTEon** = el paso crítico que nadie ha hecho.
2. **Hay que correrlo por reino** (no por especie) → 3 bibliotecas filtradas.
3. **El 59.7% de especies tienen <30 consensos** → fallback a reino es masivo.
4. **Tiempos**: 5 min (levadura) a 2 días (maíz) por genoma.
5. **Validación obligatoria**: 3 pilotos contra Dfam/rmsk antes de escalar.
6. **División de trabajo propuesta**: cada quien corre RM sobre un reino.
