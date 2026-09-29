# Plan completo · PanTEon → Inpactor 2 + Inpactor 3-SegNT + validación

Pipeline para entrenar los dos modelos con el mismo dataset derivado de
**PanTEon completo** (Plantae + Animalia + Fungi) y validar sobre 4 genomas
reales (Arabidopsis, arroz, piña, maíz).

---

## Arquitectura del pipeline

```
PanTEon FASTA (240k TEs, 3 reinos)
    │
    │  scripts/build_teon_dataset.py
    ▼
data/common_dataset/
    ├── manifest.jsonl                (~120 MB, recetas de ~100k ventanas)
    ├── panteon_ltr_subset.fasta       (~500 MB, solo LTR-RTs)
    ├── genomes/                        (~5-6 GB, fondos por reino)
    │   ├── Zea_mays.fasta
    │   ├── Homo_sapiens.fasta
    │   └── Neurospora_crassa.fasta
    └── kingdom_index.tsv
    │
    ├──► scripts/export_inpactor2_format.py ──► Inpactor 2
    │
    └──► src/inpactor3_segnt/teon_dataset.py ──► Inpactor 3-SegNT
                                      │
                                      ▼
                    Modelos entrenados
                                      │
                                      ▼
    scripts/validate_on_genomes.py
                                      │
                                      ▼
      results/validation_report.md
      (% LTR-RT en 4 genomas vs esperado biológico)
```

---

## Fase 0 · Setup local (ya tienes)

- ✅ PanTEon descargado: `data/raw/PanTEon_Database_v1.6.2.fasta` (992 MB)
- ✅ Metadata: `data/raw/PanTEon_Database_metadata_v1.6.2.csv`
- ✅ Inpactor 2 con conda env funcional: `/home/cami/miniconda3/envs/Inpactor2`

## Fase 1 · Descargar genomas de fondo (~15 min, ~5 GB)

```bash
cd /home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt
bash scripts/download_backgrounds.sh
```

Baja **Zea mays** (fondo plantas), **Homo sapiens chr21+22** (fondo animales,
solo 2 cromosomas para ahorrar espacio) y **Neurospora crassa** (fondo hongos).

## Fase 2 · Construir el dataset común (~30 min)

```bash
python scripts/build_teon_dataset.py \
    --panteon data/raw/PanTEon_Database_v1.6.2.fasta \
    --metadata data/raw/PanTEon_Database_metadata_v1.6.2.csv \
    --genomes-dir data/common_dataset/genomes \
    --out-dir data/common_dataset \
    --n-positives 50000 \
    --n-negatives 50000
```

Produce:
- `data/common_dataset/manifest.jsonl` — ~100 000 recetas de ventanas
- `data/common_dataset/panteon_ltr_subset.fasta` — solo LTRs usados

## Fase 3 · Exportar al formato Inpactor 2

```bash
python scripts/export_inpactor2_format.py \
    --manifest data/common_dataset/manifest.jsonl \
    --genomes-dir data/common_dataset/genomes \
    --panteon-subset data/common_dataset/panteon_ltr_subset.fasta \
    --out-dir data/common_dataset/inpactor2_format \
    --split train

python scripts/export_inpactor2_format.py \
    --manifest data/common_dataset/manifest.jsonl \
    --genomes-dir data/common_dataset/genomes \
    --panteon-subset data/common_dataset/panteon_ltr_subset.fasta \
    --out-dir data/common_dataset/inpactor2_format \
    --split val
```

Produce:
- `Inpactor2_train_positive.fasta` — ventanas con LTR-RT
- `Inpactor2_train_negative.fasta` — ventanas sin LTR-RT
- `Inpactor2_class_train.fasta` — TE individuales con linaje en header

## Fase 4 · Entrenar Inpactor 2 con dataset común

⚠️ Requiere modificaciones al código de Inpactor 2 para aceptar FASTA
custom. El paper de Inpactor 2 usa la utilidad `Inpactor2_utils.py`
con `retrain_a_task()`. Se corre con:

```bash
source /home/cami/miniconda3/etc/profile.d/conda.sh
conda activate Inpactor2
cd /home/cami/Desktop/ProyectoIntegrador/Inpactor2

# Ver Inpactor2_utils.py línea ~180-220 para la función retrain_a_task
python -c "
import Inpactor2_utils as u
u.retrain_a_task(
    task='detect',
    fasta_pos='data/common_dataset/inpactor2_format/Inpactor2_train_positive.fasta',
    fasta_neg='data/common_dataset/inpactor2_format/Inpactor2_train_negative.fasta',
    out_dir='models/inpactor2_teon',
)
"
```

Tiempo estimado: **4-6 h en CPU** (Inpactor 2 no aprovecha GPU al máximo).

## Fase 5 · Entrenar Inpactor 3-SegNT con dataset común

Se necesita un `train_teon.py` que use `TeonWindowDataset`. Ver
`src/inpactor3_segnt/teon_dataset.py`. Config sugerida:

```yaml
# configs/nt_50m_teon.yaml
model:
  base_model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
  freeze_encoder: true          # solo entrena cabeza para reducir sobreajuste
  head_channels: [256, 128]
  num_classes: 7                # fondo + 6 superfamilias LTR
  dropout: 0.1

data:
  manifest: data/common_dataset/manifest.jsonl
  genomes_dir: data/common_dataset/genomes
  panteon_subset: data/common_dataset/panteon_ltr_subset.fasta
  window_size: 6000
  cell_size: 100

train:
  epochs: 20
  batch_size: 8
  learning_rate: 1e-4
  pos_weight: 50.0
  early_stopping_patience: 5
  seed: 42
```

Correr en Colab con GPU T4 (~4-6 h).

## Fase 6 · Descargar los 4 genomas de validación

```bash
mkdir -p data/validation_genomes
cd data/validation_genomes

# Arabidopsis thaliana (~120 MB descargados)
for CHR in 1 2 3 4 5; do
    curl -Lk -o "chr${CHR}.fa.gz" \
        "https://ftp.ensemblgenomes.org/pub/plants/release-58/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna_sm.chromosome.${CHR}.fa.gz"
    gunzip "chr${CHR}.fa.gz"
done
cat chr*.fa > Arabidopsis_thaliana.fasta && rm chr*.fa

# Oryza sativa (~370 MB)
curl -Lk -o Oryza_sativa.fa.gz \
    "https://ftp.ensemblgenomes.org/pub/plants/release-58/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa.gz"
gunzip Oryza_sativa.fa.gz && mv Oryza_sativa.fa Oryza_sativa.fasta

# Ananas comosus (~500 MB) — desde NCBI (Ensembl no lo tiene siempre)
curl -Lk -o Ananas_comosus.fa.gz \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/001/540/865/GCF_001540865.1_ASM154086v1/GCF_001540865.1_ASM154086v1_genomic.fna.gz"
gunzip Ananas_comosus.fa.gz && mv Ananas_comosus.fa Ananas_comosus.fasta

# Zea mays (~2.5 GB) — ya está en data/common_dataset/genomes/Zea_mays.fasta
cp ../common_dataset/genomes/Zea_mays.fasta Zea_mays.fasta

# Limpiar: uppercase + solo ACGTN
for F in *.fasta; do
    python3 -c "
with open('$F') as fi, open('${F}.tmp', 'w') as fo:
    for line in fi:
        if line.startswith('>'):
            fo.write(line)
        else:
            s = line.strip().upper()
            s = ''.join(c if c in 'ACGTN' else 'N' for c in s)
            fo.write(s + '\n')
"
    mv "${F}.tmp" "$F"
done
```

## Fase 7 · Correr Inpactor 2 sobre los 4 genomas

```bash
conda activate Inpactor2
cd /home/cami/Desktop/ProyectoIntegrador/Inpactor2

for GENOME in Arabidopsis_thaliana Oryza_sativa Ananas_comosus Zea_mays; do
    mkdir -p ../Inpactor3_segnt/results/inpactor2/${GENOME}
    python Inpactor2.py \
        -f ../Inpactor3_segnt/data/validation_genomes/${GENOME}.fasta \
        -o ../Inpactor3_segnt/results/inpactor2/${GENOME} \
        -a no -c yes
    cp ../Inpactor3_segnt/results/inpactor2/${GENOME}/Inpactor2_predictions.tab \
       ../Inpactor3_segnt/results/inpactor2/${GENOME}_predictions.tab
done
```

Tiempo estimado:
- Arabidopsis: ~30 min
- Arroz: ~2 h
- Piña: ~3 h
- Maíz: ~10-15 h (grande)

**Total: ~15-20 h de CPU**. Considerar hacerlo en un servidor.

## Fase 8 · Correr Inpactor 3-SegNT sobre los 4 genomas

```bash
mkdir -p results/segnt

for GENOME in Arabidopsis_thaliana Oryza_sativa Ananas_comosus Zea_mays; do
    python -m inpactor3_segnt.predict \
        --checkpoint models/best_teon.pt \
        --genome data/validation_genomes/${GENOME}.fasta \
        --out results/segnt/${GENOME}_predictions.tab
done
```

Con GPU: **~30 min por genoma** = ~2 h total.

## Fase 9 · Reporte final de validación

```bash
python scripts/validate_on_genomes.py \
    --predictions-dir results/inpactor2 \
    --genomes-dir data/validation_genomes \
    --model-name "Inpactor 2 (retrained on PanTEon)" \
    --report results/validation_inpactor2.md

python scripts/validate_on_genomes.py \
    --predictions-dir results/segnt \
    --genomes-dir data/validation_genomes \
    --model-name "Inpactor 3-SegNT (PanTEon)" \
    --report results/validation_segnt.md
```

Produce tablas como:

```
| Especie              | Esperado | Obtenido | Δ         |
|----------------------|----------|----------|-----------|
| Arabidopsis thaliana |   10.0%  |   X.X%   | +/- X ✅ |
| Oryza sativa         |   22.5%  |   X.X%   | +/- X ⚠️ |
| Ananas comosus       |   23.0%  |   X.X%   | +/- X ✅ |
| Zea mays             |   75.0%  |   X.X%   | +/- X ⚠️ |

MAE = X pp, RMSE = X pp
```

---

## Cronograma total

| Fase | Duración | Requiere |
|---|---|---|
| 1 · Descargar fondos | 15 min | Internet |
| 2 · Build dataset | 30 min | CPU |
| 3 · Export I2 | 10 min | CPU |
| 4 · Entrenar I2 | 4-6 h | CPU |
| 5 · Entrenar SegNT | 4-6 h | GPU (Colab) |
| 6 · Bajar validación | 30 min | Internet |
| 7 · I2 en 4 genomas | 15-20 h | CPU |
| 8 · SegNT en 4 genomas | 2 h | GPU |
| 9 · Reportes | 5 min | CPU |
| **Total** | **~30 h** | 1-2 días trabajo |

---

## Riesgos y mitigaciones

- **Disco insuficiente** (~15 GB): asegurar espacio antes de empezar.
- **Descarga de Homo sapiens lenta**: bajamos solo chr21+22 para acelerar.
- **Inpactor 2 crash con dataset gigante**: reducir `n_positives` a 20 000
  y `n_negatives` a 20 000 si peta.
- **Corrida sobre maíz muy lenta**: si es prohibitivo, evaluar solo sobre
  los cromosomas 1-3 del maíz (~800 Mb) en vez del genoma completo.

---

## Meta científica

Tener una **matriz comparativa** con ambos modelos sobre los 4 genomas:

```
                   Arabidopsis   Arroz    Piña      Maíz
Esperado             10%        22.5%    23%        75%
Inpactor 2           X%         X%       X%         X%
Inpactor 3-SegNT     X%         X%       X%         X%
```

Con esa tabla se puede argumentar objetivamente cuál generaliza mejor a
través de reinos y densidades muy distintas de LTR-RTs (10% a 75%).
