# Inpactor3-SegNT

Detección de retrotransposones LTR con un **lector de ADN preentrenado y congelado** y un
cabezal pequeño entrenable.

**Proyecto Integrador II · Universidad de Caldas · 2026**

## Cómo funciona

1. Cada ventana del genoma aporta 4 bloques de 6 kb.
2. El lector (Nucleotide Transformer) convierte cada bloque en un vector (512 con v2, 768
   con NTv3). Se calcula **una sola vez** y se cachea en disco.
3. El cabezal MLP puntúa cada bloque y se queda con el máximo (MIL), o resume antes y
   puntúa una vez (pooled).
4. La salida es una probabilidad por ventana; el umbral se calibra en validación.

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install kaggle
```

Para el lector NTv3 (opcional) hace falta `transformers>=4.55`, que no convive con el
4.46 del v2. Se instala aparte y se activa por `PYTHONPATH`:

```bash
pip install --target ~/ntv3libs "transformers>=4.55" accelerate
export PYTHONPATH="$HOME/ntv3libs"
```

Guía completa para terceros: `paquete_descargas/GUIA_COMPANERO.md`.
Fuentes, formatos y errores resueltos: `docs/DESCARGAS_Y_ERRORES.md`.

## Datos

Bundles públicos en Kaggle:

| Dataset | Contenido |
|---|---|
| `cristianosoriog/inpactor2-hongos-ventanas` | hongos 24 kb y 100 kb |
| `cristianosoriog/inpactor2-plantas-ventanas` | plantas 24 kb y 100 kb |
| `cristianosoriog/inpactor2-animales-24k` / `-100k` | animales |
| `cristianosoriog/inpactor2-gold-ventanas` | genomas oro y cabezales |
| `cristianosoriog/inpactor2-plata-ventanas` | S. pombe |
| `cristianosoriog/inpactor2-mil-code` | scripts |
| `cristianosoriog/inpactor2-paquete-descargas` | código + guía + registro de errores |

## Uso rápido

```bash
kaggle datasets download -d cristianosoriog/inpactor2-mil-code -p pkg --unzip
kaggle datasets download -d cristianosoriog/inpactor2-hongos-ventanas -p vent --unzip

python pkg/mil_code/training/precompute_embeddings.py \
  --seq_dir vent/b24_bundle4_Hongos --out_dir trabajo/emb/Hongos \
  --splits val,test --per_block --trust_remote_code \
  --nt_model InstaDeepAI/nucleotide-transformer-v2-100m-multi-species

python pkg/mil_code/training/train_mil.py \
  --emb_dir trabajo/emb/Hongos --out_dir trabajo/modelo/Hongos \
  --epochs 20 --class_weight --balance global --pool logits
```

## Visor de resultados (solo lectura)

```bash
python3 visor/app.py --port 8000
# abrir http://127.0.0.1:8000
```

Lee los JSON de `docs/ventanas/`, `docs/ntv3_resultados.json` y los CSV de
`docs/ntv3_epocas/`. No ejecuta modelos ni expone pesos. Rutas: `/`, `/comparar`,
`/corrida/<id>`, `/epocas/<reino>`, `/salud`.

## Módulo integrable en Inpactor2

```bash
python -m inpactor3.detect --genome genoma.fa --head best.pt --out regiones.bed \
  --threshold 0.5 --window 50000 --chunks 4
```

Devuelve un BED (`cromosoma inicio fin inpactor3_p0.873 score .`) y, con `--gff`, un GFF3.
`--dry-run` cuenta ventanas sin cargar el modelo. Usa el mismo checkpoint y cabezal que el
resto del pipeline.

## Resultados de referencia

| Modelo | Reino | Bal acc | F1 | PR-AUC |
|---|---|---|---|---|
| v2, 50 kb | Hongos | 0,7766 | 0,7254 | 0,8434 |
| v2, 24 kb MIL | Hongos | 0,8261 | 0,7365 | 0,8094 |
| NTv3, 24 kb | Hongos | 0,8465 | 0,7122 | 0,8327 |
| NTv3, 24 kb | Plantas | 0,8271 | 0,8810 | 0,9732 |
| NTv3, test oro | Arabidopsis | 0,7241 | 0,6241 | 0,8505 |

## Estructura

```
training/    precompute, entrenamiento, evaluación, submuestreo
scripts/     generación de ventanas y bundles
data/        librerías de consensos y genomas de prueba
docs/        informe, experimentos y documentación técnica
paquete_descargas/  guía de inicio y scripts para terceros
_archivo/    presentaciones y material histórico (no forma parte del producto)
```

## Pruebas

```bash
python -m unittest discover -s tests -q
```

Cubren el cálculo de offsets (contra `training/block_layout`), la limpieza de secuencia y el
generador de ventanas. No requieren GPU ni descargar modelos.

Hook de protección (bloquea pesos, archivos >5 MB y secretos):

```bash
git config core.hooksPath .githooks
```

## Licencia, créditos y datos

- **Licencia:** GNU GPLv3 (`LICENSE`), heredada de Inpactor e Inpactor2.
- **Créditos y componentes de terceros:** `NOTICE.md`.
- **Procedencia y licencia de cada fuente de datos:** `docs/MANIFIESTO_DATOS.md`.
- Los lectores (Nucleotide Transformer v2 y NTv3) son de InstaDeep, NVIDIA y TUM, con
  licencia **no comercial**; los pesos no se redistribuyen.
- El repositorio no contiene pesos, librerías de terceros ni archivos grandes.
