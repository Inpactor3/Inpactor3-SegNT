# Descargas, formatos y errores encontrados

Registro de todo lo que se descargó para el proyecto, con qué script, en qué formato, y los
errores que aparecieron con su solución. Sirve para reproducir el pipeline y para no volver
a tropezar con lo mismo.

Última actualización: 2026-11-18

---

## 1. Fuentes de datos

| Fuente | URL base | Formato | Qué se descarga | Para qué |
|---|---|---|---|---|
| Kaggle CLI | `https://www.kaggle.com/datasets/<usuario>/<dataset>` | ZIP (y NPZ dentro) | Bundles de secuencias, resultados, checkpoints | Datos y artefactos del proyecto |
| NCBI Datasets | `https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/<ACC>/download` | ZIP → FASTA `.fna` | Genomas de referencia | Especies nuevas |
| UCSC | `https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz` | FASTA `.gz` | Genoma de Drosophila | Test oro |
| UCSC | `https://hgdownload.soe.ucsc.edu/goldenPath/dm6/database/rmsk.txt.gz` | TXT tabulado | Anotación RepeatMasker | Test oro |
| Ensembl (EBI) | `https://ftp.ebi.ac.uk/ensemblgenomes/pub/fungi/current/gff3/` | GFF3 `.gz` | Genes anotados de hongos | Elementos Tf2 de *S. pombe* |
| Dfam | `https://www.dfam.org/releases/current/families/FamDB/` | HDF5 `.h5` | Librerías de consensos | RepeatMasker genérico |
| HuggingFace | `https://huggingface.co/InstaDeepAI/<modelo>` | Safetensors + JSON + `.py` | Pesos de lectores (v1, v2, NTv3) | Backbone congelado |
| Disco USB | `/media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM` | FASTA `.fna.gz` + BED | Corpus de 400 genomas anotados | Entrenamiento |
| Librerías locales | `Inpactor3_segnt/data/panteon_libraries/` | FASTA (consensos) | 16 librerías del proyecto | Anotar especies nuevas |

---

## 2. Scripts usados

| Script | Ruta en el repo | Qué hace |
|---|---|---|
| Descarga de datasets de Kaggle | `servidor_inpactor/bajar_datos.sh` | Baja todos los datasets del proyecto |
| Precompute de embeddings v2 | `training/precompute_embeddings.py` | Lector v2 → vectores por bloque (512) |
| Precompute de embeddings NTv3 | `training/precompute_ntv3.py` | Lector NTv3 → vectores por bloque (768) |
| Entrenamiento del cabezal | `training/train_mil.py` | MIL / pooled sobre la caché |
| Evaluación en oro | `training/evaluate_gold.py` | Genoma + BED → métricas |
| Evaluación NTv3 en oro | `training/gold_ntv3.py` | Cabezal NTv3 sobre ventanas del oro |
| Submuestreo 1:1 | `training/subsample_train.py` | Balancea y aplica tope por especie |
| Conversión RepeatMasker → BED | `training/rm_out_to_bed.py` | `.out` de RM a BED con clase |
| Generación de ventanas | `scripts/build_windows_var.py` | FASTA sizes + BED → JSONL etiquetado |
| Empaquetado de bundles | `training/make_sequence_bundle.py` | JSONL + genomas → NPZ |
| Bundles con tamaño variable | `scripts/build_bundle_size.py` | Orquesta lo anterior para un reino y ventana |
| Lanzadores | `training/piloto_ntv3.sh`, `training/lora_piloto.sh`, `training/gold_ntv3.sh` | Ejecutan cada experimento de punta a punta |
| Test oro en Modal | `training/gold_modal.py` | Evalúa en la nube (ya sin créditos) |

---

## 3. Errores y soluciones

### 3.1 Entorno y credenciales

**`kaggle: no se encontró la orden`**

- Causa: el CLI vive dentro del venv del proyecto.
- Solución: activar antes de cada comando.
  ```bash
  cd ~/Escritorio/Inpactor3 && source .venv/bin/activate
  ```
  Alternativa sin activar: `~/.venvs/kaggle/bin/kaggle ...`

**`401 Unauthorized` / `GatedRepoError: You are trying to access a gated repo`**

- Causa: el modelo está restringido y el proceso no ve el token, o no se aceptó la licencia.
- Solución:
  1. Aceptar la licencia en la página del modelo (botón *Agree and access repository*).
  2. Guardar el token en el archivo de credenciales:
     ```bash
     unset HF_TOKEN
     mkdir -p ~/.cache/huggingface
     read -rsp "token: " T; echo
     printf '%s' "$T" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
     ```
- Verificación:
  ```bash
  python -c "from huggingface_hub import whoami; print(whoami()['name'])"
  ```

**`Invalid user token. The token from HF_TOKEN environment variable is invalid. Note that HF_TOKEN takes precedence over hf auth login`**

- Causa: había una variable `HF_TOKEN` exportada con un valor viejo o de ejemplo, y **manda sobre el archivo**.
- Solución: `unset HF_TOKEN` en esa terminal antes de ejecutar.

**Conflicto de versiones de `transformers`**

- NTv3 exige `transformers >= 4.55`; el pipeline v2 está clavado a `4.46.3`.
- Solución: no instalar en el venv del proyecto. Se superpone en una carpeta:
  ```bash
  source .venv/bin/activate
  pip install --target ~/ntv3libs "transformers>=4.55" accelerate
  export PYTHONPATH="$HOME/ntv3libs"
  ```

**RTX 5090: `no kernel image is available`**

- Causa: la 5090 es Blackwell (`sm_120`) y necesita CUDA 12.8.
- Solución: `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

**`torch.load(...)` falla con `UnpicklingError` / pide `weights_only`**

- Causa: desde torch 2.6 el valor por defecto es `weights_only=True` y nuestros checkpoints guardan numpy.
- Solución: `torch.load(path, map_location=dev, weights_only=False)`.

---

### 3.2 Descargas

**`unzip: cannot find zipfile directory` o `Zip archive data, at least v4.5 to extract`**

- Causa: el ZIP generado por NCBI Datasets usa una versión que el `unzip` del sistema no abre.
- Solución: extraer con Python.
  ```python
  import zipfile; zipfile.ZipFile("archivo.zip").extractall("salida")
  ```

**La descarga de Kaggle va a 0,6 MB/s y tardaría horas**

- Causa: `--dir-mode zip` recomprime archivos que ya están comprimidos.
- Solución: subir con `--dir-mode tar` (sin compresión).
  ```bash
  kaggle datasets create -p ./carpeta --dir-mode tar
  ```

**La subida se queda estancada (2 MB en cola, 7 KB/s)**

- Causa: pérdida de paquetes en la red, no el script.
- Solución: reintentar cuando mejore la conexión y dividir en datasets más pequeños.
  ```bash
  pgrep -f "datasets create" && echo subiendo
  # relanzar solo el paquete pequeño primero
  ```

**`Access to model ... is restricted and you are not in the authorized list`**

- Causa: el token es válido pero la cuenta no aceptó la licencia.
- Solución: pedir acceso en la web con la misma cuenta del token; no hay API para eso.

**`curl` no resuelve el host (`Temporary failure in name resolution`)**

- Causa: DNS intermitente del servidor.
- Solución: reintentar; si persiste,
  ```bash
  getent hosts huggingface.co
  sudo resolvectl flush-caches 2>/dev/null || sudo systemctl restart systemd-resolved
  ```

**El bucle de descargas crea un archivo `.zip` vacío**

- Causa: al lanzar comandos en segundo plano, las variables del `for` se expandían dos veces y quedaban vacías.
- Solución: poner el bucle en un archivo `.sh` y ejecutar el archivo, no un one-liner.
  ```bash
  bash /tmp/dl_nuevas.sh
  ```

---

### 3.3 Formato de las anotaciones

**Las ventanas salían con etiquetas que no cuadraban con el corpus**

- Causa: el corpus etiqueta **solo elementos LTR**, y estábamos contando también transposones de ADN (CLASSII).
- Solución: filtrar el BED por clase antes de etiquetar.
  ```python
  if "/LTR/" not in nombre and not nombre.startswith("LTR/"): continue
  ```
- Validación: regenerar a 50 kb y comparar con el JSONL del corpus → **0 diferencias en 1.096 ventanas**.

**El BED de Drosophila salía desplazado una columna**

- Causa: `rmsk.txt` trae una columna extra (`swScore`) al inicio.
- Columnas correctas: `$6` cromosoma, `$7` inicio, `$8` fin, `$11` nombre, `$12` clase, `$13` familia, `$10` hebra.
  ```bash
  zcat rmsk.txt.gz | awk -F'\t' 'BEGIN{OFS="\t"}{print $6,$7,$8,$11"|"$12"/"$13,0,$10}'
  ```

**El generador de ventanas tardaba horas con genomas grandes**

- Causa: algoritmo cuadrático (recorría todos los TE en cada ventana).
- Solución: barrido lineal con puntero por cromosoma (`scripts/build_windows_var.py`).

**Kaggle descomprime archivos y cambia la estructura**

- Ejemplos vistos:
  - `TAIR10.fasta.gz` → `gold/TAIR10.fasta/TAIR10.fasta`
  - `dm6.fa.gz` → `gold/dm6.fa`
  - bundles empaquetados con `--dir-mode zip` → carpetas ya extraídas
- Solución: que los lanzadores prueben varias rutas antes de fallar.
  ```bash
  primero() { for p in "$@"; do [ -f "$p" ] && { echo "$p"; return; }; done; }
  ```

**`ValueError: need at least one array to concatenate`**

- Causa: un `.npz` del bundle tenía 0 ventanas.
- Solución: saltar archivos vacíos y listas vacías en `precompute_ntv3.py`.
  ```python
  if N == 0: print("SKIP vacio"); continue
  ```

**`ventanas de validación: 0`**

- Causa: el recorte de validación encontraba enlaces simbólicos con el mismo nombre y los saltaba.
- Solución: borrar el directorio completo antes de escribirlo.
  ```python
  if os.path.islink(dst): os.unlink(dst)
  elif os.path.isdir(dst): shutil.rmtree(dst)
  os.makedirs(dst)
  ```

---

### 3.4 RepeatMasker

**`cannot read file /tmp/... in /opt/src`**

- Causa: se pasó la ruta del host al contenedor, que monta el volumen en `/data`.
- Solución: usar la ruta interna.
  ```bash
  docker run --rm -v "$DEST":/data -v "$LIBS":/libs:ro dfam/tetools:latest \
    /opt/RepeatMasker/RepeatMasker -pa 3 -lib "/libs/panteon_drosophila.fasta" \
    -dir "/data/rm/nombre" "/data/ruta/interna/genoma.fna"
  ```

**La librería curada de Dfam no encuentra TEs en hongos**

- Causa: la partición *curated* no incluye familias fúngicas (solo simple/low-complexity).
- Solución: usar la partición **no curada** (`dfam40.uncurated.consensus.*`), o mejor, las
  librerías del proyecto (`data/panteon_libraries`), que son las que generaron los BED del corpus.

**`ftp.ensemblgenomes.org` no resuelve**

- Solución: usar el espejo de EBI, `https://ftp.ebi.ac.uk/ensemblgenomes/`.

---

### 3.5 Cómputo

**`torch.OutOfMemoryError` en T4 con `token_batch 128`**

- Solución: `--batch_size 8 --token_batch 32` en GPU de 16 GB; 64/128 solo en 32 GB.

**Kernel de Kaggle cortado a las 9 h**

- Solución: dividir por reinos o por shards (`--num_shards` / `--shard_index`).

**Modal: `RemoteError` al agotarse los créditos**

- Causa: sin saldo, el cliente mata los procesos remotos.
- Solución: mover el trabajo a Kaggle o al servidor. Los volúmenes quedan, pero no se
  pueden leer sin cuenta activa.

**`Function.map` en Modal no ejecuta nada**

- Causa: devuelve un iterador perezoso.
- Solución: recorrerlo (`for r in f.map(...)`).

---

## 4. Formatos

| Extensión | Qué es | Campos / estructura | Se genera con | Se consume en |
|---|---|---|---|---|
| `.fna`, `.fa`, `.fasta` | Secuencia de ADN | `>nombre` y líneas de ACGT | Descarga NCBI/UCSC | `build_windows_var.py`, `evaluate_gold.py` |
| `.fna.gz` | Igual, comprimido | — | Discos y descargas | `make_sequence_bundle.py` |
| `.bed` | Anotación por intervalos | `cromosoma  inicio  fin  nombre|clase/familia  0  hebra` | `rm_out_to_bed.py`, conversión de rmsk | Etiquetado de ventanas |
| `.gff3` | Anotación de genes | 9 columnas; atributos `Name=`, `ID=` | Ensembl | Extracción de Tf2 |
| `.txt` (rmsk) | Tabla RepeatMasker de UCSC | 17 columnas con `swScore` al inicio | UCSC | Conversión a BED |
| `.out` (RM) | Salida de RepeatMasker | 15 columnas; hebra `C` = negativa | RepeatMasker | `rm_out_to_bed.py` |
| `.jsonl` | Una ventana por línea | `window_id, species, chrom, start, end, label, n_TE, bp_TE...` | `build_windows_var.py` | `make_sequence_bundle.py` |
| `.npz` | Bundle comprimido | `seq (N,C,6000) uint8`, `label`, `window_id`, `offsets`, `chunk_bp` | `make_sequence_bundle.py` | precompute, entrenamiento |
| `_emb.npy` | Caché de embeddings | `(N, C, H)` float16 — H=512 (v2) o 768 (NTv3) | precompute | `train_mil.py` |
| `.h5` | Librería FamDB de Dfam | Particiones por taxón | Descarga Dfam | `famdb.py`, RepeatMasker |
| `.zip` / `.tar` | Empaquetado | — | Kaggle CLI | Descarga |
| `.pt` | Checkpoint | `head`, `hidden`, `mu`, `sd`, `epoch`, `mode`, `threshold` | `train_mil.py`, `lora_hongos.py` | Evaluación |

---

## 5. Checklist antes de descargar algo nuevo

1. **Formato**: ¿FASTA, BED, GFF3 o tabla? Si es anotación, comprobar que la clase LTR esté
   en el nombre (`...|LTR/Gypsy`).
2. **Ensamblado**: que el genoma y el BED sean de la **misma versión** (p. ej. dm6 con
   rmsk de dm6).
3. **Licencia y acceso**: si es gated, aceptar en la web y guardar el token en
   `~/.cache/huggingface/token`.
4. **Tamaño**: si supera 5 GB, valorar `--dir-mode tar` en Kaggle y dividir por reinos.
5. **Estructura tras descomprimir**: Kaggle y `gunzip` cambian rutas; probar varias antes
   de fallar.
6. **Etiquetado**: filtrar solo LTR y validar contra un JSONL conocido antes de escalar.
7. **Nombres de especie**: el corpus usa dobles o triples guiones bajos
   (`Sp__GCA_...` / `Sp___GCA_...`). Normalizar con `re.sub(r"_+", "_", nombre)`.
8. **Registrar la fuente**: anotar URL, fecha y comando en este documento.

---

## 6. Comandos de referencia rápida

```bash
# Kaggle: bajar, crear, versionar
kaggle datasets download -d usuario/dataset -p destino --unzip
kaggle datasets create  -p ./carpeta --dir-mode tar
kaggle datasets version -p ./carpeta -m "mensaje" --dir-mode skip

# HuggingFace: verificar acceso y descargar pesos
python -c "from huggingface_hub import hf_hub_download as d; print(d('InstaDeepAI/NTv3_100M_pre','config.json'))"
hf download InstaDeepAI/NTv3_100M_pre --local-dir ~/modelos/ntv3_100m

# NCBI: genoma por accesión
curl -L -o g.zip "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000002985.6/download?include_annotation_type=GENOME_FASTA"
python -c "import zipfile; zipfile.ZipFile('g.zip').extractall('genoma')"

# UCSC: genoma y RepeatMasker de dm6
curl -L -o dm6.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz
curl -L -o rmsk.txt.gz https://hgdownload.soe.ucsc.edu/goldenPath/dm6/database/rmsk.txt.gz
zcat rmsk.txt.gz | awk -F'\t' 'BEGIN{OFS="\t"}{print $6,$7,$8,$11"|"$12"/"$13,0,$10}' > dm6_te.bed

# RepeatMasker con librerías del proyecto
docker run --rm -v "$PWD":/data -v "$PWD/libs":/libs:ro dfam/tetools:latest \
  /opt/RepeatMasker/RepeatMasker -pa 3 -lib /libs/panteon_fungi.fasta \
  -dir /data/rm /data/genoma.fna
```
