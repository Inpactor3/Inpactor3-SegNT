# Guía para empezar · detector de LTR con lector congelado

**Un solo archivo.** Con esto puedes descargar los datos, entrenar y evaluar sin leer nada más.

---

## 1. Qué es esto

Un modelo que dice si una región del genoma contiene un **retrotransposón LTR**. Funciona así:
un lector de ADN preentrenado y **congelado** convierte bloques de 6 kb en números, y un
cabezal pequeño entrenable decide a partir de esos números.

Tienes los datos, los scripts y los resultados de referencia. Lo que vas a reproducir:

| Experimento | Qué hace | Tiempo en una GPU de 32 GB |
|---|---|---|
| Hongos 24 kb | entrenar y evaluar un reino | ~2 h de precompute + 10 min |
| NTv3 en hongos | cambiar de lector | ~3 h |
| Test oro | evaluar en Arabidopsis y Drosophila | ~30 min |
| LoRA | descongelar el lector con adaptadores | ~4 h |

---

## 2. Requisitos

| Cosa | Mínimo |
|---|---|
| GPU NVIDIA | 16 GB (32 GB cómodo). En 5090 hace falta CUDA 12.8 |
| Disco | 30 GB para un reino, 150 GB para los tres |
| Python | 3.11 o 3.12 |
| Cuenta de Kaggle | con token (`~/.kaggle/access_token` o `KAGGLE_API_TOKEN`) |
| Cuenta de HuggingFace | **solo si vas a usar NTv3**, y hay que aceptar su licencia |

---

## 3. Paso 0 · Cuenta y token de Kaggle

1. Crear cuenta gratuita en <https://www.kaggle.com/account/login>.
2. **Verificar el teléfono**: Kaggle lo exige para generar tokens de API
   (Settings → Phone Verification).
3. Ir a <https://www.kaggle.com/settings> → sección **API** → **Create New Token**.
   Se descarga un archivo `kaggle.json`.
4. Colocarlo donde el CLI lo espera:

```bash
mkdir -p ~/.kaggle
mv ~/Descargas/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

5. Instalar el CLI y comprobar:

```bash
pip install kaggle
kaggle datasets list -m | head -3
```

Si lista datasets, el token funciona.

> Si prefieres no usar archivo, también sirve una variable:
> `export KAGGLE_API_TOKEN=KGAT_...`

---

## 4. Paso 1 · Entorno

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.46.3" numpy kaggle
```

Comprueba la GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Si vas a usar NTv3**, no toques este entorno. Instala en una carpeta aparte y úsala
> con `PYTHONPATH`:
> ```bash
> pip install --target ~/ntv3libs "transformers>=4.55" accelerate
> export PYTHONPATH="$HOME/ntv3libs"
> ```

---

## 5. Paso 2 · Acceso a NTv3 (opcional)

NTv3 es de acceso restringido y licencia **no comercial**.

1. Entra en <https://huggingface.co/InstaDeepAI/NTv3_100M_pre> con tu cuenta.
2. Pulsa **Agree and access repository**.
3. Guarda tu token:

```bash
unset HF_TOKEN
mkdir -p ~/.cache/huggingface
read -rsp "token: " T; echo
printf '%s' "$T" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
```

Verifica:

```bash
python -c "from huggingface_hub import hf_hub_download as d; print(d('InstaDeepAI/NTv3_100M_pre','config.json'))"
```

Si imprime una ruta, tienes acceso. Si sale `401`, no aceptaste la licencia.

---

## 6. Paso 3 · Descargar datos y código

Los datasets son públicos.

```bash
# código (scripts de entrenamiento y evaluación)
kaggle datasets download -d cristianosoriog/inpactor2-mil-code -p pkg --unzip

# bundles de secuencias listos para usar
kaggle datasets download -d cristianosoriog/inpactor2-hongos-ventanas -p vent --unzip
kaggle datasets download -d cristianosoriog/inpactor2-plantas-ventanas -p vent --unzip
kaggle datasets download -d cristianosoriog/inpactor2-animales-24k -p vent_a24 --unzip
kaggle datasets download -d cristianosoriog/inpactor2-animales-100k -p vent_a100 --unzip

# genomas y cabezales para los test externos
kaggle datasets download -d cristianosoriog/inpactor2-gold-ventanas -p gold_ds --unzip
kaggle datasets download -d cristianosoriog/inpactor2-plata-ventanas -p plata_ds --unzip

# resultados de referencia (para comparar)
kaggle datasets download -d cristianosoriog/inpactor2-ventanas-resultados -p res --unzip
```

Qué es cada bundle: un `.npz` por especie con `seq (N,4,6000) uint8`, `label`, `window_id`,
`offsets` y `chunk_bp`. Los cuatro bloques son de 6 kb; 4 bloques = 24 kb de entrada.

---

## 7. Paso 4 · Precalcular embeddings

El lector está congelado, así que se calcula una sola vez.

```bash
source .venv/bin/activate
python pkg/mil_code/training/precompute_embeddings.py \
  --seq_dir vent/b24_bundle4_Hongos --out_dir trabajo/emb/Hongos \
  --splits val,test --per_block --trust_remote_code \
  --nt_model InstaDeepAI/nucleotide-transformer-v2-100m-multi-species \
  --batch_size 8 --token_batch 32
```

Repite cambiando `--splits train`. Guarda `(N, 4, 512)` por especie.

**Ajuste por memoria:** en 16 GB usa `--token_batch 32`; en 32 GB, 64 o 128.

---

## 8. Paso 5 · Entrenar

```bash
python pkg/mil_code/training/train_mil.py \
  --emb_dir trabajo/emb/Hongos --out_dir trabajo/modelo/Hongos \
  --epochs 20 --class_weight --balance global --pool logits
```

Sale `metrics.csv` (una fila por época) y `test_metrics.json`.

**Cómo leer la salida:**

| Columna | Qué es |
|---|---|
| `loss` / `val_loss` | error en entrenamiento y validación |
| `val_f1`, `val_bal_acc` | métricas de validación; la época se elige por `bal_acc` |
| `at_calibrated_threshold` | resultado en test con el umbral elegido en validación |
| `at_050` | resultado en test a umbral 0,50 |
| `pr_auc` | calidad del ordenamiento, sin umbral |

---

## 9. Paso 6 · Evaluar en un genoma nuevo (test oro)

```bash
bash pkg/mil_code/training/gold_ventanas.sh gold_ds        # los 8 modelos v2
bash pkg/mil_code/training/gold_ntv3.sh gold_ds            # NTv3 en Arabidopsis
bash pkg/mil_code/training/plata_ventanas.sh plata_ds      # S. pombe
```

---

## 10. Experimentos completos (un comando cada uno)

```bash
# NTv3 en un reino: precompute + entrenamiento + comparación
bash pkg/mil_code/training/piloto_ntv3.sh vent/b24_bundle4_Hongos trabajo/ntv3_hongos

# LoRA sobre el lector v2
bash pkg/mil_code/training/lora_piloto.sh vent/b24_bundle4_Hongos trabajo/lora_hongos
```

---

## 11. Errores frecuentes

| Mensaje | Causa | Solución |
|---|---|---|
| `kaggle: no se encontró la orden` | el CLI está en el venv | `source .venv/bin/activate` |
| `403` al bajar un dataset | estaba privado | ya son públicos; si falla, avisa |
| `401 GatedRepoError` (NTv3) | falta aceptar la licencia | paso 5 de esta guía |
| `Invalid user token ... HF_TOKEN takes precedence` | variable de entorno vieja | `unset HF_TOKEN` |
| `Cannot access gated repo` con token correcto | cuenta equivocada | `whoami` con la misma cuenta que aceptó |
| `no kernel image is available` | 5090 sin CUDA 12.8 | `pip install torch --index-url .../cu128` |
| `ImportError` en transformers | 4.46 y 4.55 mezclados | usar `PYTHONPATH=~/ntv3libs` |
| `CUDA out of memory` | lote grande | `--token_batch 32` o `--batch_size 4` |
| `ValueError: need at least one array to concatenate` | npz vacío | el script ya lo salta; actualiza el código |
| `unzip: cannot find zipfile directory` | ZIP v4.5 | extraer con `python -c "import zipfile; zipfile.ZipFile('x.zip').extractall('.')"` |
| `cannot read file ... in /opt/src` | ruta del host en Docker | usar `/data/...` dentro del contenedor |

Más detalle, con el mensaje exacto y el porqué de cada uno: `DESCARGAS_Y_ERRORES.md`.

---

## 12. Resultados de referencia

Para saber si tu corrida está bien. Mismo test, umbral calibrado.

### Lector v2 (50 kb de etiqueta, 24 kb de entrada)

| Reino | Bal acc | F1 | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|
| Hongos | 0,7766 | 0,7254 | 0,7564 | 0,6969 | 0,8434 |
| Plantas | 0,7450 | 0,9084 | 0,9622 | 0,8604 | 0,9851 |
| Animales | 0,6632 | 0,7166 | 0,7282 | 0,7054 | 0,8116 |

### Experimento de ventanas (etiqueta de 24 kb, MIL)

| Reino | Ventanas | Bal acc | F1 | PR-AUC |
|---|---|---|---|---|
| Hongos | 24 kb | 0,8261 | 0,7365 | 0,8094 |
| Hongos | 100 kb | 0,7454 | 0,6760 | 0,8178 |
| Plantas | 24 kb | 0,7815 | 0,8551 | 0,9633 |
| Plantas | 100 kb | 0,7573 | 0,9489 | 0,9943 |
| Animales | 24 kb | 0,6319 | 0,4239 | 0,5012 |
| Animales | 100 kb | 0,5826 | 0,6174 | 0,6954 |

### NTv3 vs v2 (mismo test y mismo tope)

| Reino | v2 | NTv3 | Ganador |
|---|---|---|---|
| Hongos | 0,8261 · 0,8094 | 0,8465 · 0,8327 | **NTv3** |
| Plantas | 0,7815 · 0,9633 | 0,8271 · 0,9732 | **NTv3** |
| Animales | 0,6319 · 0,5012 | 0,6263 · 0,4868 | empate* |

\* NTv3 usó el 11 % de los datos; no es comparable.

### Test oro (Arabidopsis, 2.380 ventanas)

| Modelo | Bal acc | F1 | PR-AUC |
|---|---|---|---|
| v2 | 0,7031 | 0,5953 | 0,8243 |
| NTv3 | 0,7241 | 0,6241 | **0,8505** |

---

## 13. Cómo citar y qué no hacer

- El lector v2 y NTv3 son de **InstaDeep, NVIDIA y TUM**. Licencia **CC-BY-NC-SA** (v2) y
  **no comercial** (NTv3). Cita a Dalla-Torre et al. (Nucleotide Transformer) y, para NTv3,
  el preprint de bioRxiv de diciembre de 2025.
- **No** uses estos modelos ni sus salidas con fines comerciales.
- **No** subas los pesos a otros repositorios: el acceso es por cuenta.

---

## 14. Si algo no cuadra

1. Revisa la tabla de errores de la sección 10.
2. Busca el mensaje exacto en `DESCARGAS_Y_ERRORES.md`.
3. Comprueba que estás usando el **mismo test** y el **mismo umbral** que la referencia: casi
   todas las diferencias grandes vienen de comparar conjuntos distintos.
