# Entrenamiento binario SegNT · guía para servidor

Para correr el entrenamiento de 100 épocas sobre el dataset
`inpactor2_detect_panteon_v1` (80 000 ventanas de 50 kb) en un servidor
remoto con GPU dedicada.

## Requisitos del servidor

- Linux con CUDA 11.8+ (idealmente 12.x)
- GPU con al menos **16 GB VRAM** (recomendado A100/A6000, RTX 4090/3090 también sirve)
- **40 GB de disco libre** (dataset + modelos + logs)
- Python 3.10 o 3.11

## Instalación en el servidor

```bash
# Clonar el repo
git clone https://github.com/Inpactor3/Inpactor3-SegNT.git
cd Inpactor3-SegNT

# Crear venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Verificar GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
```

## Transferir el dataset

El dataset son 5 GB. Dos opciones:

### Opción A · subir el .rar desde tu PC

```bash
# Desde tu PC local
scp /home/cami/Desktop/ProyectoIntegrador/PanteonDataSet.rar \
    usuario@servidor:/home/usuario/Inpactor3-SegNT/data/

# En el servidor
cd data/
sudo apt install unrar
mkdir -p panteon_dataset
cd panteon_dataset
unrar x /home/usuario/Inpactor3-SegNT/data/PanteonDataSet.rar
```

### Opción B · descargar desde Zenodo y re-generar

Si el servidor tiene mejor red que tu PC:

```bash
# El dataset no está publicado en Zenodo, solo PanTEon crudo
# Mejor usar la Opción A
```

## Lanzar el entrenamiento

### Primero verificar que todo carga bien

```bash
python -c "
from inpactor3_segnt.npy_dataset import PanteonBinaryDataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
    trust_remote_code=True
)
ds = PanteonBinaryDataset('data/panteon_dataset/processed', 'dev', tokenizer=tok)
print(f'{len(ds)} muestras · shape primera: {ds[0][\"input_ids\"].shape}')
"
```

### Correr entrenamiento en screen/tmux (sobrevive desconexiones)

```bash
screen -S segnt_train
# Dentro de screen:
source .venv/bin/activate
python -m inpactor3_segnt.train_binary --config configs/nt_50m_binary.yaml \
    2>&1 | tee logs/train_binary.log

# Para desconectar sin matar: Ctrl+A, D
# Para reconectar: screen -r segnt_train
```

## Estimación de tiempo

Con GPU única:

| GPU | Tiempo por época | 100 épocas |
|---|---|---|
| T4 (16 GB) | ~15 min | ~25 h |
| A6000 (48 GB) | ~5-7 min | ~10-12 h |
| A100 (40 GB) | ~3-5 min | ~6-8 h |

Con `freeze_encoder=true`, solo entrena la cabeza (~0.5 M params) — es rápido.

## Checkpoints automáticos

El script guarda:

- `models/binary_teon/best_binary.pt` — mejor por `dev_f1`
- `models/binary_teon/ckpt_ep010.pt`, `ckpt_ep020.pt`, … — cada 10 épocas
- `logs/train_binary_history.json` — métricas por época (se escribe cada época)

Si el servidor se cae a mitad, los checkpoints más recientes están salvos.

## Lanzar en segundo plano (si no quieres screen)

```bash
nohup python -m inpactor3_segnt.train_binary \
    --config configs/nt_50m_binary.yaml \
    > logs/train_binary.log 2>&1 &
echo $! > logs/train_binary.pid

# Monitoreo:
tail -f logs/train_binary.log
# Matar:
kill $(cat logs/train_binary.pid)
```

## Recuperar resultados a tu PC

```bash
# Desde tu PC
rsync -av usuario@servidor:/home/usuario/Inpactor3-SegNT/models/binary_teon/ \
    /home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/models/binary_teon/
rsync -av usuario@servidor:/home/usuario/Inpactor3-SegNT/logs/ \
    /home/cami/Desktop/ProyectoIntegrador/Inpactor3_segnt/logs/
```

## Lo que verás durante el entrenamiento

```
[env] device=cuda · torch=2.X
[data] cargando .npy de data/panteon_dataset/processed
[data] train=80000  dev=10000
[data] train positivos=40000 (50.0%)
[model] InstaDeepAI/nucleotide-transformer-v2-50m-multi-species · 0.52 M params entrenables (pool=max)
[ep 001] loss=0.6823 · dev_f1=0.743 (P=0.740 R=0.746) · acc=0.741 · 892s
    ↳ nuevo mejor · guardado
[ep 002] loss=0.5841 · dev_f1=0.812 (P=0.80 R=0.82) · acc=0.805 · 890s
    ↳ nuevo mejor · guardado
...
[ep 100] loss=0.1223 · dev_f1=0.955 · acc=0.950 · 885s

[best] dev_f1 = 0.955

=== Evaluación final sobre test ===
[test] f1=0.952 · P=0.948 · R=0.956 · acc=0.950
```

## Interpretación esperada

Al ser dataset balanceado 50/50 y la tarea bien definida, esperamos **F1 ≥ 0.90**
en test con 100 épocas. Si queda debajo de 0.85, revisar hiperparámetros.

Comparar con:

- **Inpactor 2 Detect** del paper original: F1 ≈ 0.95 sobre su dataset
- **Nuestro Inpactor 2 reentrenado** (piloto 9k): F1 ≈ 0.39 (5 épocas)
- **SegNT con dataset piloto manifest**: F1 ≈ 0.886 (en segmentación por token)

Objetivo: superar el 0.90 en este benchmark con la arquitectura adaptada.
