#!/usr/bin/env bash
# =============================================================================
# Setup + entrenamiento 100 épocas en servidor Ubuntu (sin sudo, sin git).
#
# Uso: desde la carpeta del proyecto corre:
#    bash scripts/run_server.sh
#
# Lo que hace:
#   1. Verifica Python 3 y GPU
#   2. Crea el entorno Python si no existe
#   3. Instala dependencias
#   4. Verifica dataset (y extrae .rar si hace falta)
#   5. Lanza entrenamiento 100 épocas en nohup (sobrevive desconexión SSH)
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
echo "=========================================="
echo " Inpactor3-SegNT · entrenamiento binario"
echo " Carpeta: $PROJECT_DIR"
echo "=========================================="
echo

# ---- Paso 1: Python ----
echo "[1/6] Verificando Python..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: no hay python3. Pide al admin que lo instale."
    exit 1
fi
python3 --version

# ---- Paso 2: GPU ----
echo
echo "[2/6] Verificando GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARN: no hay nvidia-smi. El entrenamiento irá en CPU (lentísimo)."
fi

# ---- Paso 3: dataset ----
echo
echo "[3/6] Verificando dataset..."
if [ -d data/panteon_dataset/processed ] && [ -f data/panteon_dataset/processed/X_train.npy ]; then
    echo "    dataset ya extraído:"
    ls -lh data/panteon_dataset/processed/
elif [ -f data/PanteonDataSet.rar ]; then
    echo "    extrayendo PanteonDataSet.rar..."
    mkdir -p data/panteon_dataset
    cd data/panteon_dataset
    if command -v unrar >/dev/null 2>&1; then
        unrar x ../PanteonDataSet.rar
    elif command -v 7z >/dev/null 2>&1; then
        7z x ../PanteonDataSet.rar
    elif command -v bsdtar >/dev/null 2>&1; then
        bsdtar -xf ../PanteonDataSet.rar
    else
        echo "ERROR: no hay unrar, 7z ni bsdtar. Instala alguno o extrae manualmente."
        exit 1
    fi
    cd "$PROJECT_DIR"
    ls -lh data/panteon_dataset/processed/
else
    echo "ERROR: falta data/PanteonDataSet.rar o data/panteon_dataset/processed/"
    echo "    Copia la base de datos a data/ y vuelve a correr."
    exit 1
fi

# ---- Paso 4: entorno Python ----
echo
echo "[4/6] Entorno Python..."
if [ ! -d .venv ]; then
    echo "    creando venv..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
if ! python -c "import torch, transformers" 2>/dev/null; then
    echo "    instalando dependencias (3-5 min)..."
    pip install --quiet -e .
fi
echo "    OK"

# ---- Paso 5: verificar CUDA desde PyTorch ----
echo
echo "[5/6] Verificando CUDA desde PyTorch..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'    CUDA OK · GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('    WARN: PyTorch no ve GPU. Correrá en CPU (muy lento).')
"

# ---- Paso 6: lanzar entrenamiento ----
echo
echo "[6/6] Lanzando entrenamiento (100 épocas, nohup)..."
mkdir -p logs
nohup python -m inpactor3_segnt.train_binary \
    --config configs/nt_50m_binary.yaml \
    > logs/train_binary.log 2>&1 &
PID=$!
echo "$PID" > logs/train.pid
echo "    PID: $PID"
sleep 5

echo
echo "=========================================="
echo " Entrenamiento LANZADO"
echo "=========================================="
echo
echo "Ver progreso en vivo:"
echo "    tail -f $PROJECT_DIR/logs/train_binary.log"
echo
echo "Verificar que sigue corriendo:"
echo "    ps -p \$(cat $PROJECT_DIR/logs/train.pid)"
echo
echo "Matar si hace falta:"
echo "    kill \$(cat $PROJECT_DIR/logs/train.pid)"
echo
echo "Puedes cerrar la terminal SSH. El entrenamiento sigue."
echo
echo "--- primeras líneas del log ---"
tail -n 30 logs/train_binary.log
