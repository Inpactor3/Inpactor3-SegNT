#!/usr/bin/env bash
# LoRA sobre el lector v2 + cabezal, hongos 24 kb.
# Uso: bash lora_piloto.sh <bundle> [carpeta_trabajo]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:?pasa la carpeta del bundle}"
WORK="${2:-$PWD/lora_piloto}"
CAP_TRAIN="${CAP_TRAIN:-600}"
CAP_VAL="${CAP_VAL:-400}"
EPOCHS="${EPOCHS:-3}"
mkdir -p "$WORK"

echo "[$(date +%H:%M:%S)] 1/3 submuestreo 1:1 (tope $CAP_TRAIN/especie)"
[ -d "$WORK/seq/train" ] || python "$HERE/subsample_train.py" \
  --seq_dir "$SRC" --out_dir "$WORK/seq" --ratio 1 --max_per_species "$CAP_TRAIN"

echo "[$(date +%H:%M:%S)] 2/3 validación recortada a $CAP_VAL por especie"
python - "$SRC" "$WORK/seq/val" "$CAP_VAL" <<'PY'
import glob, os, shutil, sys
import numpy as np
src, dst, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
if os.path.islink(dst):
    os.unlink(dst)
elif os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
tot = 0
for f in sorted(glob.glob(os.path.join(src, "val", "*.npz"))):
    out = os.path.join(dst, os.path.basename(f))
    z = np.load(f, allow_pickle=True)
    n = min(cap, len(z["label"]))
    np.savez_compressed(out, seq=z["seq"][:n], label=z["label"][:n],
                        window_id=z["window_id"][:n], offsets=z["offsets"],
                        chunk_bp=z["chunk_bp"])
    tot += n
print("ventanas de validación:", tot, "en", len(glob.glob(dst + "/*.npz")), "archivos")
PY

echo "[$(date +%H:%M:%S)] 3/3 entrenando LoRA + cabezal ($EPOCHS épocas)"
python "$HERE/lora_hongos.py" --seq_dir "$WORK/seq" --out_dir "$WORK/modelo" \
  --epochs "$EPOCHS" --rank 8 --lr 3e-4 --batch 2

echo
echo "===== LORA · HONGOS 24 kb ====="
cat "$WORK/modelo/test_metrics.json"
echo
echo "Referencia congelada (MIL): BA 0,8261 · F1 0,7365 · PR-AUC 0,8094"
