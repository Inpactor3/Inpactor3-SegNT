#!/usr/bin/env bash
# Piloto: NTv3 vs nuestro lector, sobre hongos 24 kb.
# Uso: bash piloto_ntv3.sh <bundle4_Hongos> [carpeta_trabajo]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:?pasa la carpeta del bundle (la que contiene train/ val/ test/)}"
WORK="${2:-$PWD/piloto_ntv3}"
REPO="${REPO:-InstaDeepAI/NTv3_100M_pre}"
CAP_TRAIN="${CAP_TRAIN:-600}"
CAP_VAL="${CAP_VAL:-400}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-4}"
mkdir -p "$WORK"

echo "[$(date +%H:%M:%S)] 1/4 submuestreo 1:1 (tope $CAP_TRAIN/especie)"
[ -d "$WORK/seq/train" ] || python "$HERE/subsample_train.py" \
  --seq_dir "$SRC" --out_dir "$WORK/seq" --ratio 1 --max_per_species "$CAP_TRAIN"

echo "[$(date +%H:%M:%S)] 2/4 recortando validación a $CAP_VAL por especie"
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

echo "[$(date +%H:%M:%S)] 3/4 embeddings NTv3"
python "$HERE/precompute_ntv3.py" --seq_dir "$WORK/seq" --out_dir "$WORK/emb" \
  --splits val,test --repo "$REPO" --batch "$BATCH"
python "$HERE/precompute_ntv3.py" --seq_dir "$WORK/seq" --out_dir "$WORK/emb" \
  --splits train --repo "$REPO" --batch "$BATCH"

echo "[$(date +%H:%M:%S)] 4/4 entrenando el cabezal"
python "$HERE/train_mil.py" --emb_dir "$WORK/emb" --out_dir "$WORK/modelo" \
  --epochs "$EPOCHS" --class_weight --balance global --pool logits

echo
echo "===== PILOTO NTv3 · HONGOS 24 kb ====="
cat "$WORK/modelo/test_metrics.json"
echo
echo "Referencias (mismo test):"
echo "  v2 100M        BA 0,8261 · F1 0,7365 · PR-AUC 0,8094"
echo "  NTv3 100M_pre  BA 0,8465 · F1 0,7122 · PR-AUC 0,8327"
