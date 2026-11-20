#!/usr/bin/env bash
# NTv3 en animales con el corpus completo (tope 6000/especie = 287.814 ventanas).
# Uso:  bash training/ntv3_animales_full.sh [raiz_del_proyecto]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$PWD}"
CAP_TRAIN="${CAP_TRAIN:-6000}"
CAP_VAL="${CAP_VAL:-2000}"
EPOCHS="${EPOCHS:-20}"
WORK="$ROOT/trabajo/ntv3_animales_full"

mkdir -p "$ROOT/trabajo"
if [ ! -d "$ROOT/vent_a24/train" ]; then
  echo "descargando bundle de animales"
  kaggle datasets download -d cristianosoriog/inpactor2-animales-24k -p "$ROOT/vent_a24" --unzip >/dev/null
fi

if [ ! -f "$HOME/.cache/huggingface/token" ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "AVISO: NTv3 es de acceso restringido; guarda tu token en ~/.cache/huggingface/token"
fi

echo "=== NTv3 · Animales completo · cap $CAP_TRAIN/especie ($(date +%H:%M:%S)) ==="
CAP_TRAIN="$CAP_TRAIN" CAP_VAL="$CAP_VAL" EPOCHS="$EPOCHS" \
  bash "$HERE/piloto_ntv3.sh" "$ROOT/vent_a24" "$WORK"

echo
echo "=== ANIMALES · NTv3 corpus completo ==="
python3 - "$WORK/modelo/test_metrics.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d["at_calibrated_threshold"]
m5 = d["at_050"]
print(f"epoca {d['epoch']} · umbral {d['threshold']}")
print(f"BA {m['balanced_accuracy']:.4f} · F1 {m['f1']:.4f} · "
      f"P {m['precision']:.4f} · R {m['recall']:.4f} · PR-AUC {m['pr_auc']:.4f}")
print(f"a 0,50: BA {m5['balanced_accuracy']:.4f} · F1 {m5['f1']:.4f}")
print()
print("Referencia v2 (287.814 ventanas): BA 0,6319 · F1 0,4239 · PR-AUC 0,5012")
print("Piloto NTv3 (31.800 ventanas):    BA 0,6263 · F1 0,4110 · PR-AUC 0,4868")
PY
echo "listo: $(date +%H:%M:%S)"
