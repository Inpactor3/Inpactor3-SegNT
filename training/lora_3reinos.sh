#!/usr/bin/env bash
# LoRA sobre el lector v2 en los tres reinos, con control congelado para animales.
#
# Uso:  bash training/lora_3reinos.sh [carpeta_de_trabajo]
# Entorno: CAP_TRAIN (600), CAP_VAL (400), EPOCHS (3), REPO
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$PWD}"
CAP_TRAIN="${CAP_TRAIN:-600}"
CAP_VAL="${CAP_VAL:-400}"
EPOCHS="${EPOCHS:-3}"
mkdir -p "$ROOT/trabajo"

descargar() {  # dataset destino
  kaggle datasets download -d "cristianosoriog/$1" -p "$2" --unzip >/dev/null
}

# --- bundles ---------------------------------------------------------------
[ -d "$ROOT/vent/b24_bundle4_Hongos" ] || descargar inpactor2-hongos-ventanas "$ROOT/vent"
[ -d "$ROOT/vent/b24_bundle4_Plantas" ] || descargar inpactor2-plantas-ventanas "$ROOT/vent"
[ -d "$ROOT/vent_a24/train" ] || descargar inpactor2-animales-24k "$ROOT/vent_a24"

# --- LoRA ------------------------------------------------------------------
lora() {  # reino bundle etiqueta
  local reino=$1 bundle=$2 tag=$3
  local work="$ROOT/trabajo/lora_$tag"
  echo
  echo "===== LoRA · $reino ($(date +%H:%M:%S)) ====="
  CAP_TRAIN="$CAP_TRAIN" CAP_VAL="$CAP_VAL" EPOCHS="$EPOCHS" \
    bash "$HERE/lora_piloto.sh" "$bundle" "$work"
}

# --- control congelado para animales --------------------------------------
control_animales() {
  local src="$ROOT/trabajo/ventanas/emb_Animales_w24"
  local dst="$ROOT/trabajo/control_animales/emb"
  if [ ! -d "$src" ]; then
    echo "AVISO: no existe la cache $src; se omite el control congelado"
    return 0
  fi
  echo
  echo "===== Control congelado · Animales ($(date +%H:%M:%S)) ====="
  python3 - "$src" "$dst" "$CAP_TRAIN" "$CAP_VAL" <<'PY'
import glob, os, sys
import numpy as np
src, dst, cap_tr, cap_va = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for split, cap in (("train", cap_tr), ("val", cap_va), ("test", 0)):
    os.makedirs(os.path.join(dst, split), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, split, "*_emb.npy"))):
        out = os.path.join(dst, split, os.path.basename(f))
        if os.path.exists(out):
            continue
        e = np.load(f)
        lab = np.load(f.replace("_emb.npy", "_labels.npy"))
        if cap and len(lab) > cap:
            e, lab = e[:cap], lab[:cap]
        np.save(out, e)
        np.save(out.replace("_emb.npy", "_labels.npy"), lab)
    print(split, "listo")
PY
  python3 "$HERE/train_mil.py" --emb_dir "$dst" \
    --out_dir "$ROOT/trabajo/control_animales/modelo" \
    --epochs 20 --class_weight --balance global --pool logits
}

# --- ejecucion -------------------------------------------------------------
lora Hongos   "$ROOT/vent/b24_bundle4_Hongos"   hongos
lora Plantas  "$ROOT/vent/b24_bundle4_Plantas"  plantas
control_animales
lora Animales "$ROOT/vent_a24"                  animales

# --- resumen ---------------------------------------------------------------
echo
echo "===== RESUMEN LoRA ====="
python3 - "$ROOT" <<'PY'
import json, os, sys
raiz = sys.argv[1]
filas = [
    ("Hongos",   "congelado", 0.8261, 0.7365, 0.8094),
    ("Hongos",   "LoRA",      None,   None,   None),
    ("Plantas",  "congelado", 0.7815, 0.8551, 0.9633),
    ("Plantas",  "LoRA",      None,   None,   None),
    ("Animales", "congelado (tope 600)", None, None, None),
    ("Animales", "LoRA",      None,   None,   None),
]
rutas = {
    ("Hongos", "LoRA"): "lora_hongos/modelo/test_metrics.json",
    ("Plantas", "LoRA"): "lora_plantas/modelo/test_metrics.json",
    ("Animales", "LoRA"): "lora_animales/modelo/test_metrics.json",
    ("Animales", "congelado (tope 600)"): "control_animales/modelo/test_metrics.json",
}
for i, (reino, modelo, ba, f1, pr) in enumerate(filas):
    ruta = rutas.get((reino, modelo))
    if ruta:
        p = os.path.join(raiz, "trabajo", ruta)
        if os.path.exists(p):
            m = json.load(open(p))["at_calibrated_threshold"]
            ba, f1, pr = m["balanced_accuracy"], m["f1"], m["pr_auc"]
    fmt = lambda v: "   —  " if v is None else f"{v:.4f}".replace(".", ",")
    print(f"{reino:9s} {modelo:22s} BA {fmt(ba)} · F1 {fmt(f1)} · PR-AUC {fmt(pr)}")
PY
echo "listo: $(date +%H:%M:%S)"
