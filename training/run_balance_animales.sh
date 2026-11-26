#!/usr/bin/env bash
# Experimento de balance por grupo en Animales, sin recortar el corpus.
# Corre dos veces sobre la MISMA cache: control (sin pesos) y balance (cap 3x).
#
# Uso en la maquina con la cache y GPU:
#   bash training/run_balance_animales.sh [raiz_del_proyecto]
#
# Variables opcionales:
#   EMB_DIR=/ruta/Animales   cache con train/val/test/*_emb.npy y *_labels.npy
#   OUT_DIR=/ruta/salida     base de salida (se crean _control y _balance)
#   CLASS_WEIGHT=1           pos_weight global (por defecto 1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$PWD}"
EMB="${EMB_DIR:-$ROOT/trabajo/emb/Animales}"
BASE="${OUT_DIR:-$ROOT/trabajo/ckpt/Animales_grupo}"
CLASS_WEIGHT="${CLASS_WEIGHT:-1}"

if [ ! -d "$EMB/train" ]; then
  echo "ERROR: no existe $EMB/train"
  echo "Ajusta EMB_DIR a la cache real (train/val/test con *_emb.npy)."
  exit 1
fi
if [ ! -d "$EMB/val" ]; then
  echo "ERROR: no existe $EMB/val; sin validacion no se puede calibrar el umbral."
  exit 1
fi

ARGS=(--emb_dir "$EMB"
      --group_csv "$ROOT/docs/animales_taxonomia_limpio.csv"
      --epochs 30 --patience 10 --pool logits
      --cap 3 --min_group_windows 2000)

if [ "$CLASS_WEIGHT" = "1" ]; then
  ARGS+=(--class_weight)
fi

echo "== control (sin pesos por grupo) =="
python "$HERE/train_group_weights.py" "${ARGS[@]}" \
  --out_dir "${BASE}_control" --no_group_weights

echo
echo "== balance (peso por grupo, cap 3x) =="
python "$HERE/train_group_weights.py" "${ARGS[@]}" \
  --out_dir "${BASE}_balance"

echo
echo "== comparacion =="
python3 - "${BASE}_control" "${BASE}_balance" <<'PY'
import json
import sys

def load(path):
    with open(path + "/test_metrics.json") as f:
        return json.load(f)

a, b = load(sys.argv[1]), load(sys.argv[2])
rows = [
    ("PR-AUC global", a["at_calibrated_threshold"]["pr_auc"], b["at_calibrated_threshold"]["pr_auc"]),
    ("Balanced accuracy", a["at_calibrated_threshold"]["balanced_accuracy"], b["at_calibrated_threshold"]["balanced_accuracy"]),
    ("F1", a["at_calibrated_threshold"]["f1"], b["at_calibrated_threshold"]["f1"]),
    ("Recall", a["at_calibrated_threshold"]["recall"], b["at_calibrated_threshold"]["recall"]),
    ("Macro PR-AUC", a["macro_pr_auc"], b["macro_pr_auc"]),
]
print(f"{'metrica':22s} {'control':>9s} {'balance':>9s} {'delta':>9s}")
for name, x, y in rows:
    print(f"{name:22s} {x:9.4f} {y:9.4f} {y-x:+9.4f}")
print("\nPor grupo (PR-AUC / lift):")
for path, tag in ((sys.argv[1] + "/per_group_test.csv", "control"),
                  (sys.argv[2] + "/per_group_test.csv", "balance")):
    print(" ", tag)
    with open(path) as f:
        import csv
        for r in csv.DictReader(f):
            if r["evaluable"] == "True":
                print(f"    {r['group']:28s} n={int(r['n']):7d} "
                      f"PR-AUC={float(r['pr_auc']):.4f} lift={float(r['lift']):+.4f}")
PY

echo
echo "Archivos: ${BASE}_control/ y ${BASE}_balance/"
