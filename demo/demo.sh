#!/usr/bin/env bash
# Demo de Inpactor3 en tres pasos: pruebas, detector y visor.
# Uso:  bash demo/demo.sh          (visor + pruebas + dry-run)
#       bash demo/demo.sh --real   (ademas corre el detector en 1 ventana)
set -euo pipefail
cd "$(dirname "$0")/.."
REAL=0
[ "${1:-}" = "--real" ] && REAL=1

GEN=demo/genoma_demo.fa
if [ ! -f "$GEN" ]; then
  python3 - <<'PY'
import random
random.seed(7)
seq = "".join(random.choice("ACGT") for _ in range(120_000))
with open("demo/genoma_demo.fa", "w") as f:
    f.write(">chr_demo genoma sintetico de 120 kb\n")
    for i in range(0, len(seq), 80):
        f.write(seq[i:i + 80] + "\n")
print("genoma de demo generado: 120 kb")
PY
fi

echo "=== 1/3 pruebas ==="
python3 -m unittest discover -s tests -q

echo
echo "=== 2/3 detector (dry-run) ==="
python3 -m inpactor3.detect --genome "$GEN" \
  --head docs/ventanas/Hongos_w24_logits/best.pt \
  --out /tmp/demo.bed --dry-run

if [ "$REAL" -eq 1 ]; then
  echo
  echo "=== 2b detector real (1 ventana) ==="
  python3 -m inpactor3.detect --genome "$GEN" \
    --head docs/ventanas/Hongos_w24_logits/best.pt \
    --out /tmp/demo.bed --limit 1 --batch 1
  echo "--- BED ---"; cat /tmp/demo.bed
fi

echo
echo "=== 3/3 visor ==="
echo "abre http://127.0.0.1:8000  (Ctrl+C para terminar)"
python3 visor/app.py --port 8000
