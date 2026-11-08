#!/usr/bin/env bash
# Longer retrain of the s models (epochs=80, patience=20); new run names keep the 30-epoch results.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
DATA=$R/dataset/data.yaml
LOGS=$R/logs
export YOLO_OFFLINE=1
cd $R/weights
for m in yolo26s yolo11s; do
  n=${m}_e80
  for b in 16 8 4; do
    echo "=== $(date '+%F %T') train $n batch=$b"
    if $YOLO detect train model=$R/weights/$m.pt data=$DATA imgsz=512 epochs=80 patience=20 batch=$b device=0 workers=4 \
         project=$R/runs name=$n exist_ok=True > $LOGS/train_$n.log 2>&1; then
      break
    fi
    grep -qiE "out of memory|CUDA error|Killed" $LOGS/train_$n.log || { echo "=== $n failed (not OOM)"; break; }
    echo "=== $n OOM at batch=$b, retrying smaller"
  done
  W=$R/runs/$n/weights/best.pt
  [ -f $W ] && $YOLO detect val model=$W data=$DATA split=test imgsz=512 batch=16 device=0 \
       project=$R/runs name=${n}_test exist_ok=True > $LOGS/test_$n.log 2>&1
  echo "=== $(date '+%F %T') done $n"
done
echo "=== ALL DONE"
