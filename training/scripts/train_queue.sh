#!/usr/bin/env bash
# Sequential training queue: one model at a time (Jetson has ~7GB RAM).
# The Jetson is offline, so weights are local and cwd is the weights dir (AMP check loads yolo26n.pt/yolo11n.pt from here).
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
DATA=$R/dataset/data.yaml
LOGS=$R/logs
export YOLO_OFFLINE=1
cd $R/weights
for m in yolo11n yolo26n yolo11s yolo26s; do
  for b in 16 8 4; do
    echo "=== $(date '+%F %T') train $m batch=$b"
    if $YOLO detect train model=$R/weights/$m.pt data=$DATA imgsz=512 epochs=30 patience=10 batch=$b device=0 workers=4 \
         project=$R/runs name=$m exist_ok=True > $LOGS/train_$m.log 2>&1; then
      break
    fi
    grep -qiE "out of memory|CUDA error|Killed" $LOGS/train_$m.log || { echo "=== $m failed (not OOM)"; break; }
    echo "=== $m OOM at batch=$b, retrying smaller"
  done
  W=$R/runs/$m/weights/best.pt
  [ -f $W ] && $YOLO detect val model=$W data=$DATA split=test imgsz=512 batch=16 device=0 \
       project=$R/runs name=${m}_test exist_ok=True > $LOGS/test_$m.log 2>&1
  echo "=== $(date '+%F %T') done $m"
done
echo "=== ALL DONE"
