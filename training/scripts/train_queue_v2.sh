#!/usr/bin/env bash
# v2 dataset: evaluate old v1 models on the new test split, then train yolo26n/s and yolo11n/s sequentially.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
DATA=$R/dataset_v2/data.yaml
LOGS=$R/logs
export YOLO_OFFLINE=1
cd $R/weights
for m in yolo26n yolo26s; do
  $YOLO detect val model=$R/runs/$m/weights/best.pt data=$DATA split=test imgsz=512 batch=16 device=0 \
       project=$R/runs name=${m}_v1model_on_v2test exist_ok=True > $LOGS/test_${m}_v1model_on_v2test.log 2>&1
  echo "=== $(date '+%F %T') evaluated old $m on v2 test"
done
for m in yolo26n yolo26s yolo11n yolo11s; do
  n=${m}_v2
  for b in 16 8 4; do
    echo "=== $(date '+%F %T') train $n batch=$b"
    if $YOLO detect train model=$R/weights/$m.pt data=$DATA imgsz=512 epochs=60 patience=15 batch=$b device=0 workers=4 \
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
$R/venv/bin/python -I $R/scripts/summarize_v2.py $R > $R/summary_v2.md 2> $LOGS/summary_v2.err
echo "=== ALL DONE"
