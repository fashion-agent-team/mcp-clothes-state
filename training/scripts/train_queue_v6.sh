#!/usr/bin/env bash
# Tuned-hyperparameter training, then the same evaluations as v5 (test split, clean held-out, real photos, confidence sweep).
# usage: train_queue_v6.sh [dataset dir] [run name]   (defaults: dataset_v5 yolo26n_v6_tuned)
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
LOGS=$R/logs
DS=${1:-dataset_v5}
N=${2:-yolo26n_v6_tuned}
export YOLO_OFFLINE=1
cd $R/weights
ok=0
for b in 16 8 4; do
  echo "=== $(date '+%F %T') train $N batch=$b"
  $R/venv/bin/python -I $R/scripts/train_v6.py $R $b $DS $N > $LOGS/train_$N.log 2>&1
  rc=$?  # 137 = killed by the kernel OOM killer (the log itself does not say so)
  [ $rc -eq 0 ] && { ok=1; break; }
  [ $rc -eq 137 ] || grep -qiE "out of memory|CUDA error" $LOGS/train_$N.log || { echo "=== $N failed (not OOM, rc=$rc)"; break; }
  echo "=== $N OOM at batch=$b, retrying smaller"
done
W=$R/runs/$N/weights/best.pt
if [ $ok -eq 1 ] && [ -f $W ]; then
  $YOLO detect val model=$W data=$R/dataset_v2/data.yaml split=test imgsz=512 batch=8 device=0 \
       project=$R/runs name=${N}_test exist_ok=True > $LOGS/test_$N.log 2>&1
  $YOLO detect predict model=$W source=$R/negatives/heldout imgsz=512 conf=0.2 device=0 save=False save_txt=True save_conf=True \
       project=$R/runs/heldout_neg name=$N exist_ok=True > $LOGS/heldout_$N.log 2>&1
  $YOLO detect predict model=$W source=$R/realtest/aleminaitha/test/images imgsz=512 conf=0.2 device=0 \
       save=False save_txt=True save_conf=True project=$R/runs/realtest name=$N exist_ok=True > $LOGS/realtest_$N.log 2>&1
  cd $HOME && $R/venv/bin/python -I $R/scripts/conf_sweep.py $R $R/conf_sweep_$N.json $N > $LOGS/conf_sweep_$N.log 2>&1
fi
echo "=== $(date '+%F %T') ALL DONE"
