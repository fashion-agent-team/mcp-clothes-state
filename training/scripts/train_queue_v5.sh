#!/usr/bin/env bash
# v5: train yolo26n from the pretrained weights (not a fine-tune: short fine-tunes peaked at epoch 1 and lost tears)
# on dataset_v5, 60 epochs at 512, then the same evaluations as v4.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
LOGS=$R/logs
N=yolo26n_v5
export YOLO_OFFLINE=1
cd $R/weights
ok=0
for b in 16 8 4; do
  echo "=== $(date '+%F %T') train $N batch=$b"
  $YOLO detect train model=$R/weights/yolo26n.pt data=$R/dataset_v5/data.yaml imgsz=512 epochs=60 patience=15 batch=$b device=0 workers=2 \
       project=$R/runs name=$N exist_ok=True > $LOGS/train_$N.log 2>&1
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
fi
echo "=== $(date '+%F %T') ALL DONE"
