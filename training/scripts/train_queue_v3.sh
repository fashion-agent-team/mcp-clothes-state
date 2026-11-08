#!/usr/bin/env bash
# v3: fine-tune yolo26n_v2 on dataset_v3 (v2 + reviewed autolabel + synthetic whole-garment stains) at imgsz 1024,
# then evaluate on the unchanged v2 test split and predict on the held-out real-scenario photos.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
DATA=$R/dataset_v3/data.yaml
LOGS=$R/logs
N=yolo26n_v3_synth1024
export YOLO_OFFLINE=1
cd $R/weights
# 1024px validation runs at 2x batch and the Jetson shares 7GB RAM with the GPU, so start small and use fewer workers
ok=0
for b in 4 2; do
  echo "=== $(date '+%F %T') train $N batch=$b"
  $YOLO detect train model=$R/runs/yolo26n_v2/weights/best.pt data=$DATA imgsz=1024 epochs=20 patience=8 lr0=0.002 batch=$b device=0 workers=2 \
       project=$R/runs name=$N exist_ok=True > $LOGS/train_$N.log 2>&1
  rc=$?  # 137 = killed by the kernel OOM killer (the log itself does not say so)
  [ $rc -eq 0 ] && { ok=1; break; }
  [ $rc -eq 137 ] || grep -qiE "out of memory|CUDA error" $LOGS/train_$N.log || { echo "=== $N failed (not OOM, rc=$rc)"; break; }
  echo "=== $N OOM at batch=$b, retrying smaller"
done
W=$R/runs/$N/weights/best.pt
if [ $ok -eq 1 ] && [ -f $W ]; then
  $YOLO detect val model=$W data=$R/dataset_v2/data.yaml split=test imgsz=1024 batch=4 device=0 \
       project=$R/runs name=${N}_test exist_ok=True > $LOGS/test_$N.log 2>&1
  $YOLO detect val model=$R/runs/yolo26n_v2/weights/best.pt data=$R/dataset_v2/data.yaml split=test imgsz=1024 batch=4 device=0 \
       project=$R/runs name=yolo26n_v2_test1024 exist_ok=True > $LOGS/test_yolo26n_v2_1024.log 2>&1
  $YOLO detect predict model=$W source=$R/realtest/aleminaitha/test/images imgsz=1024 conf=0.25 device=0 \
       save=False save_txt=True save_conf=True project=$R/runs/realtest name=$N exist_ok=True > $LOGS/realtest_$N.log 2>&1
fi
echo "=== $(date '+%F %T') ALL DONE"
