#!/usr/bin/env bash
# v4: fine-tune yolo26n_v2 at 512 on dataset_v4 (v3 additions + clean-clothing negatives), evaluate, then (B) train
# yolo11s_v2 for 40 more epochs on dataset_v2 if v4 finished.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
YOLO=$R/venv/bin/yolo
LOGS=$R/logs
export YOLO_OFFLINE=1
cd $R/weights

train() {  # name model data epochs; tries smaller batches when the Jetson runs out of memory
  local n=$1 model=$2 data=$3 ep=$4 b rc
  for b in 16 8 4; do
    echo "=== $(date '+%F %T') train $n batch=$b"
    $YOLO detect train model=$model data=$data imgsz=512 epochs=$ep patience=8 lr0=0.002 batch=$b device=0 workers=2 \
         project=$R/runs name=$n exist_ok=True > $LOGS/train_$n.log 2>&1
    rc=$?  # 137 = killed by the kernel OOM killer (the log itself does not say so)
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 137 ] || grep -qiE "out of memory|CUDA error" $LOGS/train_$n.log || { echo "=== $n failed (not OOM, rc=$rc)"; return 1; }
    echo "=== $n OOM at batch=$b, retrying smaller"
  done
  return 1
}

falsealarms() {  # name weights -> predictions on the held-out clean clothes
  $YOLO detect predict model=$2 source=$R/negatives/heldout imgsz=512 conf=0.2 device=0 save=False save_txt=True save_conf=True \
       project=$R/runs/heldout_neg name=$1 exist_ok=True > $LOGS/heldout_$1.log 2>&1
}

N=yolo26n_v4_neg
if train $N $R/runs/yolo26n_v2/weights/best.pt $R/dataset_v4/data.yaml 20; then
  W=$R/runs/$N/weights/best.pt
  $YOLO detect val model=$W data=$R/dataset_v2/data.yaml split=test imgsz=512 batch=8 device=0 \
       project=$R/runs name=${N}_test exist_ok=True > $LOGS/test_$N.log 2>&1
  falsealarms yolo26n_v2 $R/runs/yolo26n_v2/weights/best.pt
  falsealarms $N $W
  $YOLO detect predict model=$W source=$R/realtest/aleminaitha/test/images imgsz=512 conf=0.2 device=0 \
       save=False save_txt=True save_conf=True project=$R/runs/realtest name=$N exist_ok=True > $LOGS/realtest_$N.log 2>&1
  echo "=== $(date '+%F %T') done $N"
  M=yolo11s_v2_plus40
  if train $M $R/runs/yolo11s_v2/weights/best.pt $R/dataset_v2/data.yaml 40; then
    $YOLO detect val model=$R/runs/$M/weights/best.pt data=$R/dataset_v2/data.yaml split=test imgsz=512 batch=8 device=0 \
         project=$R/runs name=${M}_test exist_ok=True > $LOGS/test_$M.log 2>&1
    echo "=== $(date '+%F %T') done $M"
  fi
fi
echo "=== $(date '+%F %T') ALL DONE"
