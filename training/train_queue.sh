#!/usr/bin/env bash
# 1차: v1 데이터로 4개 모델을 30에폭씩 하나씩 학습하고 test로 평가한다(젯슨 메모리가 7GB라 동시 실행 안 함).
# 젯슨이 인터넷이 안 돼서 사전학습 가중치를 <루트>/weights 에 미리 두고 거기서 실행한다.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # 프로젝트 루트(기본: 이 스크립트 폴더의 상위)
YOLO=$R/venv/bin/yolo
DATA=$R/dataset/data.yaml
LOGS=$R/logs
mkdir -p "$LOGS"
export YOLO_OFFLINE=1
cd $R/weights
for m in yolo11n yolo26n yolo11s yolo26s; do
  for b in 16 8 4; do   # 메모리가 모자라면 batch를 줄여 다시
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
