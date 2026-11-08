#!/usr/bin/env bash
# Predict on the held-out real-scenario photos (whole garments, phone camera) with the two finalist v2 models.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
export YOLO_OFFLINE=1
cd $R
for m in yolo26n_v2 yolo11s_v2; do
  venv/bin/yolo detect predict model=runs/$m/weights/best.pt source=realtest/aleminaitha/test/images imgsz=512 conf=0.25 device=0 \
       save=False save_txt=True save_conf=True project=$R/runs/realtest name=$m exist_ok=True > logs/realtest_$m.log 2>&1
done
