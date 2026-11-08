#!/usr/bin/env bash
# 학습에 안 쓴 "펼쳐 놓은 옷" 실제 사진(aleminaitha/stain-8kizb 32장)에 최종 후보 두 모델을 돌려 결과를 txt로 남긴다.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # 프로젝트 루트(기본: 이 스크립트 폴더의 상위)
mkdir -p "$R/logs"
export YOLO_OFFLINE=1
cd $R
for m in yolo26n_v2 yolo11s_v2; do
  venv/bin/yolo detect predict model=runs/$m/weights/best.pt source=realtest/aleminaitha/test/images imgsz=512 conf=0.25 device=0 \
       save=False save_txt=True save_conf=True project=$R/runs/realtest name=$m exist_ok=True > logs/realtest_$m.log 2>&1
done
