# 학습 스크립트

[docs/model-comparison.md](../docs/model-comparison.md)의 결과를 만든 스크립트입니다. 젯슨(Orin)에서 돌렸습니다.

## 폴더 구성 (프로젝트 루트 아래)

```
<루트>/
  raw/stain        Clothes-stain.v1-version1 (Roboflow, YOLOv11 형식 zip을 푼 것)
  raw/more         More clothes.v1i
  raw2/stain_v2i   stain.v2i   (s-workspace-okn3h/stain-uq7kn)
  raw2/stain_v1i   stain.v1i   (devadharshini/stain-sdji5)
  raw2/hole_v1i    Hole.v1i    (s-workspace-wsqvz/hole-950bj-lmgem)
  raw2/hole_h      hole.v1-h   (swee/hole-inskr)
  realtest/aleminaitha   펼쳐 놓은 옷 실제 사진 32장 (aleminaitha/stain-8kizb, 평가에만 사용)
  weights/         사전학습 가중치 yolo11n/s.pt, yolo26n/s.pt
  venv/            ultralytics가 설치된 파이썬 환경
  scripts/         이 폴더의 파일들
```

## 돌리는 순서

```bash
export R=<루트>
python scripts/merge_dataset.py $R      # -> $R/dataset    (1차, v1)
bash   scripts/train_queue.sh           # 1차 4개 모델 30에폭 + test 평가
python scripts/build_v2.py $R           # -> $R/dataset_v2 (2차, 중복 제거 포함)
bash   scripts/train_queue_v2.sh        # 옛 모델 재평가 + 2차 4개 모델 60에폭 + summary_v2.md
bash   scripts/realtest_predict.sh      # 실제 사진 32장에 최종 후보 2개
```

## 참고

- 젯슨이 인터넷에 안 붙어 있어서 `YOLO_OFFLINE=1`을 켜고, 가중치는 `weights/`에 미리 넣어 두고 거기서 실행합니다.
- 메모리(7GB)가 작아 모델을 하나씩 순서대로 학습하고, 메모리가 모자라면 batch를 16 → 8 → 4로 줄여 다시 돌립니다.
- 클래스는 `0 stain`, `1 torn`, 이미지 크기는 512입니다.
- 데이터 zip과 학습된 모델(best.pt)은 레포에 넣지 않았습니다. 위치는 docs/model-comparison.md의 "모델 파일 위치"에 있습니다.
