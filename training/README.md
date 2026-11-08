# 옷 얼룩·찢어짐 검출 모델 정리 (2026-11-08)

이 폴더에는 이미지와 모델 파일(.pt, .onnx)을 넣지 않았습니다. 데이터는 `DATA_SOURCES.md`의 출처에서 받아 `scripts/`로 다시 만들 수 있습니다.

## 최종 결정
- 모델: **yolo26n_v6** (`yolo26n_v6_tuned_best.pt`, ONNX `yolo26n_v6.onnx`), modelVersion `"yolo26n-v6"`
- 기준: 확신도 **0.5 이상 바로 알림 / 0.3~0.5 지인 확인 / 0.3 미만 버림**
- 옷 사진은 외부 AI(GPT)로 보내지 않음. 애매하면 지인에게만 묻고, 지인이 없거나 답이 없으면 "확실하지 않다"고 안내.
- 위치 안내: 입은 사진(worn)은 입은 사람 몸 기준(자세 인식 yolo11n-pose), 내려놓은 옷(laid)은 사진 기준(사진 왼쪽 = 사용자 왼쪽).
- 지인 질문 줄이기: 같은 옷이면 지인이 "아니야"라고 한 자리는 다시 묻지 않음(`dismissed`), 여러 장 검사 시 2장 이상에서 같은 자리에 나올 때만 질문(`inspect_views`).

## 모델 비교 (v2 test: 이미지 114, 얼룩 박스 212, 찢어짐 박스 32)
| 모델 | 전체 mAP50 / mAP50-95 | 얼룩 mAP50 | 찢어짐 mAP50 | 깨끗한 옷 200장 오탐 (0.2↑ / 0.5↑) | 속도(Jetson) |
|---|---|---|---|---|---|
| v2 (기본 데이터, 60ep) | 0.610 / 0.287 | 0.648 | 0.573 | 55 / 20 | 14ms |
| v4_neg (v2 이어서 + 음성) | 0.583 / 0.265 | 0.667 | 0.499 | 25 / 2 | 14ms |
| 조합 (얼룩 v4 + 찢어짐 v2) | 0.620 / 0.287 | 0.667 | 0.573 | 32 / 7 | 30ms |
| v5 (처음부터, 음성 + 찢어짐 2배) | 0.602 / 0.272 | 0.670 | 0.535 | 8 / 1 | 14ms |
| **v6 (v5 + 자동 조정 설정)** | **0.636 / 0.331** | 0.619 | **0.654** | 10 / 1 | 14ms |
| v7 (v6 + 실사진 12 + 입은 사진 합성 330) | 0.693 / 0.351 | 0.638 | 0.748 | 32 / 3 | 14ms |

v7은 test 점수는 가장 높지만 깨끗한 옷 오탐과 지인 질문이 늘어(합성 바탕이 깨끗한 옷 시험 사진과 같은 종류라 생긴 편향으로 추정) 채택하지 않음.

## v6 기준값별 결과 (찾는 비율은 지인 답 포함)
| 바로 알림 / 지인 확인 | 얼룩 찾음 | 찢어짐 찾음 | 지인 질문: 실제 사진 / 깨끗한 옷 | 깨끗한 옷 잘못 알림 |
|---|---|---|---|---|
| 0.6 / 0.3 | 58.5% | 69% | 25% / 2% | 0% |
| **0.5 / 0.3 (채택)** | 58.5% | 69% | 15.6% / 1.5% | 0.5% |
| 0.6 / 0.2 | 62% | 69% | 34% / 4.5% | 0% |

## v6 학습 설정
- 기본 모델 yolo26n.pt에서 처음부터, imgsz 512, epochs 60 (patience 15 → 53 epoch에서 조기 종료, 최고 38 epoch), batch 16, Jetson Orin 약 2시간 24분
- 하이퍼파라미터 (Ultralytics tuner 30회 × 10 epoch, 검증 mAP50-95 0.142 → 0.224):
```yaml
lr0: 0.01
lrf: 0.01
momentum: 0.89249
weight_decay: 0.00022
warmup_epochs: 3.63805
warmup_momentum: 0.95
box: 7.62525
cls: 0.35163
cls_pw: 0.23085
dfl: 10.81407
hsv_h: 0.09513
hsv_s: 0.51628
hsv_v: 0.0
degrees: 5.23078
translate: 0.0
scale: 0.05204
shear: 0.0
perspective: 0.0
flipud: 0.06866
fliplr: 1.0
bgr: 0.07176
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
close_mosaic: 7
```
- 영향이 컸던 값: close_mosaic, dfl, fliplr, hsv_v(밝기 변화 0), degrees, scale(크기 변화 거의 0), momentum, hsv_h

## 학습 데이터 구성 (클래스: 0 stain, 1 torn)
| 묶음 | 이미지 | 얼룩 박스 | 찢어짐 박스 | 빈 라벨(음성) | 출처 |
|---|---|---|---|---|---|
| dataset_v2/train | 5,239 | 7,858 | 3,999 | 67 | Clothes-stain(라벨 있는 것) + More clothes(torn, good→음성) + stain.v2i + Hole 2종 (중복 제거) |
| dataset_v3/auto | 365 | 715 | 0 | 0 | Clothes-stain 라벨 빈 사진 자동 라벨(0.5↑ 322장) + 직접 검토 43장 |
| synth | 411 | 954 | 0 | 0 | More clothes 깨끗한 옷에 얼룩 합성 |
| dataset_v5/neg | 577 | 0 | 0 | 577 | CLOTHES(퍼블릭 도메인) 깨끗한 옷: v2가 오탐한 380 중 찢어진 디자인 4장 제외 + 무작위 200 |
| dataset_v5/torn2x | 2,071 | 0 | 3,999 | 0 | 찢어짐 학습 사진 2번째 사본(비중 2배) |
| **v6 학습 합계** | **8,663** | 9,527 | 7,998 | 644 | |
| (v7 추가) dataset_v7/auto2 | 12 | 15 | 0 | 0 | 라벨 빈 옷 전체 사진 재검토 |
| (v7 추가) synth_worn | 330 | 629 | 0 | 0 | CLOTHES 입은 사람 사진 몸통에 얼룩 합성 |
| 검증 (dataset_v2/valid) | 207 | 291 | 53 | 19 | |
| test (dataset_v2/test) | 114 | 212 | 32 | 10 | |

평가 전용(학습에 안 씀): CLOTHES 깨끗한 옷 200장(오탐 시험), aleminaitha 실제 옷 전체 사진 32장(실사용 시험).

## ONNX (Java 서버용)
- 파일: `yolo26n_v6.onnx` (9.7MB, opset 17, 고정 입력)
- 입력 `images`: [1, 3, 512, 512], RGB, 0~1, 비율 유지 축소 후 가운데 정렬 회색(114) 패딩(letterbox)
- 출력 `output0`: [1, 6, 5376] — 행 0~3 = 박스 중심 x, y, 너비, 높이(512 기준 픽셀), 행 4 = 얼룩 확률, 행 5 = 찢어짐 확률(이미 0~1). NMS는 서버에서 직접(IoU 0.7 정도).
- 확인: 같은 입력으로 PyTorch와 출력 최대 차이 0.00066. PyTorch 기본 추론은 정사각형이 아닌 크기(예: 512×384)로 넣어서 확신도가 조금 다를 수 있음(예 0.637 vs ONNX 0.596). 기준값은 ONNX(512×512) 결과로 다시 한 번 확인하는 게 좋음.

## 남은 한계와 다음 단계
- 옷 전체·입은 사람 사진에서 진짜 얼룩의 확신도가 낮아(0.2~0.4) 지인 질문이 생김. 근본 해결은 실제 사용 사진(지인 답)으로 재학습. 저장 형식: `friend_check_spec.json`, `scripts/clothes_inspect_model.py`의 `save_feedback()`.
- 찢어짐 평가 박스가 32개뿐이라 찢어짐 수치는 오차가 큼.

자세한 비교 기록은 [docs/model-comparison.md](../docs/model-comparison.md)에 있습니다.

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
  scripts/         이 폴더의 scripts/
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
