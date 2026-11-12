# 학습·평가 데이터 출처

이미지는 저장소에 올리지 않습니다. 아래 출처에서 받은 뒤 `scripts/`의 병합 스크립트(merge_dataset.py, build_v2.py, build_v3.py, build_v4.py, build_v5.py, build_v7.py)로 다시 만들 수 있습니다. 모든 데이터는 Roboflow Universe에서 YOLOv11 형식으로 받았습니다.

| 데이터셋 | 출처 | 버전 | 라이선스 | 쓰임 |
|---|---|---|---|---|
| Clothes-stain | https://universe.roboflow.com/laundrycareteam1/clothes-stain/dataset/1 | 1 | CC BY 4.0 | 얼룩(stain) 학습. 라벨 빈 사진은 자동 라벨 후 검토해 일부 사용 |
| More clothes | https://universe.roboflow.com/memorial-university-of-newfoundland-y9kcb/more-clothes/dataset/1 | 1 | CC BY 4.0 | 찢어짐(torn) 학습, good 사진은 깨끗한 옷(음성)과 얼룩 합성 바탕 |
| stain (s-workspace-okn3h) | https://universe.roboflow.com/s-workspace-okn3h/stain-uq7kn/dataset/2 | 2 | CC BY 4.0 | 얼룩 학습(기존 데이터와 겹치는 사진·'sss' 박스 제외) |
| stain (devadharshini) | https://universe.roboflow.com/devadharshini/stain-sdji5/dataset/1 | 1 | CC BY 4.0 | 149장 중 148장이 Clothes-stain과 겹쳐 1장만 사용 |
| hole (swee) | https://universe.roboflow.com/swee/hole-inskr/dataset/1 | 1 | CC BY 4.0 | 찢어짐(Hole → torn) 학습, train만 |
| Hole (s-workspace-wsqvz) | https://universe.roboflow.com/s-workspace-wsqvz/hole-950bj-lmgem/dataset/1 | 1 | CC BY 4.0 | 찢어짐 학습·검증·test |
| CLOTHES | https://universe.roboflow.com/s-workspace-wsqvz/clothes-upt5u-lutml/dataset/1 | 1 | Public Domain | 깨끗한 옷 음성 577장, 오탐 평가 200장(학습 제외), v7 입은 사진 얼룩 합성 바탕 |
| stain (aleminaitha) | https://universe.roboflow.com/aleminaitha/stain-8kizb/dataset/1 | 1 | CC BY 4.0 | 학습 제외, 실제 옷 전체 사진 32장 평가용 |

## 옷 종류 모델 (`yolo26n_fashion12.onnx`)

| 데이터셋 | 출처 | 라이선스 | 쓰임 |
|---|---|---|---|
| Fashionpedia | https://fashionpedia.github.io/home/ | 주석 CC BY 4.0 (사진은 원래 사이트 조건을 따름) | 12종으로 묶어 학습 43,623장·검증 1,158장·시험 2,000장(학습에서 따로 뺌) |
| Clothing dataset (Alexey Grigorev) | https://github.com/alexeygrigorev/clothing-dataset | CC0 | 내려놓거나 옷걸이에 건 옷. 학습 4,104장, 평가 208장(학습 제외). 박스는 1차 모델로 그림 |
| More clothes·얼룩 데이터의 옷 전체 사진 | 위 표의 Roboflow 출처 | CC BY 4.0 | 직접 검토한 내려놓은 옷 61장(5배 반복) 학습, 15장 평가 |
| Clothing Dataset for Second-Hand Fashion v3 (앞면만, HF `fnauman/fashion-second-hand-front-only-rgb`) | https://zenodo.org/records/13788681 · https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb | CC BY 4.0 | fashion12c부터. 학습 5,809장(가디건·조끼·니트·재킷·코트 위주. 겉옷 496장과 Vest 396장은 직접 보고 나누거나 뺌), 시험 449장. 상자는 우리 모델로 그림 |
| Clothing dataset 신발 사진 (Alexey Grigorev) | https://github.com/alexeygrigorev/clothing-dataset | CC0 | fashion12d부터. 신발 431장 중 한 켤레를 정확히 잡은 372장: 학습 332장, 시험 40장. 상자는 OWLv2(Apache 2.0, 맥에서 로컬 실행)로 그림 |
| object_detection_dom v1 (Roboflow, shoe-detect) | https://universe.roboflow.com/shoe-detect/object_detection_dom/dataset/1 | CC BY 4.0 | fashion12d부터. 원본 158장 중 78장 학습(영수증·주소 보이는 사진, 확대·밑창 사진 뺌). 상자는 한 짝씩 다시 그림 |
| Fashion-1K (Humanbased-AI, 예전 Codatta) | https://huggingface.co/datasets/Humanbased-AI/Fashion-1K | openrail, 이미지 출처 표기 없음 | **학습에 안 씀.** 내려놓은 신발 확인용으로만 48장 봄(브랜드 상품 사진 콜라주, 워터마크 있음) |

## 옷 색 모델 (`color20_cnn.onnx`)

| 데이터셋 | 출처 | 라이선스 | 쓰임 |
|---|---|---|---|
| K-Fashion 이미지 (AI Hub, 과학기술정보통신부·한국지능정보사회진흥원) | https://aihub.or.kr | AI Hub 이용 약관(데이터 재배포 금지, 결과물 공개 때 출처 표기) | Validation 사진 119,023장의 정답 상자 옷 171,538벌. 학습 128,142 / 검증 16,880 / 시험 25,626(사진 단위로 나눔). 과정은 [color-model/README.md](color-model/README.md) |
| Second-Hand Fashion v3 시험 사진 | [training-log.md](../docs/training-log.md) fashion12c 데이터와 같음 | CC BY 4.0 | 학습 제외. 내려놓은 옷 45벌 색 눈 확인 |

사전학습 가중치: Ultralytics `yolo26n.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo26s.pt`(검출), `yolo11n-pose.pt`(입은 사람 위치 안내). Ultralytics 코드와 가중치는 AGPL-3.0입니다(상용 배포 시 라이선스 확인 필요).

## 무늬 모델 (`yolo26n_pattern8.onnx`)

| 데이터셋 | 출처 | 라이선스 | 쓰임 |
|---|---|---|---|
| Fashionpedia 무늬 속성(textile pattern, animal) | https://fashionpedia.github.io/home/ | 주석 CC BY 4.0 (사진은 원래 사이트 조건) | 옷(카테고리 0~11)마다 무늬 속성이 정확히 하나인 것만 옷 상자를 8% 넓혀 잘라 씀. 학습 21,085개(무지는 9,000개로 줄임), 검증 1,821개(원래 val), 시험 7,465개(옷 종류 시험 2,000장 + 학습에서 뺀 3,000장의 옷) |
| Clothing Dataset for Second-Hand Fashion v3 (앞면만) | https://zenodo.org/records/13788681 · https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb | CC BY 4.0 | `pattern` 칸 사용(빈 값 = 무지). 학습 약 10,600장(무지 5,000장으로 줄임), 검증 5%, 시험은 공식 test 1,959장. Logo print(작은 로고)·Animal print(동물 그림)·Other·Lace·Embroidered·Transparent·Glitter·Metallic은 우리 분류와 뜻이 달라 뺌 |

사전학습 가중치: Ultralytics `yolo26n-cls.pt`(AGPL-3.0). 스크립트는 `training/scripts/pattern/`.

CC BY 4.0 데이터를 쓴 모델·결과를 공개할 때는 위 출처 표기를 함께 남깁니다.
