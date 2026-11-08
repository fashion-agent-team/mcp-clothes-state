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

사전학습 가중치: Ultralytics `yolo26n.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo26s.pt`(검출), `yolo11n-pose.pt`(입은 사람 위치 안내). Ultralytics 코드와 가중치는 AGPL-3.0입니다(상용 배포 시 라이선스 확인 필요).

CC BY 4.0 데이터를 쓴 모델·결과를 공개할 때는 위 출처 표기를 함께 남깁니다.
