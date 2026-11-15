# 입은 옷 종류 분류기(probe) 학습 결과

FashionSigLIP 옷 벡터 위에 종류마다 벡터 하나를 학습한 분류기예요. `probe.json`은 서버 GarmentTextCheck가 읽는 모양(garment-text-fsiglip.json과 같은 형식)이에요.

- 학습: 2026-11-15 젯슨 나노(GPU, open_clip fp16), `train_probe.py`
- 스크립트: `make_lists.py`(목록) → `embed_crops.py`(벡터, GPU 또는 `--onnx`) → `train_probe.py`(학습·평가)
- 벡터 파일(npz)은 레포에 넣지 않았어요(젯슨 나노 `~/closet-probe/`에 있음)

| 평가(소스) | 수 | 글 벡터(지금 서버) | 학습 |
|---|---|---|---|
| CC0 내려놓은 옷 | 1019 | 86.8% | 88.1% |
| Fashionpedia 입은 옷(val) | 1831 | 79.0% | 87.5% |
| 빈 벽·배경을 none으로 잡음 | 420 | 95.5% | 99.8% |

옷 빠짐(옷을 none으로 잘못 봄)은 두 방식 모두 0.5%예요.

## 데이터 출처
- Fashionpedia (Jia et al., ECCV 2020), CC BY 4.0: https://fashionpedia.github.io/home/
- clothing-dataset (Alexey Grigorev), CC0: https://github.com/alexeygrigorev/clothing-dataset
- walls: 스크립트로 만든 빈 벽·바닥 그림
