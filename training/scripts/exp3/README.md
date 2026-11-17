# exp3: 강사님 실험 세 가지 (11-14 ~ 11-15, team4 젯슨)

결과와 설정은 [docs/training-log.md](../../../docs/training-log.md#강사님-실험-세-가지옷-11-14--11-15), 비교표는 [docs/lab.md 8번](../../../docs/lab.md#8-강사님-실험-세-가지옷)에 있습니다. 사진·데이터·가중치는 레포에 넣지 않았습니다. 모두 젯슨의 `~/clothes-state-train`(이하 루트)에서 돌린다고 가정합니다.

| 파일 | 하는 일 | 쓰는 법 |
| --- | --- | --- |
| `realtest_eval.py` | 실사진 32장 + 12장, 사진 단위 찾음/애매/놓침 + 박스 재현율 | `python realtest_eval.py <루트> <run 이름> <출력 json> [device]` |
| `build_v6ft.py` | v6ft 데이터(실사진 12장×5, 깨끗한 옷 400장)와 `data_v6ft.yaml` 만들기 | `python build_v6ft.py` |
| `train_v6ft.py` | v6 best에서 이어서 학습(v6 설정 그대로) | `python train_v6ft.py <batch> <run 이름>` |
| `run_B.sh` | v6ft를 batch 8로, 감시에 걸려 멈추면 batch 4로 한 번 더 | `./run_B.sh` |
| `eval_B.py` | v2 시험 mAP50(얼룩·찢어짐) | `python eval_B.py <출력 json> <run> [run...]` |
| `run_queue.sh` | B 끝나면 B 채점 → C1 → C2 차례로 | `./run_queue.sh` |
| `run_C1.sh` | C1만 다시(LoRA → 분류 층만 학습) | `./run_C1.sh` |
| `memwatch.sh` | 1분마다 메모리·컨테이너 기록, 위험할 때만 학습 멈춤 | `HEALTH_URL=... ./memwatch.sh` |
| `c1_siglip_lora.py` | FashionSigLIP + LoRA / 분류 층만 학습(몸통 고정, linear probe), 무늬 8종 학습과 채점 | `python c1_siglip_lora.py <lora\|probe> <출력 폴더> [batch]` |
| `c2_owlv2_lora.py` | OWLv2 + LoRA 얼룩·찢어짐: 시간 재기 / 학습 / 채점 | `python c2_owlv2_lora.py <time\|train\|eval> <출력 폴더> [크기] [장수\|에폭]` |

**무늬 입은 옷 90% (11-15 ~ 11-17)**, 기록은 [training-log.md](../../../docs/training-log.md#무늬-입은-옷-90-만들기-11-15--11-17)

| 파일 | 하는 일 | 쓰는 법 |
| --- | --- | --- |
| `c1_siglip_lora.py` | 위 학습. 환경 변수로 LoRA r·α·에폭·patience·warmup·클래스 가중치·K-Fashion 추가(EXTRA_DS)·이어서 학습(INIT_LORA)·입력 크기(IMG)·grad accumulation | 파일 맨 위 설명 |
| `kf_pattern.py` | AI Hub K-Fashion 라벨의 프린트 + 옷 상자 → 무늬 8종 조각(레포에 넣지 않음) | `python kf_pattern.py <K-Fashion Validation 폴더> <출력> [작업 수]` |
| `c1_ensemble.py` | LoRA·분류 층만·pattern B 확률 저장(원본·좌우 뒤집기) + 섞기 비교(val로 고름) | `python c1_ensemble.py <lora 폴더> <probe 폴더> <출력>` |
| `c1_gpt.py` | 모름만 GPT에 묻고 정확도(GPT_DETAIL·GPT_MAX·MIX_W) | `OPENAI_API_KEY=... python c1_gpt.py <ens 폴더>` |
| `c1_select.py` | 기준·가중치·GPT 따르는 조건을 val로 고르고 시험 한 번 | `OPENAI_API_KEY=... PATTERN_DS=... python c1_select.py <ens 폴더>` |
| `c1_compare.py` | GPT 모델·detail·후보 힌트·무지도 묻기 비교(val) 후 1등만 시험 | `OPENAI_API_KEY=... PATTERN_DS=... python c1_compare.py <ens 폴더>` |
| `c1_quick_eval.py` | 학습 중 체크포인트를 작은 batch로 시험 | `python c1_quick_eval.py <lora 폴더> <head.pt>` |
| `export_c1_onnx.py` | LoRA를 몸통에 합쳐 ONNX(입력 RGB 0~1) + PyTorch와 같은지 + BILINEAR/BICUBIC 점수 | `python export_c1_onnx.py <lora 폴더> <out.onnx>` |
| `quant_c1_onnx.py` | INT8(가중치) ONNX와 속도·점수 비교 | `python quant_c1_onnx.py <fp32.onnx> <int8.onnx>` |
| `make_lora_parity.py` | Java 입력 맞추기 시험(OnnxPatternParityTest)의 expected.json | `python make_lora_parity.py <onnx> <사진 폴더> <출력> [장수]` |
| `run_C1v2.sh`, `run_C1kf.sh`, `run_C1_384.sh` | 2단계(r 16) / K-Fashion / 384 차례 실행 | `./run_C1kf.sh` |

필요한 것: 젯슨 venv에 ultralytics 8.4.184, torch 2.14.1+cu132(그대로), peft 0.21.3, open_clip_torch 4.2.2, transformers 5.20.1. 모델 가중치는 `exp3/fashion-siglip`(Marqo/marqo-fashionSigLIP), `exp3/owlv2-base`(google/owlv2-base-patch16-ensemble)에 미리 받아 둡니다.
