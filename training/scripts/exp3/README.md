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
| `run_C1.sh` | C1만 다시(LoRA → 머리만) | `./run_C1.sh` |
| `memwatch.sh` | 1분마다 메모리·컨테이너 기록, 위험할 때만 학습 멈춤 | `HEALTH_URL=... ./memwatch.sh` |
| `c1_siglip_lora.py` | FashionSigLIP + LoRA / 머리만, 무늬 8종 학습과 채점 | `python c1_siglip_lora.py <lora\|probe> <출력 폴더> [batch]` |
| `c2_owlv2_lora.py` | OWLv2 + LoRA 얼룩·찢어짐: 시간 재기 / 학습 / 채점 | `python c2_owlv2_lora.py <time\|train\|eval> <출력 폴더> [크기] [장수\|에폭]` |

필요한 것: 젯슨 venv에 ultralytics 8.4.184, torch 2.14.1+cu132(그대로), peft 0.21.3, open_clip_torch 4.2.2, transformers 5.20.1. 모델 가중치는 `exp3/fashion-siglip`(Marqo/marqo-fashionSigLIP), `exp3/owlv2-base`(google/owlv2-base-patch16-ensemble)에 미리 받아 둡니다.
