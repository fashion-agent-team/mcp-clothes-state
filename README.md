# mcp-clothes-state

옷 사진 한 장을 받아 **옷 한 벌마다 종류·색·무늬·얼룩·찢어짐**을 찾아 주는 MCP 서버입니다.
같이 입을 옷들의 **색 조합과 자리(면접·장례식 등)에 맞는 색**도 봅니다(`outfit_color_check`).
패션 에이전트(시각장애인용 음성 코디 도우미)의 한 부분이고, 옷 추천 2단계에서 agent가 사진마다 부릅니다.

담당: 정지원

| 항목 | 값 |
| --- | --- |
| 도구 | `clothes_inspect`(사진 → 옷·결함), `outfit_color_check`(옷 목록 → 색 조합 판단, [규칙](docs/color-rules.md)) |
| 포트 · 엔드포인트 | `8093` · `http://mcp-clothes-state:8093/mcp` (Streamable HTTP, sync) |
| 상태 확인 | `GET /actuator/health` |
| 기술 | Java 21, Spring Boot 4.0.9, Spring AI MCP Server 2.0.2, ONNX Runtime 1.20.0 (CPU) |
| 모델 | 얼룩·찢어짐 `yolo26n_v6.onnx`, 옷 종류 12종 `yolo26n_fashion12.onnx`, 옷 색 20가지 `color20_cnn.onnx`, 무늬 8종 `yolo26n_pattern8.onnx` (모두 우리가 학습) |
| `modelVersion` | `yolo26n-v6+yolo26n-fashion12e+yolo26n-pattern8` (없는 모델은 빠짐) |

## 목차

1. [하는 일과 지키는 것](#하는-일과-지키는-것)
2. [MCP 도구 `clothes_inspect`](#mcp-도구-clothes_inspect)
3. [처리 순서](#처리-순서)
4. [판단 기준](#판단-기준)
5. [GPT 확인](#gpt-확인)
6. [모델](#모델)
7. [실행 방법](#실행-방법)
8. [설정](#설정)
9. [테스트](#테스트)
10. [폴더 구성](#폴더-구성)
11. [데이터 출처와 라이선스](#데이터-출처와-라이선스)
12. [문서](#문서)
13. [남은 일](#남은-일)

## 하는 일과 지키는 것

- **얼룩·찢어짐을 찾는 것까지가 이 서버 일입니다.** 어떻게 말할지(빨기, 다른 옷으로 바꾸기)는 agent의 GPT가 결과 글만 보고 정합니다.
- 모델은 이 서버 안에서 돌립니다. **찍은 사진 전체는 외부로 보내지 않고 저장하지 않습니다.** GPT에는 동의한 사용자일 때 필요한 부분(얼룩 조각, 내려놓은 사진의 옷 한 벌)만 보냅니다.
- 확실하지 않은 결함은 사용자가 동의했을 때(`allowGpt=true`)만 **그 부분 조각만** 잘라 GPT에게 묻습니다([GPT 확인](#gpt-확인)). 무늬를 모르는 옷은 내려놓은 사진일 때만 그 옷 박스를 묻습니다.
- 입은 사진은 동의한 사용자일 때 옷 실수(뒤집어 입음·앞뒤·단추·택)를 보려고 옷 박스를 보내는데, 얼굴이 들어갈 수 있는 위쪽은 회색으로 가리고 저장하지 않습니다(`wearIssues`).
- 저장하는 것은 GPT가 결함이라고 한 조각과 GPT에게 무늬를 물은 내려놓은 옷 그림뿐이고, 다시 학습할 후보로 30일만 보관합니다. GPT 답도 틀릴 수 있어 사람이 확인한 뒤 학습합니다.
- 한 장에 옷이 여러 벌 있을 수 있습니다(두 벌 놓고 비교).
- 실패해도 예외를 던지지 않고 `{"error": {"code", "message"}}`를 돌려줍니다.

## MCP 도구 `clothes_inspect`

입출력 약속은 앱 레포 [MCP 도구 입출력](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md)을 따릅니다.
코드는 [`ClothesTools.java`](src/main/java/com/fashionagent/clothesstate/ClothesTools.java), 응답 모양은 [`InspectResult.java`](src/main/java/com/fashionagent/clothesstate/inspect/InspectResult.java)입니다.

### 입력

| 이름 | 필수 | 값 | 설명 |
| --- | --- | --- | --- |
| `imageBase64` | 예 | JPEG base64 | 풀었을 때 8MB까지. 입은 사진도 **좌우를 뒤집지 않고 원본 그대로** |
| `mode` | 아니요 | `laid`(기본) · `worn` | `laid`: 내려놓은 옷을 위에서 찍음. `worn`: 옷을 입고 폰 앞에 섬 |
| `view` | 아니요 | `front` · `back` | `mode=worn`일 때만 씀. 없으면 `front`. 입고 검사는 앞·뒤 두 장(옆모습은 2026-11-12에 뺌, `side`는 예전 호환으로만 받음) |
| `allowGpt` | 아니요 | `true` · `false`(기본) | 사용자가 GPT 확인에 동의했을 때만 `true` |

### 출력

```jsonc
{
  "garments": [
    {
      "id": "g1",                 // laid는 왼쪽부터, worn은 위쪽부터 g1, g2…
      "category": "top",          // top | bottom | outer | dress | shoes | other
      "name": "셔츠",             // 12종 이름(아래 표)
      "colorName": "남색",
      "colorFamily": "navy",
      "tone": "dark",             // light | mid | dark
      "subColorName": "흰색",     // 무늬 옷만: colorName과 귀로 다른 두 번째 색(체크·줄·호피의 섞인 색). 없으면 null
      "subColorFamily": "white",
      "pattern": "stripe",        // plain | stripe | check | floral | dot | graphic | animal | other, 모르면 null
      "patternCheckedBy": "model", // model | gpt(모델이 몰라서 GPT에게 물음) | null
      "patternGuess": null,       // pattern이 null(모름)인데 모델 1등이 무늬면 "other"(그래픽만 "graphic"). "무늬가 있는 것 같아요"로만
      "material": null,           // 니트·가디건만 "니트", 나머지 null
      "notes": null,
      "formality": "smart",       // 이름으로 분명한 것만, 나머지 unknown
      "defects": [
        { "kind": "stain", "where": "옷 오른쪽 아래쪽", "confidence": 0.62, "checkedBy": "model" },
        { "kind": "tear",  "where": "옷 왼쪽",          "confidence": 0.34, "checkedBy": "gpt" }
      ],
      "box": { "x": 0.10, "y": 0.20, "w": 0.30, "h": 0.40 },  // 사진 크기를 1로 본 비율
      "position": "left",         // laid만: 박스 중심 x 0.4 미만 left, 0.6 초과 right, 그 사이 center. worn은 null
      "confidence": 0.91,         // 옷 종류 모델 확신도. 사진 전체를 한 벌로 본 경우 null
      "embedding": [0.01234, -0.0567, …]  // 옷 박스 이미지 벡터 768개(길이 1). 벡터 모델이 없으면 null
    }
  ],
  "framing": null,
  "lighting": "ok",               // ok | dark
  "source": "model",
  "modelVersion": "yolo26n-v6+yolo26n-fashion12e",
  "pendingHelper": null,          // 확인하지 못한 후보가 있을 때만(아래)
  "embeddingModel": "marqo-fashionSigLIP"  // embedding을 만든 모델. 없으면 null
}
```

- **`defects[].checkedBy`**
  - `model`: 우리 모델 확신도가 0.5 이상이라 바로 알림.
  - `gpt`: 모델이 애매하게 봐서(0.2~0.5) 조각을 GPT에게 물었더니 얼룩·찢어짐이라고 함. agent는 "~인 것 같아요"처럼 조금 약하게 말합니다.
  - `confidence`는 두 경우 모두 **우리 모델 확신도**입니다.
- **`kind`**: `stain`(얼룩), `tear`(찢어짐·구멍).
- **`where`**: 그 옷 박스 안을 가로·세로 셋으로 나눠 말합니다. 사용자가 "어디야?"라고 물을 때만 말합니다.
  - `laid`: 사진 기준(사진 왼쪽 = 사용자 왼쪽). 예: `옷 오른쪽 아래쪽`.
  - `worn` 앞모습: 입은 사람 몸 기준(사진 오른쪽 = 그 사람 왼쪽). 예: `몸 왼쪽 위쪽`.
  - `worn` 뒷모습: `등 …`, 사진 왼쪽 = 왼쪽.
  - 지금은 사람이 화면 가운데 서 있다고 보고 정합니다. 자세 인식(pose)은 다음 단계입니다.
- **`pendingHelper`**: GPT로 확인하지 못한 애매한 결함(0.3~0.5)이 있을 때만 들어갑니다. 한 벌에 한 번 묻는 약속이라 가장 확신도가 높은 후보가 있는 옷 한 벌만 담습니다(후보 최대 5개).

  ```json
  {
    "garmentId": "g1",
    "candidates": [ { "kind": "stain", "where": "옷 가운데", "confidence": 0.41, "box": { "x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1 } } ],
    "note": "확실하지 않은 부분이 있어요."
  }
  ```

  agent는 "확실하지 않은 부분이 있어요"라고만 말합니다.
- **`embedding`**: 옷장(mcp-closet)이 "같은 옷인지" 판단할 때 쓰는 옷 박스 이미지 벡터입니다(이슈 #15). 사진은 저장하지 않고 숫자만 돌려줍니다. 옷장은 `embeddingModel`이 같은 벡터끼리만 비교합니다. 사람에게 읽어 줄 값이 아니라서 agent는 GPT에게 넘기기 전에 빼는 것이 좋습니다(옷 한 벌에 숫자 768개).
  - 만드는 법(이슈 #15 측정과 같음): 옷 박스를 사방 3% 넓혀 자름 → 긴 변 기준 정사각형으로 회색(124,116,104) 채움 → 224×224(PIL BICUBIC과 같은 계산) → (x/255 − 0.5)/0.5 → 원본과 좌우 뒤집은 사진 벡터를 각각 길이 1로 맞춰 평균 → 다시 길이 1, 소수점 5자리.
- **`wearIssues`** (입은 사진만): 옷 실수. 우리 모델로 늘 보고, 동의했으면 GPT 결과도 합칩니다(같은 옷·같은 실수는 모델 것 하나). 확인하지 않았으면 `null`, 확인했는데 없으면 `[]`.

  ```json
  "wearIssues": [ { "kind": "tag_out", "garmentId": "g1", "category": "top", "confidence": 0.9, "checkedBy": "model" } ]
  ```

  - `kind`: `inside_out`(뒤집어 입음), `backwards`(앞뒤 바뀜), `buttons_misaligned`(단추 어긋남, 셔츠·가디건·자켓·코트·조끼·원피스만), `tag_out`(택 튀어나옴).
  - **우리 모델(`checkedBy: model`)**: 택·단추·뒤집힘 모델 세 개(YOLO26n-cls, Fashionpedia에 실수를 합성해 학습, [training-log](docs/training-log.md#입은-옷-실수-택단추뒤집힘-wear1-2026-11-13-claude))를 이 서버에서 돌립니다. 사진이 밖으로 안 나가서 `allowGpt`와 상관없이 입은 사진이면 늘 봅니다. 택은 뒷모습이면 `tag_out`, 앞모습(앞 목에 택)이면 `backwards`. 단추는 셔츠·가디건 앞모습만. 확률 0.8 이상만(뒤집힘은 0.7). 2차 합성 검증 사진 기준 택 90.9%/오탐 0.6%, 단추 93.7%/1.2%, 뒤집힘 90.0%/1.9%이고 **실제 사진 평가는 아직**입니다. 끄려면 `CLOTHES_WEAR_MODEL_ON=false`.
  - **GPT(`checkedBy: gpt`)**: 동의했을 때만. 옷마다(신발 빼고 위에서부터 최대 3벌) 옷 박스를 사방 5% 넓혀 자르고, **얼굴이 들어갈 수 있는 사진 위쪽(앞모습 25%, 뒷모습 12%)은 회색으로 칠한 뒤** 긴 변 768px로 줄여 GPT에게 보냅니다. 조각은 저장하지 않습니다. 옷을 못 찾아 사진 전체를 한 벌로 본 경우는 보내지 않습니다.
  - GPT 확신도 0.7 이상만 넣습니다. 사진 한 장 판단이라 agent는 "~인 것 같아요"로 말하고 손으로 확인해 보라고 합니다.
  - 얼룩 GPT 확인과 동시에 보내고 3초(`clothes.wear-check.timeout`)까지 기다립니다. 끄려면 `CLOTHES_WEAR_CHECK=false`, 모델만 바꾸려면 `CLOTHES_WEAR_MODEL`(비우면 `CLOTHES_GPT_MODEL`).
- **`lighting: "dark"`**: 사진 평균 밝기(L\*)가 25보다 낮으면 검사하지 않고 `garments: []`로 돌려줍니다. agent는 밝은 곳에서 다시 찍어 달라고 안내합니다.

### 오류

| `code` | 언제 |
| --- | --- |
| `no_image` | 사진이 없음 |
| `bad_image` | 사진이 8MB를 넘거나, 읽지 못하거나, 32px보다 작음 |
| `bad_request` | `mode`·`view` 값이 틀림 |
| `not_ready` | 얼룩 모델 파일이 없음 |
| `upstream_failed` | 그 밖의 서버 오류("지금은 옷 상태를 확인하지 못했어요.") |

## 처리 순서

코드는 [`ClothesInspectService.java`](src/main/java/com/fashionagent/clothesstate/inspect/ClothesInspectService.java)입니다.

1. **사진 풀기**: base64 → 이미지. 큰 사진은 줄여서 읽습니다.
2. **밝기 확인**: 너무 어두우면 `lighting: dark`로 끝.
3. **옷 찾기** (fashion12): 옷 박스를 찾습니다. `laid`는 확신도 0.25 이상, `worn`은 0.4 이상을 옷으로 봅니다. 종류가 달라도 박스가 거의 같으면(IoU 0.7) 한 벌로 합치고, 최대 6벌까지 돌려줍니다. 모델이 없거나 옷을 못 찾으면 사진 전체를 옷 한 벌(`category: other`)로 봅니다.
4. **얼룩·찢어짐 찾기** (v6): 확신도 0.2 이상 박스를 받습니다. 각 결함은 박스 중심이 들어가는 옷(겹치면 더 작은 옷)에 붙입니다.
5. **판단**: 0.5 이상은 바로 `defects`, 0.2~0.5는 GPT 확인([아래](#판단-기준)).
6. **색** (color20): 옷 박스 조각을 64×64로 줄여 색 인식 모델에 넣고 20가지 색 중 하나를 고릅니다([아래](#옷-색-20가지-color20_cnnonnx)). 밝기(`tone`)는 옷 픽셀 중간값으로 보탭니다. 옷 박스가 없으면(사진 전체가 한 벌) 예전처럼 픽셀로 정합니다. 색을 세 무리로 나눠 가장 큰 무리를 쓰고, 둘레 띠의 절반 넘게 차지한 색(흰 배경판 등)은 배경으로 건너뛰고, 입은 사진은 피부색을 뺍니다.
7. **무늬** (pattern8): 옷 박스를 8% 넓혀 잘라 8종으로 나눕니다. 1등이 무늬인데 확률이 0.7보다 낮으면 "모름"이고, 동의한 사용자의 내려놓은 사진이면 그 옷 박스만 GPT에게 물어 `patternCheckedBy: gpt`로 채웁니다. 그래도 모르면 `null`, 신발은 `null`입니다. 무늬 옷이면 박스 가운데에서 두 번째 색(`subColorName`)도 찾고, 두 번째 색이 없는데 모델 확률이 0.9 미만이면 무늬를 버립니다(모니터 화면 줄·청바지 결을 줄무늬로 보던 문제, 시험 사진에서 무지→무늬 0.7→0.0%, 맞힌 무늬 손실 1.7~4.5%p). **무늬 이름은 모델 확률 0.9 이상일 때만** 댑니다. 0.7~0.9면 `pattern: null`, `patternGuess: "other"`("무늬가 있는 것 같아요")로 주고, 동의한 사용자의 내려놓은 사진이면 GPT에게 물어 이름을 채웁니다. 호피(animal)는 0.9 이상이어도 동의했으면 GPT에게 다시 묻습니다(2026-11-13 호피 옷을 "줄무늬"로 단정한 문제. 근거: 무늬 시험 사진 9,424장에서 무늬 이름이 맞은 비율 test_fp 0.7~0.8 41%·0.8~0.9 52%·0.9~0.95 62%·0.95~ 83%, test_sh 74%·78%·94%·97%. 1등 호피 42장 중 19장이 틀림). 로그 `clothes_inspect patterns`에 옷마다 모델 1등과 확률을 남깁니다. 무늬를 모르는데 1등이 무늬(0.35 이상)이고 두 번째 색이 있으면 `patternGuess`([아래](#무늬-8종-yolo26n_pattern8onnx)).
8. **로그**: 단계별 시간(풀기·옷 종류·얼룩·GPT·무늬·색·전체 ms)과 개수만 남깁니다. 사진 내용은 남기지 않습니다.

모델 입력은 학습 때와 같게 비율을 유지해 긴 변 512로 줄이고, 짧은 변은 32의 배수까지만 회색으로 채웁니다. 그래서 확신도가 학습·평가 때(PyTorch)와 같습니다.

## 판단 기준

v6 확신도 기준입니다(정지원 2026-11-09, 근거는 [docs/training-log.md "사진 단위 정답률"](docs/training-log.md#사진-단위-정답률-11-09)).

| v6 확신도 | GPT를 쓸 수 있을 때 | GPT를 쓸 수 없을 때 |
| --- | --- | --- |
| **0.5 이상** | 바로 알림 (`checkedBy: model`) | 바로 알림 (`checkedBy: model`) |
| **0.3 ~ 0.5** | 조각을 GPT에게 물음 → 결함이면 `checkedBy: gpt`, 아니면 버림 | `pendingHelper` ("확실하지 않은 부분이 있어요") |
| **0.2 ~ 0.3** | 위와 같음 | 버림 |
| **0.2 미만** | 버림 | 버림 |

GPT를 쓸 수 없는 경우는 네 가지입니다.

- 사용자가 동의하지 않음(`allowGpt`가 `true`가 아님)
- `OPENAI_API_KEY`가 비어 있음
- GPT가 3초 안에 답하지 않거나 오류가 남(다시 보내지 않음)
- 입은 사진에서 사진 위쪽 25%에 걸친 후보(얼굴이 들어갈 수 있어 보내지 않음)

GPT를 못 쓸 때 0.2~0.3을 버리는 건 "확실하지 않아요"가 너무 잦지 않게 예전 기준(0.3)을 지키려는 것입니다.

## GPT 확인

자세한 내용과 시험 결과는 [docs/gpt-check.md](docs/gpt-check.md)에 있습니다. 코드는 [`verify/`](src/main/java/com/fashionagent/clothesstate/verify/)입니다.

**보내는 것**

- 결함 박스를 사방으로 박스 크기의 30%만큼 넓혀 자른 **조각만** 보냅니다. 64px보다 작으면 64px까지 늘리고, 긴 변은 512px 이하, JPEG 품질 90.
- 옷 전체 사진, 얼굴은 보내지 않습니다.
- 모델 `gpt-4.1-mini`, Chat Completions(`response_format: json_object`, 그림 `detail: high`). 답은 `{"label": "stain" | "tear" | "none", "confidence"}`.
- 동시에 최대 4개, 한 번에 **3초**까지만 기다리고 다시 보내지 않습니다. 사진 한 장 전체도 약 3.5초 안에 끝납니다(agent는 MCP를 20초까지 기다림).
- 프롬프트는 `OpenAiCropVerifier.PROMPT`입니다. 시험과 같은 문장이라 바꾸면 다시 재야 합니다.

**저장하는 것**

- GPT가 얼룩·찢어짐이라고 한 조각만 `data/train-candidates/날짜/아이디.jpg`와 `아이디.json`(종류, 우리 모델 확신도, GPT 확신도, 모델 이름, `reviewed: false`)으로 저장합니다. 사용자 정보는 넣지 않습니다.
- 바로 학습하지 않습니다. 사람이 확인해 `reviewed: true`로 바꾼 것만 다음 학습에 넣고, 새 모델은 시험 점수가 오를 때만 바꿉니다.
- **30일 보관**: 새 날짜 폴더를 만들 때 30일이 지난 날짜 폴더를 지웁니다.
- `data/`는 `.gitignore`에 있어 레포에 올라가지 않습니다.

**시험 결과** (11-09, 공개 데이터 시험 사진 314장, 사진 단위로 결함 있다/없다)

| 방식 | 정답률 | 결함 찾음 | 깨끗한 옷 맞힘 |
| --- | --- | --- | --- |
| v6만 (0.5 / 0.2) | 93.0% | 79.8% | 99.5% |
| **v6 + gpt-4.1-mini** | **98.1%** | **95.2%** | 99.5% |

GPT에게 묻는 사진은 전체의 8.3%, 한 번에 약 0.9초입니다. 시험 사진이 대부분 가까이 찍은 사진이라 옷 전체 사진이나 입은 사진에서는 조금 낮을 수 있습니다.

**사용자 동의**: 앱을 처음 쓸 때 한 번 받습니다. 동의 문구에 "그 부분 조각만 GPT(OpenAI)로 보낸다", "얼룩으로 확인된 조각은 30일 보관해 학습에 쓴다"가 들어가야 합니다. 문구 예는 [docs/gpt-check.md](docs/gpt-check.md#사용자-동의-앱-처음-쓸-때-한-번-정지원-11-09)에 있습니다.

## 모델

얼룩·찢어짐과 옷 종류는 Ultralytics YOLO26n을 우리 데이터로 학습해 ONNX로 내보낸 것이고, 색은 직접 만든 작은 CNN입니다. 모두 `src/main/resources/models/`에 들어 있습니다. YOLO 두 모델은 팀 젯슨(Jetson Orin)에서, 색 모델은 클라우드 작업 공간 CPU에서 학습했고 학습 사진은 외부 AI로 보내지 않았습니다.

### 얼룩·찢어짐: `yolo26n_v6.onnx`

- 클래스 2개: `stain`(얼룩), `torn` → 응답에서는 `tear`.
- v1~v7을 비교해 고른 모델입니다(과정은 [docs/model-comparison.md](docs/model-comparison.md)). 하이퍼파라미터 자동 조정(tune 30회) 설정으로 기본 YOLO26n에서 60에폭 학습했습니다.
- 학습 이미지 8,663장(얼룩 박스 9,527, 찢어짐 박스 7,998, 깨끗한 옷 644장).

| 시험 (v2 test 114장) | 값 |
| --- | --- |
| 전체 mAP50 / mAP50-95 | 0.636 / 0.331 |
| 얼룩 mAP50 / 재현율 | 0.619 / 0.61 |
| 찢어짐 mAP50 / 재현율 | 0.654 / 0.66 |
| 깨끗한 옷 200장 오탐 (0.5 이상) | 1장 |
| 사진 단위 정답률 (v6만, 0.5/0.2, 346장) | 91.9% |
| 사진 단위 정답률 (v6 + GPT, 314장) | 98.1% |

한계: 시험 사진 대부분이 얼룩을 가까이서 찍은 웹 사진입니다. 옷 전체를 펼쳐 찍으면 얼룩이 훨씬 작게 보여 실제 성능은 직접 찍은 사진으로 다시 확인해야 합니다. 찢어짐 시험 박스가 32개뿐이라 찢어짐 수치는 오차가 큽니다. 놓치는 것은 대부분 아주 작거나 흐린 얼룩입니다.

### 옷 종류 12종: `yolo26n_fashion12.onnx`

| 모델 클래스 | `category` | `name` | `formality` | `material` |
| --- | --- | --- | --- | --- |
| shirt | top | 셔츠 | smart | |
| tshirt | top | 티셔츠 | casual | |
| sweater | top | 니트 | unknown | 니트 |
| cardigan | outer | 가디건 | unknown | 니트 |
| jacket | outer | 자켓 | unknown | |
| coat | outer | 코트 | smart | |
| vest | outer | 조끼 | unknown | |
| pants | bottom | 바지 | unknown | |
| shorts | bottom | 반바지 | casual | |
| skirt | bottom | 치마 | unknown | |
| dress | dress | 원피스 | unknown | |
| shoes | shoes | 신발 | unknown | |

격식은 이름만으로 분명한 것만 채우고 나머지는 `unknown`으로 둬서 agent가 고르게 합니다.

코트·재킷이 애매하면(1등이 코트나 재킷이고 다른 쪽 점수가 1등의 0.1배 이상) `name`을 **겉옷**(`category: outer`, `formality: unknown`)으로 냅니다. 틀린 이름을 확신해서 말하지 않기 위해서입니다(`clothes.outer-unsure-ratio`, [training-log](docs/training-log.md#코트재킷이-애매하면-겉옷-11-11)).

**점수** (2026-11-11, fashion12e, 모두 학습에 안 쓴 사진)

- 내려놓은 옷 203장: 종류 정답률 **86.7%**, 큰 분류(상의·하의·겉옷 등) **91.6%**
- 바닥에 놓인 신발 40장: mAP50 **0.991** (예전 모델은 0.015)
- 펼쳐 놓은 중고 의류 449장: mAP50 **0.859** (가디건 0.82, 코트 0.85, 조끼 0.97)
- 시험 2,000장(Fashionpedia, 대부분 입은 사진): mAP50 **0.756**

| 잘하는 종류 (2,000장 mAP50) | 약한 종류 (2,000장 mAP50) |
| --- | --- |
| 바지·원피스·신발·반바지 0.9대, 재킷 0.8대, 치마·티셔츠 0.8 안팎 | 코트 0.68, 니트·셔츠 0.6대, **가디건 0.56, 조끼 0.41** |

약한 곳: 입은 사진의 조끼(겹쳐 입어 일부만 보임)와 바닥에 접어 놓은 코트(7장 중 1장)입니다. 학습 기록은 [docs/training-log.md](docs/training-log.md).

> **재학습 진행 중 (2026-11-10)**: 내려놓은 옷을 보강하려고 중고 의류 데이터로 다시 학습하고 있습니다. 결과가 정해지기 전까지 서버는 위 fashion12를 씁니다.

### 무늬 8종: `yolo26n_pattern8.onnx`

| `pattern` | 뜻 |
| --- | --- |
| plain | 무지 |
| stripe | 줄무늬 |
| check | 체크(하운즈투스·아가일 포함) |
| floral | 꽃·식물 무늬 |
| dot | 도트 |
| graphic | 글자·그림 프린트 |
| animal | 호피·지브라 같은 동물 가죽 무늬 |
| other | 기타 무늬(기하학·페이즐리·카모 등) |
| `null` | 모름(무늬 같지만 확신 0.7 미만), 신발, 모델 없음 |

**점수** (2026-11-11, 학습에 안 쓴 사진)

- 내려놓은 옷 1,959장(중고 의류): 정확도 **0.884**, 무지/무늬 구분 **0.919**. 꽃무늬 0.84, 체크 0.82, 줄무늬 0.74, 도트 0.40
- 입은 옷 7,465개(Fashionpedia): 정확도 0.845
- 기준 0.7일 때 내려놓은 옷에서 무지를 무늬로 말하는 비율 **1.4%**, 무늬 종류를 틀리는 비율 4.5%, 모름 6.2%
- "모름"이면 GPT(gpt-4.1-mini)에게 옷 박스를 물음: 내려놓은 옷 정확도 86.5% → **90.5%**, 모름 6.2% → 0.1%, 응답 중앙값 0.9초. 대신 무지를 무늬로 2.3%, 종류 틀림 7.3%로 조금 늘어 `patternCheckedBy: gpt`로 구분합니다. 입은 사진은 몸·얼굴이 들어갈 수 있어 보내지 않습니다.
- 약한 곳: 도트·기타 무늬·글자그림. 무늬 옷의 약 15%는 무지로 봅니다(특히 줄무늬).

모델은 6MB이고 입력은 옷 박스를 8% 넓혀 자른 뒤 짧은 변 224로 줄여 가운데를 자릅니다(학습 평가와 같음). 학습 기록은 [docs/training-log.md](docs/training-log.md#무늬-모델-11-11).

### 속도

- 젯슨에서 재 보니 TensorRT FP16이 점수는 그대로이고 모델 계산이 약 4배 빨랐습니다(얼룩 + 옷 종류 약 50ms). INT8은 점수가 약 0.1 떨어져 쓰지 않습니다([docs/training-log.md](docs/training-log.md#젯슨-정밀도인트-체크속도메모리-11-09-저녁)).
- **지금 서버는 ONNX Runtime CPU로 돌립니다.** FP16을 쓰려면 젯슨 TensorRT 경로를 따로 붙여야 합니다.

## 실행 방법

필요한 것: **Java 21** (맥은 Homebrew `openjdk@21`). Gradle은 `./gradlew`가 받아 옵니다.

```bash
git clone https://github.com/fashion-agent-team/mcp-clothes-state.git
cd mcp-clothes-state

# 맥에서 Java 21을 못 찾으면
export JAVA_HOME="$(brew --prefix openjdk@21)"

./gradlew bootRun          # http://localhost:8093/mcp
curl localhost:8093/actuator/health
```

### GPT 확인을 켜려면 (`OPENAI_API_KEY`)

키가 없어도 서버는 돌아갑니다. 그때는 GPT에게 묻지 않고 애매한 결함(0.3~0.5)을 `pendingHelper`로 돌려줍니다.

키는 **레포에 넣지 않고** 환경 변수로만 줍니다. 로컬은 레포 루트에 `.env`를 만들어 두면 `.gitignore`에 걸려 올라가지 않습니다.

```bash
# .env (커밋 금지)
OPENAI_API_KEY=여기에_키
```

Spring Boot는 `.env`를 직접 읽지 않으니 셸에 불러온 뒤 실행합니다.

```bash
set -a; source .env; set +a
./gradlew bootRun
```

키를 넣어도 agent가 `allowGpt=true`를 보낸 요청에서만 GPT에게 묻습니다.

### 옷 벡터를 켜려면 (`CLOTHES_EMBED_MODEL`)

Marqo FashionSigLIP vision ONNX(Apache-2.0, 약 372MB)는 GitHub 파일 크기 제한 때문에 레포에 넣지 않습니다. 파일을 받아 경로를 알려 주면 켜지고, 없으면 `embedding` 없이 지금처럼 응답합니다. 입력은 `[batch, 3, 224, 224]`, 출력은 `[batch, 768]`이어야 합니다(batch가 정해져 있지 않으면 원본·뒤집기를 한 번에 넣음). CPU로 옷 한 벌에 약 0.1~0.2초입니다.

```bash
# .env
CLOTHES_EMBED_MODEL=/path/to/marqo-fashionSigLIP-vision.onnx
```

### Docker

```bash
docker build -t ghcr.io/fashion-agent-team/mcp-clothes-state:dev .
docker run --rm -p 8093:8093 -e OPENAI_API_KEY ghcr.io/fashion-agent-team/mcp-clothes-state:dev
```

컨테이너 안에서 학습 후보를 남기려면 `/app/data`에 볼륨을 붙이세요(안 붙이면 컨테이너를 지울 때 함께 사라집니다).

## 설정

`src/main/resources/application.yml`의 `clothes.*`입니다. 환경 변수가 있는 항목은 환경 변수로 바꿀 수 있습니다.

| 설정 | 환경 변수 | 기본값 | 뜻 |
| --- | --- | --- | --- |
| `clothes.model` | `CLOTHES_MODEL` | `classpath:models/yolo26n_v6.onnx` | 얼룩·찢어짐 모델 |
| `clothes.tell-threshold` | | `0.5` | 이 이상 바로 알림 |
| `clothes.ask-threshold` | | `0.2` | 이 이상 애매함(GPT 확인) |
| `clothes.max-image-bytes` | | `8MB` | 사진 최대 크기 |
| `clothes.garment-model` | `CLOTHES_GARMENT_MODEL` | `classpath:models/yolo26n_fashion12.onnx` | 옷 종류 모델. 없으면 사진 전체를 한 벌로 봄 |
| `clothes.garment-classes` | | `shirt,tshirt,…,shoes` (12종) | 학습 클래스 순서. 5종 모델이면 `top,bottom,outer,dress,shoes` |
| `clothes.color-model` | `CLOTHES_COLOR_MODEL` | `classpath:models/color20_cnn.onnx` | 색 인식 모델. `none`이나 없는 파일이면 픽셀 규칙으로 색을 정함 |
| `clothes.garment-threshold` | | `0.4` | 입은 사진에서 옷으로 볼 최소 확신도 |
| `clothes.laid-garment-threshold` | | `0.25` | 내려놓은 옷 사진의 최소 확신도 |
| `clothes.pattern.model` | `CLOTHES_PATTERN_MODEL` | `classpath:models/yolo26n_pattern8.onnx` | 무늬 모델. 없으면 `pattern`은 `null` |
| `clothes.pattern.sure-threshold` | | `0.7` | 1등이 무늬일 때 이 확률 이상이어야 무늬라고 말함 |
| `clothes.embed.model` | `CLOTHES_EMBED_MODEL` | (없음) | 옷 벡터 모델(FashionSigLIP vision ONNX) 경로. 비면 `embedding` 없음 |
| `clothes.embed.version` | | `marqo-fashionSigLIP` | 응답의 `embeddingModel` |
| `clothes.gpt.api-key` | `OPENAI_API_KEY` | (없음) | 비면 GPT를 쓰지 않음 |
| `clothes.gpt.model` | `CLOTHES_GPT_MODEL` | `gpt-4.1-mini` | GPT 모델 |
| `clothes.gpt.candidate-dir` | `CLOTHES_CANDIDATE_DIR` | `data/train-candidates` | 학습 후보 폴더. 비우면 저장 안 함 |
| `clothes.gpt.timeout` | | `3s` | GPT 한 번 기다리는 시간 |
| `clothes.gpt.fallback-ask-threshold` | | `0.3` | GPT를 못 쓸 때 `pendingHelper`로 돌려줄 최소 확신도 |
| `clothes.gpt.worn-skip-top` | | `0.25` | 입은 사진에서 이보다 위에 걸친 조각은 보내지 않음 |
| `clothes.gpt.retention-days` | | `30` | 학습 후보 보관 날수 |

## 테스트

```bash
./gradlew test
```

- 모델 대신 가짜 검출기(`FakeDetector`, `FakeGarmentDetector`), 가짜 벡터 모델(`FakeEmbedder`, 입출력 모양만 같은 작은 ONNX `test/resources/models/tiny-embed.onnx`)과 가짜 GPT(`FakeVerifier`)를 써서 키·네트워크 없이 돕니다. 색 인식 모델(`color20_cnn.onnx`, 0.8MB)만은 `ColorModelTest`에서 실제로 돌립니다.
- 다루는 것: 판단 기준과 `pendingHelper`(`ClothesInspectServiceTest`), MCP 서버로 도구 부르기(`McpServerTest`), YOLO 출력 풀기(`YoloOutputTest`), 옷 종류 모델(`OnnxGarmentDetectorTest`), 위치 말(`WhereTest`), 색(`GarmentColorTest`, `ColorModelTest`), 색 조합(`ColorHarmonyTest`), 사진 풀기(`ImagesTest`), GPT 요청·응답 모양(`OpenAiCropVerifierTest`), 옷 벡터 입력 만들기·정규화(`EmbedInputTest`, `OnnxGarmentEmbedderTest`), 무늬 판단·입력 만들기·GPT 무늬 요청(`OnnxPatternClassifierTest`, `OpenAiPatternVerifierTest`). 가짜 무늬 모델은 `FakePattern`, `FakePatternVerifier`.
- PR과 `develop`·`main` 푸시 때 GitHub Actions(`.github/workflows/ci.yml`)가 Java 21로 같은 테스트를 돌립니다.

## 폴더 구성

```
src/main/java/com/fashionagent/clothesstate/
  ClothesTools.java          MCP 도구 clothes_inspect
  inspect/                   검사 흐름, 응답 모양, 옷 종류 이름, 위치 말(where)
  detect/                    ONNX YOLO 실행(얼룩·찢어짐, 옷 종류)
  verify/                    GPT 확인, 학습 후보 저장
  color/                     색 인식 모델(ColorModel)·픽셀 규칙으로 색 이름, 색 조합 규칙(ColorHarmony)
  pattern/                   무늬 모델 실행(8종), GPT 무늬 확인
  image/                     사진 풀기, letterbox
src/main/resources/
  application.yml
  models/                    yolo26n_v6.onnx, yolo26n_fashion12.onnx, color20_cnn.onnx, yolo26n_pattern8.onnx
src/test/java/               테스트
training/                    학습 스크립트·결과 CSV·데이터 출처(사진·.pt는 넣지 않음)
docs/                        학습 기록, 모델 비교, 성능 기준, GPT 확인
```

### 옷 색 20가지: `color20_cnn.onnx`

- AI Hub K-Fashion 이미지(Validation)의 옷 171,538벌을 정답 상자로 잘라, 사람이 붙인 색 20가지(화이트·블랙·베이지·핑크·그레이·스카이블루·네이비·브라운·블루·카키·레드·그린·라벤더·실버·옐로우·와인·민트·퍼플·오렌지·골드)로 학습한 작은 CNN(입력 64×64, 0.8MB)입니다.
- 시험 옷 25,626벌(학습에 안 씀): 색 계열 정답률 **76.8%** (예전 픽셀 규칙 45.3%), 흰 옷 87% (예전 15%).
- AI Hub 코디 사진 24장(옷 52벌) 눈 확인: 73% → 92%.
- 약한 곳: 네이비(검정·파랑과 헷갈림), 카키, 체크·플로럴 무늬 옷, 내려놓은 흰 옷.
- 과정과 비교는 [training/color-model](training/color-model/README.md), [docs/lab.md](docs/lab.md#6-옷-색-인식).

## 데이터 출처와 라이선스

이미지는 레포에 올리지 않습니다. 전체 표는 [training/DATA_SOURCES.md](training/DATA_SOURCES.md)에 있고, `training/scripts/`로 데이터를 다시 만들 수 있습니다.

**얼룩·찢어짐 (v6)**: 모두 Roboflow Universe

| 데이터셋 | 라이선스 | 쓰임 |
| --- | --- | --- |
| Clothes-stain (laundrycareteam1) | CC BY 4.0 | 얼룩 학습 |
| More clothes (Memorial University of Newfoundland) | CC BY 4.0 | 찢어짐 학습, 깨끗한 옷·얼룩 합성 바탕 |
| stain (s-workspace-okn3h), stain (devadharshini) | CC BY 4.0 | 얼룩 학습 |
| hole (swee), Hole (s-workspace-wsqvz) | CC BY 4.0 | 찢어짐 학습·검증·시험 |
| CLOTHES (s-workspace-wsqvz) | Public Domain | 깨끗한 옷 음성, 오탐 평가 200장 |
| stain (aleminaitha) | CC BY 4.0 | 학습 제외, 실제 옷 전체 사진 32장 평가 |

**옷 색 (color20)**: AI Hub [K-Fashion 이미지](https://aihub.or.kr) Validation(과학기술정보통신부·한국지능정보사회진흥원). 사진·조각은 재배포하지 않습니다.

**옷 종류 (fashion12)**

| 데이터셋 | 라이선스 | 쓰임 |
| --- | --- | --- |
| [Fashionpedia](https://fashionpedia.github.io/home/) | 주석 CC BY 4.0 (사진은 원래 사이트 조건) | 학습 43,623장, 검증 1,158장, 시험 2,000장 |
| [Clothing dataset](https://github.com/alexeygrigorev/clothing-dataset) (Alexey Grigorev) | CC0 | 내려놓거나 옷걸이에 건 옷. 학습 4,104장, 평가 208장 |
| More clothes·얼룩 데이터의 옷 전체 사진 | CC BY 4.0 | 직접 검토한 내려놓은 옷 61장 학습, 15장 평가 |

**무늬 (pattern8)**

| 데이터셋 | 라이선스 | 쓰임 |
| --- | --- | --- |
| [Fashionpedia](https://fashionpedia.github.io/home/) 무늬 속성 | 주석 CC BY 4.0 (사진은 원래 사이트 조건) | 옷 상자를 잘라 학습 21,085개, 시험 7,465개 |
| [Clothing Dataset for Second-Hand Fashion v3](https://zenodo.org/records/13788681) | CC BY 4.0 | 내려놓은 옷 무늬. 학습 약 1만 장, 시험 1,959장(공식 test) |

**옷 벡터**: [Marqo FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP)(Apache-2.0)을 그대로 씁니다(학습하지 않음). 옷장 쪽 보정 행렬(WCCN)은 비상업 데이터(DeepFashion)로 만들어 이 레포에 넣지 않습니다.

**사전학습 가중치**: Ultralytics `yolo26n.pt` 등. Ultralytics 코드와 가중치는 **AGPL-3.0**입니다(상용 배포 시 라이선스 확인 필요).

CC BY 4.0 데이터로 학습한 모델이나 결과를 공개할 때는 위 출처 표기를 함께 남깁니다.

## 문서

| 문서 | 내용 |
| --- | --- |
| [docs/lab.md](docs/lab.md) | 모든 실험을 같은 시험 기준 점수표로 모은 비교표(채택·비교·탈락과 이유) |
| [docs/training-log.md](docs/training-log.md) | 학습·실험 기록(옷 종류 모델, 사진 단위 정답률, 젯슨 FP16·INT8 비교) |
| [docs/model-comparison.md](docs/model-comparison.md) | 얼룩·찢어짐 모델 v1~v7 비교와 v6를 고른 이유 |
| [docs/benchmark.md](docs/benchmark.md) | 용어 설명, 모델 고르는 기준, 재는 방법 |
| [docs/color-rules.md](docs/color-rules.md) | 색 조합·자리별 색 규칙(`outfit_color_check`), 왜 규칙인지, 예시 출력 |
| [docs/gpt-check.md](docs/gpt-check.md) | GPT 확인 흐름, 보내는 것·저장하는 것, 동의 문구, 시험 결과 |
| [training/README.md](training/README.md) | v6 학습 설정, 학습 데이터 구성, 스크립트 돌리는 순서 |
| [training/DATA_SOURCES.md](training/DATA_SOURCES.md) | 데이터 출처·라이선스 전체 |

모든 레포가 지키는 공통 약속은 앱 레포에 있습니다. **코드보다 약속 문서를 먼저 고칩니다.**

- [서비스 간 약속](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/README.md)
- [MCP 도구 입출력](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md)
- [앱 ↔ 게이트웨이 API (OpenAPI)](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/openapi.yaml)
- [아키텍처와 기술 스택](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/ARCHITECTURE.md)

## 남은 일

- [ ] 옷 종류 재학습(중고 의류 데이터) 결과 확인 후 모델 교체 여부 결정 — 진행 중
- [ ] 팀이 직접 찍은 펼쳐 놓은 옷 사진(얼룩·찢어짐·깨끗한 옷)으로 다시 확인
- [ ] 입은 사진 위치를 자세 인식(pose)으로 더 정확하게
- [ ] 같은 옷에서 "아니야"라고 한 자리 다시 안 묻기(약속 문서에 입력 추가 필요)
- [ ] 젯슨 TensorRT FP16으로 돌리는 경로
- [x] 색 조합 규칙을 AI Hub 코디 선호도 데이터로 검증([색 조합 규칙](docs/color-rules.md))
- [x] 색 인식 학습(AI Hub K-Fashion, [color20](training/color-model/README.md))
- [ ] 무늬 옷(체크·플로럴) 색 판단을 무늬 인식 결과로 약하게, 내려놓은 옷 색 보강
- [ ] 무늬: 팀이 찍은 내려놓은 옷 사진과 K-Fashion(한국 옷) 라벨로 다시 확인, 도트·줄무늬 보강(320픽셀 재학습 후보), GPT에게 물어 쌓인 후보로 재학습
- [ ] 뒷모습 입은 사진 시험
- [ ] 실제 FashionSigLIP 모델로 이슈 #15 측정 스크립트와 벡터가 같은지 확인
- 다른 레포: agent가 확인 단계에서 `outfit_color_check`를 부르기, 약속 문서(mcp-tools.md)에 도구 추가
- 다른 레포: 옷장은 `embedding`을 pgvector에 저장(박창근 님, 이슈 #15). 약속 문서(mcp-tools.md)에 `embedding`·`embeddingModel` 추가 필요
- 다른 레포: 약속 문서(mcp-tools.md)에 `pattern` 값 8종·`null`(모름)과 `patternCheckedBy` 추가 필요. agent가 무늬끼리 겹침, 무늬 옷 색 판단에 쓰는 건 agent 담당과 정함
- 다른 레포: agent는 동의한 사용자에게만 `allowGpt=true`를 보내고 `checkedBy: gpt`는 약하게 말하기, 앱은 처음 쓸 때 동의 화면

## 브랜치와 비밀 값

- GitFlow: `develop`에서 브랜치를 따서 `develop`으로 PR을 올립니다. 규칙은 앱 레포 `docs/GITFLOW.md`.
- API 키는 레포에 넣지 않습니다. 로컬은 `.env`(커밋 금지), 배포는 서버 환경 변수. 목록은 앱 레포 `docs/infra/.env.example`.
