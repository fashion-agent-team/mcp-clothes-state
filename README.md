# mcp-clothes-state

**clothes-state MCP 서버 · 옷 사진 분석** · 패션 에이전트(시각장애인용 음성 AI 옷 추천)의 한 부분입니다.

담당: (미정)

## 한눈에

- 포트: 8093, 엔드포인트: `http://mcp-clothes-state:8093/mcp` (Streamable HTTP)
- 기술: Spring Boot, Spring AI MCP Server, YOLO 파인튜닝(얼룩·찢어짐·보풀), 옷 종류 YOLO(Fashionpedia CC BY 4.0·Clothing dataset CC0로 직접 학습, 12종), ONNX Runtime Java. 애매한 결함(0.2~0.5)은 동의한 사용자만 그 부분 조각을 GPT(gpt-4.1-mini)로 확인([docs/gpt-check.md](docs/gpt-check.md)). 학습은 Python(Ultralytics, Roboflow 라벨링, Colab)
- 상태 확인: `GET /actuator/health`

## MCP 도구

| 도구 | 하는 일 |
| --- | --- |
| `clothes_inspect` | 사진 한 장 → 옷마다 종류·색·격식·위치(left·center·right)·얼룩/찢어짐, 구도(framing), 조명(lighting), source(model/friend) |

입출력 JSON은 [MCP 도구 입출력](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md)를 그대로 따릅니다.

## 이 레포가 지킬 것

- 옷 추천 2단계에서 앱이 자동 전송한 한 장마다 agent가 부릅니다. 한 장에 두 벌이 있을 수 있습니다(비교).
- **얼룩을 찾는 것까지가 이 서버 일**이고, 어떻게 말할지(빨기·바꾸기)는 agent의 GPT가 정합니다(사진은 넘기지 않고 결과 글만).
- 응답 모양은 지금 Workers 서버의 ClothesScan에 `position`, `source`, `modelVersion`을 더한 것. 왼쪽·오른쪽은 펼친 옷(laid)이면 사진 기준(사진 왼쪽 = 사용자 왼쪽), 입은 사진(worn)이면 입은 사람 몸 기준.
- 결함 확신도: 0.5 이상 바로 알림, 0.3~0.5는 지인(helper)에게 확인, 0.3 미만 버림(2026-11-08 확정). 옷 사진은 GPT·Gemini로 보내지 않음.
- 옷 전체 사진은 저장하지 않습니다. GPT가 결함이라고 한 조각만 다시 학습할 후보로 저장합니다([docs/gpt-check.md](docs/gpt-check.md)).
- 공개 원단 결함 데이터(AITEX 등)는 공장 원단 사진이라 우리 옷 사진을 직접 찍어 라벨링합니다(결함 종류마다 수백 장 목표).

## 할 일

> 모델 비교 결과(YOLO11·YOLO26, 2026-11-05~08): [docs/model-comparison.md](docs/model-comparison.md). 쓰는 모델은 YOLO26n v6.

- [ ] 학습용 사진 찍기·라벨링 (다 같이)
- [x] `training/`: YOLO 학습, ONNX 내보내기 (v6, 출처는 [training/DATA_SOURCES.md](training/DATA_SOURCES.md))
- [x] Spring Boot + Spring AI MCP Server, `clothes_inspect` (결함 찾기·위치·지인 확인 후보)
- [ ] 입은 사진 위치를 자세 인식(pose)으로 더 정확하게(지금은 사진을 셋으로 나눈 대략적 위치)
- [ ] 같은 옷에서 지인이 "아니야"라고 한 자리 다시 안 묻기(약속 문서에 입력 추가 필요)
- [x] 픽셀로 색 이름
- [x] 옷 종류(상의·하의·아우터·원피스·신발) — 서버 연결. 모델 파일은 아직 레포에 없음
- [ ] 내려놓은 옷 종류 보강 학습(지금 24장 중 17~18장만 맞음)
- [x] 격식(이름으로 분명한 것만)
- [ ] 세부 12종 모델 결과 확인·모델 파일 올리기
- [x] ~~Gemini 폴백~~ 쓰지 않기로 함(2026-11-06, 옷 사진은 외부 AI로 보내지 않음)
- [ ] 모델 준비 전까지 아래 Workers 서버를 데모용 예비로 유지

## Java MCP 서버 (clothes_inspect)

`src/main/java`. 우리가 학습한 YOLO26n v6(`src/main/resources/models/yolo26n_v6.onnx`)를 ONNX Runtime으로 서버 안에서 돌립니다. 옷 전체 사진은 외부로 보내지 않고 저장하지 않습니다. 애매한 결함 조각만 GPT로 확인합니다(동의한 사용자, `OPENAI_API_KEY`).

```bash
./gradlew test       # 테스트(모델 대신 가짜 검출기를 씀)
./gradlew bootRun    # http://localhost:8093/mcp, 상태 확인 GET /actuator/health
```

- 확신도 0.5 이상은 `defects`(바로 알림), 0.3~0.5는 `pendingHelper.candidates`(지인 확인 후보), 0.3 미만은 버립니다. `application.yml`의 `clothes.*`에서 바꿉니다.
- 사진은 학습 때와 같게 비율을 유지해 긴 변 512, 짧은 변은 32의 배수까지만 회색으로 채워 넣습니다. 그래서 확신도가 학습·평가 때(PyTorch)와 같습니다.
- 색(`colorFamily`·`colorName`·`tone`)은 학습 없이 픽셀로 정합니다. 옷 박스가 있으면 박스 안에서 재고, 없으면(옷 종류 모델이 없거나 못 찾음) 화면 가운데 영역의 색을 세 무리로 나눠 가장 큰 무리를 씁니다. 가운데를 둘러싼 띠의 절반 넘게 차지한 색(흰 배경판 등)은 배경으로 보고 건너뛰고, 입은 사진은 피부색을 뺍니다. 화면에 띄운 사진을 비춰도 모니터 테두리를 바닥으로 잘못 잡지 않습니다(2026-11-09 리허설). 노란 조명이나 무늬 옷은 틀릴 수 있습니다.
- 옷 종류는 옷 종류 모델(`yolo26n_fashion12.onnx`, Fashionpedia CC BY 4.0·Clothing dataset CC0로 학습, 셔츠·티셔츠·니트·가디건·자켓·코트·조끼·바지·반바지·치마·원피스·신발)이 옷 박스를 찾아 정합니다. 세부 종류는 `category`와 `name`(예: top·니트)으로 바꾸고, 이름만으로 분명한 격식(셔츠·코트 smart, 티셔츠·반바지 casual)과 소재(니트·가디건은 니트)를 채웁니다. 나머지 격식은 `unknown`입니다. 5종 모델을 쓰려면 `clothes.garment-classes=top,bottom,outer,dress,shoes`. 입은 사진은 확신도 0.4 이상, 내려놓은 옷은 0.25 이상을 옷으로 보고(내려놓은 옷은 모델이 확신도를 낮게 주고, 종류는 틀려도 위험하지 않아서), 종류가 달라도 박스가 거의 같으면(IoU 0.7) 한 벌로 합칩니다. 순서는 내려놓은 옷은 왼쪽부터, 입은 사진은 위쪽부터 `g1`, `g2`…
- 색은 그 옷 박스 안에서 재고, `position`은 박스 중심으로 정합니다(0.4 미만 left, 0.6 초과 right).
- 얼룩·찢어짐은 중심이 들어가는 옷(겹치면 더 작은 옷)에 붙이고, `where`는 그 옷 박스 안 위치로 말합니다. `pendingHelper`는 한 벌에 한 번 묻는 약속이라 가장 확신도 높은 후보가 있는 옷 한 벌만 담습니다.
- 옷 종류 모델 파일이 없거나 옷을 못 찾으면 예전처럼 사진 전체를 옷 한 벌(`g1`, `category: other`)로 봅니다. 
- 12종 모델(2026-11-09): 시험 2,000장 mAP50 0.756, 내려놓은 옷 203장 종류 정답률 80.3%(큰 분류 87.7%). 바지·원피스·신발·반바지·재킷·치마·티셔츠는 0.8 이상, 가디건 0.53·조끼 0.36은 약합니다([docs/training-log.md](docs/training-log.md)).
- `where`: `laid`는 사진 기준(사진 왼쪽 = 사용자 왼쪽), `worn`은 입은 사람 몸 기준(앞모습은 사진 오른쪽이 그 사람의 왼쪽).
- 어두우면 `lighting: dark`, 모델 파일이 없으면 `{"error": {"code": "not_ready"}}`.
- 모델은 CC BY 4.0·CC0 데이터와 Ultralytics(AGPL-3.0) 사전학습 가중치로 학습했습니다. 출처는 [training/DATA_SOURCES.md](training/DATA_SOURCES.md).

## 지금 있는 코드 (임시, Cloudflare Workers + Gemini)

아래는 앱 레포 `server/`에서 옮겨 온 예전 서버입니다. 위 Java MCP 서버로 바꾸면 정리합니다.

앱과 비전 LLM 사이에 두는 작은 서버입니다. **API 키는 여기에만 두고 앱에는 넣지 않습니다.** 프롬프트와 응답 형식도 여기서 관리해서, 앱을 다시 배포하지 않고 고칠 수 있게 합니다.

Cloudflare Workers + Gemini API로 만든 뼈대입니다. 배포 주소는 `https://fashion-agent-vision.fashion-agent-team.workers.dev`이고, 앱은 기본으로 이 주소를 씁니다. 서버 없이 돌리려면 `--dart-define=FAKE_VISION=true`를 주면 `FakeVisionService`(예시 응답)로 돕니다.

### 파일

| 파일 | 하는 일 |
| --- | --- |
| `src/index.ts` | 요청 받기. 주소·purpose·사진 크기·앱 토큰 확인, 오류 코드 정리 |
| `src/prompts.ts` | 옷 검사(outfit) 프롬프트. 격식·얼룩/찢어짐 기준 포함. **문구 다듬기는 여기서** |
| `src/gemini.ts` | Gemini 호출. 응답을 아래 JSON 형식으로만 받도록 강제 |
| `src/schema.ts` | 응답 형식 정의와 정리(모르는 값은 기본값, id는 g1·g2… 부여) |
| `test/` | 가짜 Gemini로 하는 테스트 |

### 실행

Node 22.6 이상이 필요합니다.

```bash
cd server
npm install
npm test                       # 가짜 Gemini로 테스트
cp .dev.vars.example .dev.vars # GEMINI_API_KEY 채우기 (커밋 금지)
npm run dev                    # http://localhost:8787
```

앱을 로컬 서버에 붙이기:

```bash
flutter run --dart-define=VISION_URL=http://localhost:8787
## 안드로이드 에뮬레이터는 http://10.0.2.2:8787
```

배포:

```bash
npx wrangler login
npx wrangler secret put GEMINI_API_KEY
npm run deploy                 # 나온 https://….workers.dev 주소를 VISION_URL로
```

모델은 `wrangler.toml`의 `GEMINI_MODEL`에서 바꿉니다. `gemini-2.5-flash`는 새로 발급한 키로는 404가 나서 `gemini-3.5-flash`를 기본으로 둡니다(2026-11-03 실제 키로 확인). 사용 가능한 모델은 `GET /v1beta/models`로 볼 수 있습니다. `APP_TOKEN` 비밀 값을 넣으면 `X-App-Token` 헤더가 같은 요청만 받습니다. 앱은 아직 이 헤더를 보내지 않으니, 토큰을 켜기 전에 `HttpVisionService`에 헤더를 추가해야 합니다.

### 오류 응답

200이 아니면 앱은 "분석하지 못했어요"라고 안내하고 다시 찍게 합니다. 본문은 `{"error": {"code", "message"}}`입니다.

| 상태 | code | 언제 |
| --- | --- | --- |
| 400 | `bad_json`, `bad_purpose`, `no_image` | 요청 형식이 틀림 |
| 401 | `unauthorized` | 앱 토큰이 다름 |
| 413 | `image_too_large` | 사진이 약 6MB 넘음 |
| 502 | `ai_failed`, `ai_bad_response` | Gemini 오류(무료 한도 초과 429 포함), 형식이 이상한 응답. `message`에 구글이 준 이유가 붙음 |
| 504 | `ai_timeout` | 25초 안에 답이 없음 |

`ai_failed`일 때 이유 보기: `message`의 `Gemini 오류 (400): …` 뒷부분을 보거나, 다른 터미널에서 `npx wrangler tail`을 켜 두고 요청을 다시 보내면 로그에 찍힙니다. 흔한 이유는 `API key not valid`(비밀 값을 잘못 붙여 넣음, `npx wrangler secret put GEMINI_API_KEY`로 다시 넣기), 모델 이름이 없음(404), 사진이 JPEG가 아님(`Unable to process input image`)입니다.

### 남은 일

- TODO(AI 담당): 실제 옷 사진으로 프롬프트 다듬기, `confidence` 기준이 맞는지 확인
- TODO: 앱 `HttpVisionService`에 `X-App-Token` 헤더 추가 후 토큰 켜기
- TODO: 무료 한도 초과(429)나 구글 쪽 혼잡(503) 때 한 번 재시도할지, 사용자에게 따로 안내할지 정하기
- TODO(AI 담당): 손상 위치 `where`가 "사진 기준 오른쪽"처럼 나올 수 있음. 펼쳐 놓은 옷에서 왼쪽·오른쪽을 무엇 기준으로 말할지 프롬프트에 정하기

### 엔드포인트

#### `POST /describe-clothes`

옷 검사에서 펼쳐 놓은 옷을 분석합니다. 일정·날씨에 맞는지 판단은 앱이 하고, AI는 격식과 얼룩·찢어짐 같은 보이는 사실만 돌려줍니다.

요청

```json
{
  "purpose": "outfit",        // 지금은 outfit 하나
  "image": "<base64 JPEG>"
}
```

응답 (앱의 `ClothesScan.fromJson`과 같은 모양)

```json
{
  "garments": [
    {
      "id": "g1",
      "category": "top",            // top | bottom | outer | dress | shoes | other
      "name": "셔츠",
      "colorName": "남색",          // 사용자에게 그대로 읽어 줄 색 이름
      "colorFamily": "navy",        // black white gray navy beige brown khaki blue red pink orange yellow green purple unknown
      "tone": "dark",               // light | mid | dark
      "pattern": "줄무늬",          // 무늬 없으면 null
      "material": "면 같음",        // 사진으로 본 추정. 모르면 null
      "notes": "약간 구김 있음",    // 그 밖의 상태. 없으면 null
      "formality": "smart",         // formal(정장·구두) | smart(셔츠·니트·슬랙스) | casual(티·청바지) | sporty(운동복) | unknown
      "defects": [                  // 얼룩·찢어짐. 없으면 []
        { "kind": "stain", "where": "오른쪽 소매 끝", "confidence": 0.85 }  // kind: stain | tear | lint | other
      ],
      "box": { "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 },  // 사진 크기를 1로 본 비율
      "confidence": 0.86            // 0~1. 색을 얼마나 확신하는지
    }
  ],
  "framing": null,                  // 구도가 안 맞으면 "옷이 오른쪽에 잘렸어요. 폰을 오른쪽으로 조금 옮겨 주세요."
  "lighting": "ok"                  // ok | dark
}
```

### 프롬프트 원칙

- **본 사실만** 답하게 합니다. "어울린다", "예쁘다" 같은 판단은 묻지도 받지도 않습니다. 판단은 앱의 `lib/styling/` 규칙이 합니다.
- `defects`는 사진에서 실제로 보이는 것만 넣게 합니다. `where`는 "오른쪽 소매 끝", "밑단 왼쪽"처럼 손으로 찾아갈 수 있는 말로 받습니다. 애매하면 `confidence`를 낮추게 합니다. 앱은 0.7 미만이면 "손으로 확인해 보세요"라고만 말합니다.
- 색이 조명 때문에 애매하면 `confidence`를 낮추게 합니다. 앱은 0.6 미만이면 판단하지 않고 다시 비춰 달라고 합니다.
- `framing`은 "흐려요"가 아니라 "폰을 오른쪽으로", "한 걸음 뒤로"처럼 사용자가 바로 따라 할 수 있는 문장으로 받습니다.
- 응답은 위 JSON 스키마로 강제합니다(LLM의 구조화 출력 기능 사용).

## 공통 약속

모든 레포가 지키는 약속은 앱 레포에 있습니다. **코드보다 약속 문서를 먼저 고칩니다.**

- [서비스 간 약속](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/README.md): 공통 규칙, 포트, 인증(`X-User-Id`), 오류 모양, DB 스키마, 옷 추천 흐름
- [MCP 도구 입출력](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md)
- [앱 ↔ 게이트웨이 API (OpenAPI)](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/openapi.yaml)
- [아키텍처와 기술 스택](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/ARCHITECTURE.md)

## 버전

| 항목 | 버전 |
| --- | --- |
| Java | 21 |
| Spring Boot | 4.0.9 |
| Spring Cloud | 2025.1.4 |
| Spring AI (BOM) | 2.0.2 |

## 브랜치

GitFlow를 따릅니다. `develop`에서 `feature/…` 브랜치를 따서 PR로 합칩니다. 규칙은 앱 레포의 `docs/GITFLOW.md`를 봅니다.

## 비밀 값

API 키와 DB 비밀번호는 레포에 넣지 않습니다. 로컬은 `.env`(커밋 금지), 배포는 서버 환경 변수로 둡니다. 목록은 앱 레포 `docs/infra/.env.example`에 있습니다.
