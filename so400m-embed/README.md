# so400m-embed

SigLIP2 So400m 이미지 벡터 서버예요. 옷 종류 보조 경로에 쓰고, 아직 배포하지 않았어요.

`mcp-clothes-state`는 옷 종류가 아주 애매한 옷만 이 서버에 물어요. 조건은 두 가지예요.
- 옷 종류 모델 확신도 < 0.7
- probe_v4 종류 점수 1·2등 차이 < 0.015

이 서버가 돌려준 벡터와 `models/garment-kind-so400m-probe.json`으로 종류를 다시 골라요. 서버가 없거나 800ms 안에 답하지 않으면 지금 결과를 그대로 써요.

## 근거 (training/probe/README.md 12차, 2026-11-17)
- 폰 사진(DF2 validation) 1,841장에서 옷 15%만 다시 물어도 세부 정답률이 81.4%에서 83.1%로 올라요. Fashionpedia val은 92.6%에서 93.0%예요.
- 젯슨 오린 나노에서 이미지 탑만 fp16 GPU에 올렸을 때 가중치는 827MB, 프로세스는 약 1.95GB예요. 박스 하나(좌우 2장)에 166ms가 걸려요.
- probe는 DeepFashion2 train(비상업 연구·교육용)도 써서 학습했어요. 그래서 **학교 프로젝트 전용**이에요.

## 붙이는 법 (2026-11-19 team4에 이렇게 배포함)
compose 파일은 지금 젯슨에 있는 파일에서 필요한 줄만 고쳐요. 고치기 전에 백업해요.

```yaml
  so400m-embed:
    build: ../so400m-embed
    image: fashion-agent/so400m-embed:local
    runtime: nvidia
    mem_limit: 2560m
    memswap_limit: 2560m      # 스왑 금지(다른 서버가 느려지지 않게)
    ports: ["100.90.21.23:8096:8096"]   # Tailscale IP에만
    environment:
      SO400M_WEIGHTS: /models/so400m/open_clip_model.safetensors
      NVIDIA_VISIBLE_DEVICES: all          # 없으면 "Found no NVIDIA driver"
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
    volumes:
      - ../models/so400m:/models/so400m:ro
  mcp-clothes-state:
    environment:
      CLOTHES_SO400M_URL: http://so400m-embed:8096
```

`CLOTHES_SO400M_URL`을 비우면 보조 경로가 꺼져요(기본값). 기다리는 시간은 `CLOTHES_SO400M_TIMEOUT`(기본 800ms)으로 바꿔요.

## 확인
```bash
curl -s localhost:8096/health
python3 -c "import base64,json,urllib.request;b=base64.b64encode(open('shirt.jpg','rb').read()).decode();print(json.loads(urllib.request.urlopen(urllib.request.Request('http://localhost:8096/embed',json.dumps({'imageBase64':b}).encode(),{'content-type':'application/json'})).read())['ms'])"
```

## 2026-11-19 배포 측정 (team4 Orin Nano, L4T R39)
- 이미지는 9.5GB예요. torch cu132와 nvidia 휠이 들어 있어요. 시작에 9.6초 걸리고, 이미지 탑은 GPU 827MB, 컨테이너 메모리는 1.79GB예요.
- 나노에서 /embed를 불러 보면 서버 처리 86ms, 왕복 약 111ms예요. 첫 호출은 328ms였어요.
- clothes_inspect 로그는 "So400m 다시 묻기 1벌 중 1벌 답, 132~362ms"였어요. 애매한 사진 12장의 응답 중간값은 2.20초에서 2.27초로 늘었어요.
