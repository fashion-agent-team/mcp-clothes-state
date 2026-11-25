"""SigLIP2 So400m 이미지 벡터 서버(옷 종류 보조 경로용, 젯슨 GPU).

mcp-clothes-state가 아주 애매한 옷의 박스 조각(이미 넓혀 자르고 회색으로 정사각형을 채운 JPEG)을 보내면,
224로 줄이고 원본과 좌우 뒤집은 것의 벡터를 평균해 길이 1인 1152차원 벡터를 돌려줍니다.

메모리를 아끼려고 이미지 부분만 fp16으로 GPU에 올립니다(글 부분은 안 씀). 2026-11-17 젯슨 오린 나노 측정:
가중치 827MB, 프로세스 약 1.95GB, 박스 하나(2장) 166ms.

POST /embed {"imageBase64": "..."} -> {"vector": [1152 floats], "ms": 166}
GET  /health -> {"status": "ok", "model": "..."}
"""
import base64
import io
import os
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

ARCH = os.environ.get("SO400M_ARCH", "ViT-SO400M-14-SigLIP2")
# open_clip_model.safetensors 경로(호스트의 Hugging Face 캐시를 읽기 전용으로 붙임). 비우면 HF 캐시에서 찾음.
WEIGHTS = os.environ.get("SO400M_WEIGHTS", "")
SIZE = 224
MAX_BYTES = 2 * 1024 * 1024
ALLOWED_FORMATS = ("PNG", "JPEG")

app = FastAPI(title="so400m-embed")
_lock = threading.Lock()  # GPU는 한 번에 하나씩
_model = None


def _weights_path():
    if WEIGHTS:
        return WEIGHTS
    from huggingface_hub import hf_hub_download
    return hf_hub_download("timm/" + ARCH, "open_clip_model.safetensors", local_files_only=True)


def load():
    """이미지 탑만 meta로 만들고 safetensors에서 visual.* 만 fp16으로 GPU에 바로 올림(CPU 메모리를 거의 안 씀)."""
    import open_clip
    from open_clip.model import _build_vision_tower
    from safetensors import safe_open
    cfg = open_clip.get_model_config(ARCH)
    with torch.device("meta"):
        vis = _build_vision_tower(cfg["embed_dim"], cfg["vision_cfg"])
    sd = {}
    with safe_open(_weights_path(), "pt") as f:
        for k in f.keys():
            if k.startswith("visual."):
                sd[k[len("visual."):]] = f.get_tensor(k).half().cuda()
    missing = vis.load_state_dict(sd, strict=False, assign=True)
    if missing.missing_keys:
        raise RuntimeError(f"weights missing: {missing.missing_keys[:5]}")
    vis.eval().cuda()
    mean = np.array(vis.preprocess_cfg["mean"], np.float32) if hasattr(vis, "preprocess_cfg") else np.full(3, 0.5, np.float32)
    std = np.array(vis.preprocess_cfg["std"], np.float32) if hasattr(vis, "preprocess_cfg") else np.full(3, 0.5, np.float32)
    return vis, mean, std


class EmbedRequest(BaseModel):
    imageBase64: str


@app.on_event("startup")
def startup():
    global _model
    t = time.time()
    _model = load()
    # 데우기
    embed_image(Image.new("RGB", (SIZE, SIZE), (124, 116, 104)))
    print(f"So400m 이미지 탑을 올렸어요: {time.time() - t:.1f}초, GPU {torch.cuda.memory_allocated() // 2**20}MB", flush=True)


def embed_image(im: Image.Image) -> np.ndarray:
    vis, mean, std = _model
    im = im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    x = (np.asarray(im, np.float32) / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)
    batch = np.stack([x, x[:, :, ::-1].copy()])
    with _lock, torch.no_grad():
        v = vis(torch.from_numpy(batch).cuda().half()).float().cpu().numpy()
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[0] + v[1]
    return v / np.linalg.norm(v)


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(503, "loading")
    return {"status": "ok", "model": ARCH}


@app.post("/embed")
def embed(req: EmbedRequest):
    if _model is None:
        raise HTTPException(503, "loading")
    try:
        raw = base64.b64decode(req.imageBase64, validate=True)
        if len(raw) > MAX_BYTES:
            raise ValueError("too large")
        # clothes-state는 PNG, 테스트·예전 호출은 JPEG만 보냅니다. 다른 형식의 디코더는 아예 열지 않아
        # pillow 디코더 취약점(osv-scanner.toml 예외 참고)이 닿는 길을 줄입니다.
        im = Image.open(io.BytesIO(raw), formats=ALLOWED_FORMATS)
        im.load()
    except Exception:
        raise HTTPException(400, "bad image")
    t = time.time()
    v = embed_image(im)
    return {"vector": [round(float(a), 6) for a in v], "ms": round((time.time() - t) * 1000)}
