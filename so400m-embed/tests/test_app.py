"""so400m-embed 서버 테스트(모델 없이). HTTP 계층(/health, /embed 입력 검사)과 embed_image의 계산(좌우 평균, 길이 1)을 봅니다.

실행: cd so400m-embed && pip install -r tests/requirements.txt && python -m pytest tests -q
"""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as server


def fake_vis(batch):
    """가짜 이미지 탑: 사진 왼쪽 절반과 오른쪽 절반의 채널 평균(6개)에 조금 다른 무게를 줘서 '벡터'를 만든다.

    좌우를 뒤집으면 왼쪽·오른쪽 값이 서로 바뀌므로, 원본과 뒤집은 것을 평균하는지 확인할 수 있다.
    """
    x = batch.a if hasattr(batch, "a") else batch.numpy()
    half = x.shape[3] // 2
    left = x[:, :, :, :half].mean(axis=(2, 3))
    right = x[:, :, :, half:].mean(axis=(2, 3))
    v = np.concatenate([left * 1.0 + 2.0, right * 3.0 + 1.0], axis=1)
    import torch
    return torch.from_numpy(v.astype(np.float32))


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.setattr(server, "_model", (fake_vis, np.full(3, 0.5, np.float32), np.full(3, 0.5, np.float32)))


@pytest.fixture
def client():
    # with 블록 없이 만들면 startup(진짜 모델 올리기)이 돌지 않는다.
    return TestClient(server.app)


def jpeg_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def two_tone() -> Image.Image:
    """왼쪽은 빨강, 오른쪽은 파랑인 사진."""
    im = Image.new("RGB", (64, 64), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 32, 64))
    return im


def test_health_모델_올리기_전에는_503(client, monkeypatch):
    monkeypatch.setattr(server, "_model", None)
    assert client.get("/health").status_code == 503


def test_health_모델이_있으면_ok와_모델_이름(client, loaded):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": server.ARCH}


def test_embed_모델_올리기_전에는_503(client, monkeypatch):
    monkeypatch.setattr(server, "_model", None)
    assert client.post("/embed", json={"imageBase64": jpeg_b64(two_tone())}).status_code == 503


@pytest.mark.parametrize("payload", ["!!!not-base64!!!", base64.b64encode(b"not an image").decode()])
def test_embed_base64가_아니거나_사진이_아니면_400(client, loaded, payload):
    res = client.post("/embed", json={"imageBase64": payload})
    assert res.status_code == 400
    assert res.json()["detail"] == "bad image"


def test_embed_2MB_넘는_사진은_400(client, loaded):
    big = base64.b64encode(b"\xff" * (server.MAX_BYTES + 1)).decode()
    assert client.post("/embed", json={"imageBase64": big}).status_code == 400


def test_embed_imageBase64가_없으면_422(client, loaded):
    assert client.post("/embed", json={}).status_code == 422


def test_embed_정상_사진은_길이_1인_벡터와_ms(client, loaded):
    res = client.post("/embed", json={"imageBase64": jpeg_b64(two_tone())})
    assert res.status_code == 200
    body = res.json()
    v = np.array(body["vector"])
    assert v.shape == (6,)  # 가짜 vis 차원(진짜는 1152)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4
    assert isinstance(body["ms"], int) and body["ms"] >= 0


def test_embed_image_좌우를_뒤집어도_같은_벡터(loaded):
    im = two_tone()
    a = server.embed_image(im)
    b = server.embed_image(im.transpose(Image.FLIP_LEFT_RIGHT))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_embed_image_크기와_색_모드가_달라도_224로_맞춘다(loaded):
    seen = {}

    def spy(batch):
        seen["shape"] = batch.numpy().shape if hasattr(batch, "numpy") else batch.a.shape
        return fake_vis(batch)

    server._model = (spy, server._model[1], server._model[2])
    server.embed_image(Image.new("L", (300, 120), 128))
    assert seen["shape"] == (2, 3, server.SIZE, server.SIZE)
