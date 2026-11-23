"""모델 올리기(load·startup) 테스트. 젯슨의 open_clip·safetensors·GPU 없이, 같은 모양의 가짜 모듈로 순서와 규칙만 봅니다.

- 가중치 파일에서 이미지 탑(visual.*)만 골라 앞의 "visual."을 떼고 넣는지
- 빠진 가중치가 있으면 시작을 멈추는지(엉뚱한 벡터를 내지 않게)
- 전처리 평균·표준편차가 없으면 SigLIP 기본값(0.5)을 쓰는지
- 가중치 경로: SO400M_WEIGHTS가 있으면 그 파일, 없으면 HF 캐시(내려받지 않음)
"""
import contextlib
import sys
import types

import numpy as np
import pytest

import app as server
import torch


class FakeWeight:
    def __init__(self, name):
        self.name = name
        self.steps = []

    def half(self):
        self.steps.append("half")
        return self

    def cuda(self):
        self.steps.append("cuda")
        return self


class FakeVision:
    def __init__(self, missing=(), preprocess=None):
        self.missing = list(missing)
        self.loaded = None
        self.calls = []
        if preprocess is not None:
            self.preprocess_cfg = preprocess

    def load_state_dict(self, sd, strict, assign):
        self.loaded = sd
        self.calls.append(("load", strict, assign))
        return types.SimpleNamespace(missing_keys=self.missing)

    def eval(self):
        self.calls.append("eval")
        return self

    def cuda(self):
        self.calls.append("cuda")
        return self


@pytest.fixture
def fake_libs(monkeypatch, tmp_path):
    """open_clip, open_clip.model, safetensors를 가짜로 바꾼다. 만든 vis와 읽은 키를 돌려준다."""
    state = {"vis": FakeVision(), "opened": None, "built": None}
    weights = {name: FakeWeight(name) for name in ["visual.trunk.w", "visual.head.b", "text.token.w"]}
    state["weights"] = weights

    open_clip = types.ModuleType("open_clip")
    open_clip.get_model_config = lambda arch: {"embed_dim": 1152, "vision_cfg": {"arch": arch}}
    model_mod = types.ModuleType("open_clip.model")

    def build(embed_dim, vision_cfg):
        state["built"] = (embed_dim, vision_cfg)
        return state["vis"]

    model_mod._build_vision_tower = build
    open_clip.model = model_mod

    safetensors = types.ModuleType("safetensors")

    @contextlib.contextmanager
    def safe_open(path, framework):
        state["opened"] = (path, framework)
        yield types.SimpleNamespace(keys=lambda: list(weights), get_tensor=lambda k: weights[k])

    safetensors.safe_open = safe_open

    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "open_clip.model", model_mod)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    if not hasattr(torch, "device"):
        monkeypatch.setattr(torch, "device", lambda _name: contextlib.nullcontext(), raising=False)
    monkeypatch.setattr(server, "WEIGHTS", str(tmp_path / "w.safetensors"))
    return state


def test_load_only_visual_weights_in_fp16_on_gpu(fake_libs):
    vis, mean, std = server.load()

    assert vis is fake_libs["vis"]
    assert fake_libs["built"] == (1152, {"arch": server.ARCH})
    assert fake_libs["opened"][1] == "pt"
    assert sorted(vis.loaded) == ["head.b", "trunk.w"]
    assert fake_libs["weights"]["visual.trunk.w"].steps == ["half", "cuda"]
    assert fake_libs["weights"]["text.token.w"].steps == []
    assert vis.calls == [("load", False, True), "eval", "cuda"]
    # 전처리 설정이 없으면 SigLIP 기본값 0.5
    np.testing.assert_allclose(mean, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(std, [0.5, 0.5, 0.5])


def test_load_uses_model_preprocess_values(fake_libs):
    fake_libs["vis"] = FakeVision(preprocess={"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]})
    _, mean, std = server.load()
    np.testing.assert_allclose(mean, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(std, [0.4, 0.5, 0.6])
    assert mean.dtype == np.float32


def test_load_stops_when_weights_are_missing(fake_libs):
    fake_libs["vis"] = FakeVision(missing=["trunk.blocks.0.w", "trunk.blocks.1.w"])
    with pytest.raises(RuntimeError, match="weights missing"):
        server.load()


def test_weights_path_prefers_env(monkeypatch):
    monkeypatch.setattr(server, "WEIGHTS", "/models/so400m.safetensors")
    assert server._weights_path() == "/models/so400m.safetensors"


def test_weights_path_reads_hf_cache_without_download(monkeypatch):
    asked = {}
    hub = types.ModuleType("huggingface_hub")

    def download(repo, filename, local_files_only):
        asked.update(repo=repo, filename=filename, local_files_only=local_files_only)
        return "/cache/model.safetensors"

    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(server, "WEIGHTS", "")

    assert server._weights_path() == "/cache/model.safetensors"
    assert asked == {
        "repo": "timm/" + server.ARCH,
        "filename": "open_clip_model.safetensors",
        "local_files_only": True,
    }


def test_startup_loads_model_and_warms_up(monkeypatch):
    seen = []

    def vis(batch):
        seen.append(batch)
        return torch.from_numpy(np.ones((2, 4), np.float32))

    model = (vis, np.full(3, 0.5, np.float32), np.full(3, 0.5, np.float32))
    monkeypatch.setattr(server, "load", lambda: model)
    monkeypatch.setattr(server, "_model", None)
    if not hasattr(torch.cuda, "memory_allocated"):
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0, raising=False)

    server.startup()

    assert server._model is model
    assert len(seen) == 1, "회색 사진 한 장으로 미리 한 번 돌려 둔다"
