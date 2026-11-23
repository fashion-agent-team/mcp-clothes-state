"""테스트용 설정. 젯슨용 torch(CUDA)가 없는 곳(CI, 맥)에서도 돌도록, torch가 없으면 아주 작은 가짜 torch를 끼웁니다.

app.py가 torch에서 쓰는 건 no_grad(), from_numpy(...).cuda().half(), 결과의 .float().cpu().numpy()뿐이라
numpy 배열을 감싼 가짜로 충분합니다. 진짜 모델(SigLIP2)은 젯슨에서만 올리고 여기서는 가짜 vis를 씁니다.
"""
import contextlib
import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def cuda(self):
        return self

    def half(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


try:
    import torch  # noqa: F401
except ImportError:
    fake = types.ModuleType("torch")
    fake.no_grad = contextlib.nullcontext
    fake.from_numpy = FakeTensor
    fake.cuda = types.SimpleNamespace(memory_allocated=lambda: 0)
    sys.modules["torch"] = fake
