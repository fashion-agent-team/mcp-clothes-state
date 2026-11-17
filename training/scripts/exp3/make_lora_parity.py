"""Expected answers for the Java LoRA parity test: whole photo (box 0,0,1,1, margin 0) -> torchvision
Resize(224, BICUBIC) + CenterCrop(224) + ToTensor (PIL, like training) -> ONNX. Writes <out>/expected.json + copies photos.
usage: python make_lora_parity.py <onnx> <photo dir (class folders)> <out dir> [n]"""
import json, shutil, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
from torchvision import transforms as T
M, SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 60
classes = json.loads(M.with_suffix(".json").read_text())["classes"]
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()])
s = ort.InferenceSession(str(M), providers=["CPUExecutionProvider"])
ps = sorted(SRC.glob("*/*.jpg")) + sorted(SRC.glob("*/*.png")); pick = [ps[i] for i in np.random.default_rng(1).choice(len(ps), N, replace=False)]
out = []
for p in pick:
    pr = s.run(None, {"images": tf(Image.open(p).convert("RGB"))[None].numpy()})[0][0]
    shutil.copy(p, OUT / p.name); out.append({"file": p.name, "top": classes[int(pr.argmax())], "probs": [float(v) for v in pr]})
(OUT / "expected.json").write_text(json.dumps(out)); print("wrote", len(out))
