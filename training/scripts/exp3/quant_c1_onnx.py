"""Dynamic INT8 (weights) version of the pattern ONNX; compare speed (ORT CPU, batch 1) and full-test top1 vs fp32.
usage: python quant_c1_onnx.py <fp32 .onnx> <out int8 .onnx>"""
import sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image
from torchvision import transforms as T
F32, OUT = Path(sys.argv[1]), Path(sys.argv[2]); D = Path.home() / "clothes-state-train/pattern/ds"
quantize_dynamic(str(F32), str(OUT), weight_type=QuantType.QInt8)
print("int8", round(OUT.stat().st_size / 2**20, 1), "MB", flush=True)
classes = sorted(p.name for p in (D / "train").iterdir()); tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()])
o = ort.SessionOptions(); o.intra_op_num_threads = 4
for name, f in (("fp32", F32), ("int8", OUT)):
    s = ort.InferenceSession(str(f), o, providers=["CPUExecutionProvider"]); x = tf(Image.open(sorted((D / "test_fp").glob("*/*.jpg"))[0]).convert("RGB"))[None].numpy()
    s.run(None, {"images": x}); t0 = time.time(); [s.run(None, {"images": x}) for _ in range(20)]; print(name, "ORT CPU 4 threads", round((time.time() - t0) * 50), "ms/photo", flush=True)
s = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
for sp in ("test_fp", "test_sh"):
    ps = sorted((D / sp).glob("*/*.jpg")); ys = np.array([classes.index(p.parent.name) for p in ps]); pr = []
    for i in range(0, len(ps), 16):
        pr.append(s.run(None, {"images": np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in ps[i:i + 16]])})[0])
    pr = np.concatenate(pr); np.save(OUT.parent / f"{OUT.stem}_{sp}_bicubic.npy", pr); print(sp, "int8 top1", round(float((pr.argmax(1) == ys).mean()), 4), flush=True)
