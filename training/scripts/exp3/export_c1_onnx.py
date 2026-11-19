"""Merge the SigLIP LoRA into the encoder and export encoder + 8-class head + softmax to ONNX.
Input: [N, 3, 224, 224] RGB in 0..1 (same tensor the server already makes for pattern8 B: PatternInput, x/255);
the (x - 0.5) / 0.5 normalisation is inside the graph. Output: [N, 8] probabilities, classes in folder order.
Then checks: ONNX vs PyTorch on test_fp photos (max prob diff, top1 agreement), and the full test score when crops are
resized like the Java server (PIL BILINEAR) instead of BICUBIC used in training.
usage: python export_c1_onnx.py <lora dir with lora_best/ + head_best.pt> <out .onnx> [n parity photos]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, open_clip
from PIL import Image
from peft import PeftModel
from torchvision import transforms as T

R = Path.home() / "clothes-state-train"; D = R / "pattern/ds"; W = R / "exp3/fashion-siglip"
LD, OUT = Path(sys.argv[1]), Path(sys.argv[2]); NP = int(sys.argv[3]) if len(sys.argv) > 3 else 300
classes = sorted(p.name for p in (D / "train").iterdir())
vis = PeftModel.from_pretrained(open_clip.create_model("local-dir:" + str(W)).visual, str(LD / "lora_best")).merge_and_unload().eval()
head = nn.Linear(768, 8); head.load_state_dict(torch.load(LD / "head_best.pt", map_location="cpu"))

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.vis, self.head = vis, head
    def forward(self, x):
        return self.head(self.vis((x - 0.5) / 0.5)).softmax(-1)
net = Net().eval()
torch.onnx.export(net, torch.rand(1, 3, 224, 224), str(OUT), input_names=["images"], output_names=["probs"],
                  dynamic_axes={"images": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
print("exported", OUT, round(OUT.stat().st_size / 2**20, 1), "MB", flush=True)
(OUT.with_suffix(".json")).write_text(json.dumps({"classes": classes, "input": "RGB 0..1, 224x224", "lora": str(LD)}))

import onnxruntime as ort
sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
def tf(interp): return T.Compose([T.Resize(224, interpolation=interp), T.CenterCrop(224), T.ToTensor()])
bic, bil = tf(T.InterpolationMode.BICUBIC), tf(T.InterpolationMode.BILINEAR)
fp = sorted((D / "test_fp").glob("*/*.jpg")); rng = np.random.default_rng(0); pick = [fp[i] for i in rng.choice(len(fp), NP, replace=False)]
x = torch.stack([bic(Image.open(p).convert("RGB")) for p in pick])
with torch.no_grad(): pt = torch.cat([net(x[i:i + 32]) for i in range(0, NP, 32)]).numpy()
t0 = time.time(); ox = np.concatenate([sess.run(None, {"images": x[i:i + 1].numpy()})[0] for i in range(NP)]); cpu_ms = (time.time() - t0) * 1000 / NP
print(f"parity {NP} photos: max |p_onnx - p_torch| {np.abs(ox - pt).max():.2e}, top1 agree {(ox.argmax(1) == pt.argmax(1)).mean():.4f}, "
      f"ORT CPU batch1 {cpu_ms:.0f} ms/photo", flush=True)
# full tests with server-style bilinear resize, PyTorch on GPU
net = net.cuda()
for s in ("test_fp", "test_sh"):
    ps = sorted((D / s).glob("*/*.jpg")); ys = np.array([classes.index(p.parent.name) for p in ps]); out = {}
    for name, t in (("bicubic", bic), ("bilinear", bil)):
        pr = []
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            for i in range(0, len(ps), 32):
                pr.append(net(torch.stack([t(Image.open(p).convert("RGB")) for p in ps[i:i + 32]]).cuda()).float().cpu())
        pr = torch.cat(pr).numpy(); out[name] = round(float((pr.argmax(1) == ys).mean()), 4)
        np.save(OUT.parent / f"{OUT.stem}_{s}_{name}.npy", pr)
    print(s, "top1 acc", out, flush=True)
