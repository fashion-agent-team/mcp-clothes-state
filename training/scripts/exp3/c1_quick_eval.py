"""Quick test of a saved SigLIP LoRA checkpoint (lora dir + head .pt) on test_fp / test_sh while training runs elsewhere.
Small batch, no workers, does not touch exp3/train.pid. usage: python c1_quick_eval.py <lora dir> <head .pt> [batch]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, open_clip
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from peft import PeftModel
sys.argv += [] ; R = Path.home() / "clothes-state-train"; D = R / "pattern/ds"; W = R / "exp3/fashion-siglip"; B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
sys.path.insert(0, str(R / "exp3")); from c1_ensemble import score
classes = sorted(p.name for p in (D / "train").iterdir())
m = PeftModel.from_pretrained(open_clip.create_model("local-dir:" + str(W)).visual, sys.argv[1]).cuda().eval()
head = nn.Linear(768, 8).cuda(); head.load_state_dict(torch.load(sys.argv[2]))
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
for s in ("val", "test_fp", "test_sh"):
    ds = datasets.ImageFolder(D / s, tf); ys = np.array([classes.index(ds.classes[y]) for _, y in ds.samples]); ps = []; t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        for x, _ in DataLoader(ds, B, num_workers=0): ps.append(head(m(x.cuda()).float()).softmax(1).cpu())
    r = score(torch.cat(ps).numpy(), ys, classes)
    print(s, "acc", r["acc"], "pat_macro", r["pattern_macro_recall"], "animal", r["animal"], "t0.7", r["t0.7"], f"{time.time()-t0:.0f}s", flush=True)
