"""Step 1 (no training): class probabilities on val / test_fp / test_sh from SigLIP+LoRA, SigLIP linear probe and
pattern8 B, each plain and left-right flipped; then pick combinations on val and score them on the tests.
usage: python c1_ensemble.py <lora dir> <probe dir> <out dir>      (lora dir = exp3/c1_lora with lora_best/ + head_best.pt)
       python c1_ensemble.py --score <out dir>                       (only recombine saved probabilities)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch

R = Path.home() / "clothes-state-train"; D = R / "pattern/ds"; W = R / "exp3/fashion-siglip"
SPLITS = ("val", "test_fp", "test_sh")

def score(probs, ys, classes):
    """same rules as pattern_eval.py / pattern_threshold.py"""
    plain = classes.index("plain"); top = probs.argmax(1); C = len(classes); per = {}
    for c in range(C):
        m = ys == c
        if m.sum(): per[classes[c]] = round(float((top[m] == c).mean()), 3)
    pat = [c for c in per if c != "plain"]
    res = {"acc": round(float((top == ys).mean()), 4), "pattern_macro_recall": round(sum(per[c] for c in pat) / len(pat), 3),
           "plain_vs_pattern_acc": round(float(((top == plain) == (ys == plain)).mean()), 3), "animal": per.get("animal"), "per_class": per}
    pt = probs.max(1); said = np.where(top == plain, plain, np.where(pt >= 0.7, top, -1)); pl, pa = ys == plain, ys != plain
    res["t0.7"] = {"plain_to_pattern": round(100 * float(((said != plain) & (said != -1))[pl].mean()), 1),
                   "pattern_right": round(100 * float((said == ys)[pa].mean()), 1),
                   "pattern_wrong_kind": round(100 * float(((said != plain) & (said != -1) & (said != ys))[pa].mean()), 1),
                   "pattern_to_plain": round(100 * float((said == plain)[pa].mean()), 1),
                   "unsure": round(100 * float((said == -1).mean()), 1)}
    return res

def collect(lora_dir, probe_dir, out):
    import open_clip, torch.nn as nn
    from PIL import Image
    from torchvision import datasets, transforms as T
    from torch.utils.data import DataLoader
    from peft import PeftModel
    from ultralytics import YOLO
    (R / "exp3/train.pid").write_text(str(os.getpid())); dev = "cuda"
    classes = sorted(p.name for p in (D / "train").iterdir())
    vis = open_clip.create_model("local-dir:" + str(W)).visual
    IMG = int(os.environ.get("IMG", 224))
    if IMG != 224: vis.trunk.set_input_size(img_size=(IMG, IMG))
    lora = PeftModel.from_pretrained(vis, str(lora_dir / "lora_best")).to(dev).eval()
    head_l = nn.Linear(768, 8).to(dev); head_l.load_state_dict(torch.load(lora_dir / "head_best.pt"))
    head_p = nn.Linear(768, 8).to(dev); head_p.load_state_dict(torch.load(probe_dir / "head_probe.pt"))
    yb = YOLO(R / "pattern/runs/pattern_B/weights/best.pt"); bmap = [classes.index(yb.names[i]) for i in range(len(yb.names))]
    norm = T.Normalize([0.5] * 3, [0.5] * 3)
    tfs = {f: T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMG)]
                        + ([T.RandomHorizontalFlip(1.0)] if f else []) + [T.ToTensor(), norm]) for f in (0, 1)}
    for split in SPLITS:
        ds = datasets.ImageFolder(D / split); samples = [(p, classes.index(ds.classes[y])) for p, y in ds.samples]
        ys = np.array([y for _, y in samples]); np.save(out / f"{split}_y.npy", ys); t0 = time.time()
        for f in (0, 1):
            ds.transform = tfs[f]; ds.samples = [(p, y) for p, y in samples]
            pl = []
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                for x, _ in DataLoader(ds, 64 if IMG == 224 else 32, num_workers=2):
                    pl.append(head_l(lora(x.to(dev)).float()).softmax(1).cpu())
            np.save(out / f"{split}_lora_f{f}.npy", torch.cat(pl).numpy())
            # probe uses the same encoder without the adapter
            pp = []
            with torch.no_grad(), torch.autocast("cuda", torch.float16), lora.disable_adapter():
                for x, _ in DataLoader(ds, 64 if IMG == 224 else 32, num_workers=2):
                    pp.append(head_p(lora(x.to(dev)).float()).softmax(1).cpu())
            np.save(out / f"{split}_probe_f{f}.npy", torch.cat(pp).numpy())
            pb = []
            for i in range(0, len(samples), 64):
                ims = []
                for p, _ in samples[i:i + 64]:
                    a = np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1]  # BGR like cv2
                    ims.append(np.ascontiguousarray(a[:, ::-1] if f else a))
                for r in yb.predict(ims, imgsz=224, device=0, verbose=False):
                    v = r.probs.data.cpu().numpy(); o = np.zeros(8, np.float32); o[bmap] = v; pb.append(o)
            np.save(out / f"{split}_B_f{f}.npy", np.stack(pb))
        print(split, len(samples), "photos", round(time.time() - t0), "s", flush=True)
    (out / "classes.json").write_text(json.dumps(classes)); (R / "exp3/train.pid").unlink(missing_ok=True)

def combos(out):
    classes = json.loads((out / "classes.json").read_text()); L = lambda s, n: np.load(out / f"{s}_{n}.npy")
    def mix(s, cfg):
        m, flip, w = cfg
        get = (lambda n: (L(s, n + "_f0") + L(s, n + "_f1")) / 2) if flip else (lambda n: L(s, n + "_f0"))
        if m == "lora": return get("lora")
        other = get("B" if m == "lora+B" else "probe"); return w * get("lora") + (1 - w) * other
    cfgs = [("lora", 0, 1), ("lora", 1, 1)] + [(m, f, w) for m in ("lora+B", "lora+probe") for f in (0, 1) for w in (0.5, 0.6, 0.7, 0.8)]
    rows = []
    for c in cfgs:
        r = {"cfg": f"{c[0]}{' +flip' if c[1] else ''}{f' w={c[2]}' if c[0] != 'lora' else ''}", "val_acc": score(mix("val", c), L("val", "y"), classes)["acc"]}
        rows.append((r, c))
    rows.sort(key=lambda x: -x[0]["val_acc"])
    print("val ranking:"); [print(" ", r["cfg"], r["val_acc"]) for r, _ in rows]
    res = {"val_ranking": [r for r, _ in rows], "test": {}}
    picks = [rows[0][1], ("lora", 0, 1), ("lora", 1, 1)]
    best_lb = max((x for x in rows if x[1][0] == "lora+B"), key=lambda x: x[0]["val_acc"])[1]
    best_lp = max((x for x in rows if x[1][0] == "lora+probe"), key=lambda x: x[0]["val_acc"])[1]
    for c in dict.fromkeys(picks + [best_lb, best_lp]):
        name = rows[[x[1] for x in rows].index(c)][0]["cfg"]
        res["test"][name] = {s: score(mix(s, c), L(s, "y"), classes) for s in ("test_fp", "test_sh")}
        for s in ("test_fp", "test_sh"):
            t = res["test"][name][s]; print(name, s, "acc", t["acc"], "pat_macro", t["pattern_macro_recall"], "animal", t["animal"], "t0.7", t["t0.7"])
    res["val_best"] = rows[0][0]["cfg"]
    (out / "ensemble.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    if sys.argv[1] == "--score": combos(Path(sys.argv[2]))
    else:
        out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        collect(Path(sys.argv[1]), Path(sys.argv[2]), out); combos(out)
