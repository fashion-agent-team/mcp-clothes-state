"""C2: OWLv2 base (google/owlv2-base-patch16-ensemble) + LoRA for stain/torn on the Jetson.
Trainable: LoRA r8/alpha16 on vision encoder q_proj/v_proj, class head, box head, final layer norm, and the two class
query embeddings (initialised from the mean text embedding of the zero-shot 'base' prompts).
Loss: DETR-style Hungarian matching per photo (cost 2*cls + 5*L1 + 2*(1-GIoU)), sigmoid focal loss on every patch,
L1 + GIoU on matched patches. Input: pad to square (0.5 grey, bottom/right like Owlv2Processor), resize to SIZE.
batch 1 + grad accumulation 8, fp16 autocast, gradient checkpointing.
Scoring writes the same json as owl-eval/zeroshot_eval.py (summary/test/clean/real) so photo_level.py works on it.
usage: python c2_owlv2_lora.py time <out dir> [size] [n photos]   # time + memory on n photos, 1 epoch
       python c2_owlv2_lora.py train <out dir> [size] [epochs]
       python c2_owlv2_lora.py eval <out dir> [size]              # score <out dir>/trainable.pt (or zero-shot if absent)"""
import json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

R = Path.home() / "clothes-state-train"; E = R / "exp3"; M = E / "owlv2-base"; OE = E / "owl-eval"
MODE, OUT = sys.argv[1], Path(sys.argv[2]); SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 768
ARG4 = int(sys.argv[4]) if len(sys.argv) > 4 else None
OUT.mkdir(parents=True, exist_ok=True); (E / "train.pid").write_text(str(os.getpid()))
dev = "cuda"; torch.manual_seed(0); random.seed(0)
logf = open(OUT / "log.txt", "a")
def say(*a):
    s = time.strftime("%F %T ") + " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

from transformers import Owlv2ForObjectDetection, Owlv2Processor
proc = Owlv2Processor.from_pretrained(M); model = Owlv2ForObjectDetection.from_pretrained(M).to(dev)
PROMPTS = {0: ["a stain on clothing", "a dirty spot on fabric", "a food stain on a shirt"],
           1: ["a hole in clothing", "a tear in fabric", "ripped fabric"]}
with torch.no_grad():
    q = []
    for c in (0, 1):
        t = proc(text=PROMPTS[c], return_tensors="pt").to(dev)
        e = model.owlv2.get_text_features(**t); e = getattr(e, "pooler_output", e)
        q.append(F.normalize(e, dim=-1).mean(0))
query = nn.Parameter(torch.stack(q))  # [2, 512]
model.owlv2.text_model = None; torch.cuda.empty_cache()  # text tower no longer needed
MEAN = torch.tensor(proc.image_processor.image_mean).view(3, 1, 1); STD = torch.tensor(proc.image_processor.image_std).view(3, 1, 1)

def prep(path, flip=False):
    """pixel tensor [3,S,S] and the scale to map square-normalised coords back to the photo"""
    im = Image.open(path).convert("RGB"); w, h = im.size; s = max(w, h)
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    sq = Image.new("RGB", (s, s), (128, 128, 128)); sq.paste(im, (0, 0))
    x = torch.from_numpy(np.asarray(sq.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255).permute(2, 0, 1)
    return (x - MEAN) / STD, w, h, s

def labels(lab, w, h, s, flip=False):
    """YOLO label rows (boxes or polygons) -> [(cls, cx, cy, bw, bh) in padded-square coords]"""
    out = []
    for l in (lab.read_text().splitlines() if lab.exists() else []):
        r = list(map(float, l.split()))
        if not r: continue
        if len(r) == 5: c, x, y, bw, bh = r
        else:
            xs, ys = r[1::2], r[2::2]; c = r[0]; x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if flip: x = 1 - x
        out.append((int(c), x * w / s, y * h / s, bw * w / s, bh * h / s))
    return out

def forward(px):
    feat_map, _ = model.image_embedder(pixel_values=px, interpolate_pos_encoding=True)
    b, gh, gw, d = feat_map.shape; feats = feat_map.reshape(b, gh * gw, d)
    logits, _ = model.class_predictor(feats, query[None].expand(b, -1, -1).to(feats.dtype))
    boxes = model.box_predictor(feats, feat_map, interpolate_pos_encoding=True)
    return logits.float(), boxes.float()  # [b, P, 2], [b, P, 4] cxcywh in the square

def cxcywh2xyxy(b): return torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)
def giou(a, b):  # a [n,4], b [m,4] xyxy -> [n,m]
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(0) * (x[:, 3] - x[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:]); inter = (rb - lt).clamp(0).prod(-1)
    union = area(a)[:, None] + area(b)[None] - inter; iou = inter / union.clamp(1e-9)
    lt2 = torch.min(a[:, None, :2], b[None, :, :2]); rb2 = torch.max(a[:, None, 2:], b[None, :, 2:]); enc = (rb2 - lt2).clamp(0).prod(-1)
    return iou - (enc - union) / enc.clamp(1e-9)

def loss_fn(logits, boxes, gt):
    logits, boxes = logits[0], boxes[0]; tgt = torch.zeros_like(logits)
    n = max(1, len(gt)); l_box = l_giou = torch.zeros((), device=dev)
    if gt:
        g = torch.tensor([r[1:] for r in gt], device=dev); gc = torch.tensor([r[0] for r in gt], device=dev)
        with torch.no_grad():
            p = logits.sigmoid()[:, gc]  # [P, n]
            cost = -2 * p + 5 * torch.cdist(boxes, g, p=1) - 2 * giou(cxcywh2xyxy(boxes), cxcywh2xyxy(g))
            ri, ci = linear_sum_assignment(cost.cpu().numpy())
        ri, ci = torch.as_tensor(ri, device=dev), torch.as_tensor(ci, device=dev)
        tgt[ri, gc[ci]] = 1
        l_box = F.l1_loss(boxes[ri], g[ci], reduction="sum") / n
        l_giou = (1 - torch.diag(giou(cxcywh2xyxy(boxes[ri]), cxcywh2xyxy(g[ci])))).sum() / n
    pr = logits.sigmoid(); ce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    pt = pr * tgt + (1 - pr) * (1 - tgt); focal = (ce * (1 - pt) ** 2 * (0.25 * tgt + 0.75 * (1 - tgt))).sum() / n
    return 2 * focal + 5 * l_box + 2 * l_giou

TRAINABLE = ["class_head", "box_head", "layer_norm"]
def setup_train():
    from peft import LoraConfig, inject_adapter_in_model
    inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                                       target_modules=r".*vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)"), model)
    for n_, p in model.named_parameters():
        p.requires_grad = "lora_" in n_ or any(n_.startswith(t) for t in TRAINABLE)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    say("trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad) + query.numel())

def save_trainable(path):
    torch.save({"model": {n_: p.detach().cpu() for n_, p in model.named_parameters() if p.requires_grad}, "query": query.detach().cpu()}, path)

def train_list(limit=None):
    items = []
    for d in (R / "dataset_v2/train", R / "dataset_v5/neg"):
        for p in sorted((d / "images").iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"): items.append((p, d / "labels" / (p.stem + ".txt")))
    random.Random(0).shuffle(items)
    return items[:limit] if limit else items

def run_train(items, epochs, ckpt):
    setup_train(); ACC = 8
    groups = [{"params": [p for n_, p in model.named_parameters() if p.requires_grad and "lora_" in n_] + [query], "lr": 1e-4},
              {"params": [p for n_, p in model.named_parameters() if p.requires_grad and "lora_" not in n_], "lr": 5e-5}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01); steps = math.ceil(len(items) * epochs / ACC)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(); step = 0; t_all = time.time(); torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        model.train(); te = time.time(); tot = 0; random.Random(ep).shuffle(items)
        for i, (p, lab) in enumerate(items):
            flip = random.random() < 0.5; px, w, h, s = prep(p, flip); gt = labels(lab, w, h, s, flip)
            with torch.autocast("cuda", torch.float16):
                lg, bx = forward(px[None].to(dev))
            loss = loss_fn(lg, bx, gt) / ACC; scaler.scale(loss).backward(); tot += float(loss) * ACC
            if (i + 1) % ACC == 0 or i + 1 == len(items):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            if i % 100 == 0 or i + 1 == len(items):
                el = time.time() - te
                say(f"ep {ep+1}/{epochs} it {i+1}/{len(items)} loss {tot/(i+1):.4f} {el:.0f}s ({el/(i+1):.2f}s/photo) "
                    f"peak {torch.cuda.max_memory_allocated()/2**20:.0f}MB eta_epoch {el/(i+1)*len(items)/60:.0f}min")
        save_trainable(ckpt); say(f"epoch {ep+1} done {time.time()-te:.0f}s")
    return time.time() - t_all

@torch.no_grad()
def detect(path):
    px, w, h, s = prep(path)
    with torch.autocast("cuda", torch.float16):
        lg, bx = forward(px[None].to(dev))
    sc = lg[0].sigmoid().cpu(); b = cxcywh2xyxy(bx[0]).cpu() * s / torch.tensor([w, h, w, h])
    dets = []
    for c in (0, 1):
        idx = torch.where(sc[:, c] >= 0.05)[0]; idx = idx[sc[idx, c].argsort(descending=True)[:300]]  # cap for python NMS
        for i in idx.tolist():
            dets.append((c, float(sc[i, c]), *b[i].clamp(0, 1).tolist()))
    return ns["nms"](dets)

code = (OE / "zeroshot_eval.py").read_text(); ns = {"np": np}
exec(code[code.index("def iou("):code.index("res, times = {}, []")], ns)

def run_eval(tag):
    model.eval(); res, times, gts = {}, [], {}
    for p in sorted((OE / "data/v2test/images").iterdir()):
        lab = OE / "data/v2test/labels" / (p.stem + ".txt"); g = []
        for l in (lab.read_text().splitlines() if lab.exists() else []):
            c, x, y, bw, bh = map(float, l.split()); g.append((int(c), x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2))
        gts[p.name] = g; torch.cuda.synchronize(); t = time.time(); res[p.name] = detect(p); torch.cuda.synchronize(); times.append(time.time() - t)
    clean = {p.name: detect(p) for p in sorted((OE / "data/clean").iterdir())}
    real = {p.name: detect(p) for p in sorted((OE / "data/realtest").iterdir())}
    sm = {"model": f"owlv2_lora_{tag}", "size": SIZE, "device": "jetson cuda fp16", "sec_per_photo": round(float(np.median(times)), 3),
          "ap50_stain": round(ns["ap50"](res, gts, 0), 3), "ap50_tear": round(ns["ap50"](res, gts, 1), 3),
          "clean_imgs_ge_0.3": sum(any(d[1] >= 0.3 for d in ds) for ds in clean.values()),
          "clean_imgs_ge_0.5": sum(any(d[1] >= 0.5 for d in ds) for ds in clean.values()),
          "real_imgs_ge_0.3": sum(any(d[1] >= 0.3 for d in ds) for ds in real.values()),
          "real_boxes_ge_0.3": sum(d[1] >= 0.3 for ds in real.values() for d in ds)}
    sm["map50"] = round((sm["ap50_stain"] + sm["ap50_tear"]) / 2, 3)
    (OUT / f"results_{tag}.json").write_text(json.dumps({"summary": sm, "test": res, "clean": clean, "real": real}))
    say("eval", json.dumps(sm))

if MODE == "time":
    sec = run_train(train_list(ARG4 or 50), 1, OUT / "trainable_time.pt")
    n = ARG4 or 50; per = sec / n; full = len(train_list())
    say(json.dumps({"photos": n, "sec": round(sec), "sec_per_photo": round(per, 2), "train_photos": full,
                    "est_epoch_h": round(per * full / 3600, 2), "peak_cuda_MB": round(torch.cuda.max_memory_allocated() / 2**20)}))
elif MODE == "train":
    EP = ARG4 or 3; sec = run_train(train_list(), EP, OUT / "trainable.pt"); say("train seconds", round(sec)); run_eval("trained")
elif MODE == "eval":
    ck = OUT / "trainable.pt"
    if ck.exists():
        setup_train(); st = torch.load(ck); model.load_state_dict(st["model"], strict=False); query.data = st["query"].to(dev); tag = "trained"
    else: tag = "zeroshot_meanquery"
    run_eval(tag)
(E / "train.pid").unlink(missing_ok=True)
