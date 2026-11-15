"""C1: FashionSigLIP (Marqo, open_clip) vision encoder + 8-class pattern head on the Jetson.
mode lora : LoRA r8/alpha16/dropout0.1 on every block's attn.qkv and attn.proj + linear head, 5 epochs
mode probe: frozen encoder, features extracted once, linear head only, 3 epochs
Plain cross-entropy with no class weights, like pattern_B (Ultralytics classify default).
Scoring = pattern_eval.py + pattern_threshold.py: test_fp/test_sh acc, macro recall, pattern-only macro recall,
plain-vs-pattern acc, per-class recall, ms/img at batch 64, threshold table.
usage: python c1_siglip_lora.py <lora|probe> <out dir> [batch]"""
import json, math, os, sys, time
from pathlib import Path
import torch, torch.nn as nn, open_clip
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

R = Path.home() / "clothes-state-train"; D = R / "pattern/ds"; W = R / "exp3/fashion-siglip"
MODE, OUT = sys.argv[1], Path(sys.argv[2]); B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
OUT.mkdir(parents=True, exist_ok=True); (R / "exp3/train.pid").write_text(str(os.getpid()))
dev = "cuda"; torch.manual_seed(0)
log = open(OUT / "log.txt", "a")
def say(*a):
    s = time.strftime("%F %T ") + " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

model = open_clip.create_model("local-dir:" + str(W))  # config + open_clip_model.safetensors from the folder
vis = model.visual.to(dev); del model; torch.cuda.empty_cache()
norm = T.Normalize([0.5] * 3, [0.5] * 3)
tf_eval = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), norm])
tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), norm])
train_ds = datasets.ImageFolder(D / "train", tf_train if MODE == "lora" else tf_eval)
classes = train_ds.classes; C = len(classes); plain = classes.index("plain")
def split_ds(name):  # test splits may miss some class folders: map folder names onto train class ids
    ds = datasets.ImageFolder(D / name, tf_eval)
    ds.samples = [(p, classes.index(ds.classes[y])) for p, y in ds.samples]; ds.targets = [y for _, y in ds.samples]
    return ds
val_ds, fp_ds, sh_ds = split_ds("val"), split_ds("test_fp"), split_ds("test_sh")
say("mode", MODE, "batch", B, "classes", classes, "train", len(train_ds), "val", len(val_ds), "fp", len(fp_ds), "sh", len(sh_ds))
head = nn.Linear(768, C).to(dev)

def predict_feats(ds, bs=64):
    """encoder features (no grad, fp16 autocast) + ms/img"""
    vis.eval(); fs, ys = [], []; dl = DataLoader(ds, bs, num_workers=2); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        for x, y in dl:
            fs.append(vis(x.to(dev, non_blocking=True)).float().cpu()); ys.append(y)
    torch.cuda.synchronize()
    return torch.cat(fs), torch.cat(ys), (time.time() - t0) * 1000 / len(ds)

def score(probs, ys, ms):
    names = classes; top = probs.argmax(1); n = len(ys)
    per = {}
    for c in range(C):
        m = ys == c
        if m.sum(): per[names[c]] = {"n": int(m.sum()), "recall": round(float((top[m] == c).float().mean()), 3),
                                     "precision": round(float(((top == c) & m).sum() / max(1, (top == c).sum())), 3)}
    pat = [c for c in per if c != "plain"]
    res = {"n": n, "acc": round(float((top == ys).float().mean()), 3),
           "macro_recall": round(sum(per[c]["recall"] for c in per) / len(per), 3),
           "pattern_macro_recall": round(sum(per[c]["recall"] for c in pat) / len(pat), 3),
           "plain_vs_pattern_acc": round(float(((top == plain) == (ys == plain)).float().mean()), 3),
           "ms_per_img": round(ms, 1), "per_class": per, "thr": {}}
    p_top = probs.max(1).values
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):  # same rule as pattern_threshold.py
        said = torch.where(top == plain, plain, torch.where(p_top >= t, top, -1))
        pl, pa = ys == plain, ys != plain
        res["thr"][str(t)] = {"plain_to_pattern": round(float(((said != plain) & (said != -1))[pl].float().mean()) * 100, 1),
                              "pattern_right": round(float((said == ys)[pa].float().mean()) * 100, 1),
                              "pattern_wrong_kind": round(float(((said != plain) & (said != -1) & (said != ys))[pa].float().mean()) * 100, 1),
                              "pattern_to_plain": round(float((said == plain)[pa].float().mean()) * 100, 1),
                              "unsure": round(float((said == -1).float().mean()) * 100, 1)}
    return res

def full_eval(tag):
    out = {}
    for name, ds in (("test_fp", fp_ds), ("test_sh", sh_ds)):
        f, y, ms = predict_feats(ds)
        with torch.no_grad(): probs = head(f.to(dev)).softmax(1).cpu()
        out[name] = score(probs, y, ms)
        say(tag, name, json.dumps({k: v for k, v in out[name].items() if k not in ("per_class", "thr")}),
            "animal", out[name]["per_class"].get("animal", {}).get("recall"), "t0.7", out[name]["thr"]["0.7"])
    return out

t_start = time.time()
if MODE == "probe":
    for p in vis.parameters(): p.requires_grad = False
    ftr, ytr, _ = predict_feats(train_ds); fva, yva, _ = predict_feats(val_ds)
    say("features done", round(time.time() - t_start), "s")
    opt = torch.optim.AdamW(head.parameters(), 1e-3); best = (-1, None)
    for ep in range(3):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]; loss = nn.functional.cross_entropy(head(ftr[j].to(dev)), ytr[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): acc = float((head(fva.to(dev)).argmax(1).cpu() == yva).float().mean())
        say("probe epoch", ep + 1, "val acc", round(acc, 4))
        if acc > best[0]: best = (acc, {k: v.clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
else:
    from peft import LoraConfig, get_peft_model
    vis.set_grad_checkpointing(True) if hasattr(vis, "set_grad_checkpointing") else None
    vis = get_peft_model(vis, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=r".*blocks\.\d+\.attn\.(qkv|proj)"))
    vis.print_trainable_parameters()
    opt = torch.optim.AdamW([{"params": [p for p in vis.parameters() if p.requires_grad], "lr": 1e-4},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    EP = 5; dl = DataLoader(train_ds, B, shuffle=True, num_workers=2, drop_last=True, pin_memory=False)
    steps = EP * len(dl); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(); best = (-1, None)
    for ep in range(EP):
        vis.train(); te = time.time(); tot = 0
        for i, (x, y) in enumerate(dl):
            with torch.autocast("cuda", torch.float16):
                loss = nn.functional.cross_entropy(head(vis(x.to(dev))).float(), y.to(dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += float(loss)
            if i % 200 == 0: say(f"ep {ep+1} it {i}/{len(dl)} loss {tot/(i+1):.4f} {time.time()-te:.0f}s mem {torch.cuda.max_memory_allocated()/2**20:.0f}MB")
        fva, yva, _ = predict_feats(val_ds)
        with torch.no_grad(): acc = float((head(fva.to(dev)).argmax(1).cpu() == yva).float().mean())
        say("lora epoch", ep + 1, "val acc", round(acc, 4), "epoch s", round(time.time() - te))
        if acc > best[0]:
            best = (acc, ep + 1); vis.save_pretrained(OUT / "lora_best"); torch.save(head.state_dict(), OUT / "head_best.pt")
    from peft import PeftModel
    vis.load_adapter(OUT / "lora_best", adapter_name="best"); vis.set_adapter("best")
    head.load_state_dict(torch.load(OUT / "head_best.pt"))
    say("best epoch", best[1], "val acc", round(best[0], 4))
train_s = round(time.time() - t_start)
res = {"mode": MODE, "batch": B, "train_seconds": train_s, "test": full_eval(MODE)}
if MODE == "probe": torch.save(head.state_dict(), OUT / "head_probe.pt")
(OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
(R / "exp3/train.pid").unlink(missing_ok=True); say("done", train_s, "s")
