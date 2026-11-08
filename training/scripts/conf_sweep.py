"""Confidence-threshold sweep (two-model setup by default, or one model): per-class precision/recall
on the v2 test split (IoU >= 0.5 greedy matching) and the number of clean held-out photos with any box."""
import json, sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

R = Path(sys.argv[1]); out = Path(sys.argv[2])
# default: stain from v4_neg, tear from v2; optional 3rd arg = one run name used for both classes (e.g. yolo26n_v5)
if len(sys.argv) > 3:
    one = YOLO(R / f"runs/{sys.argv[3]}/weights/best.pt"); models = {0: one, 1: one}
else:
    models = {0: YOLO(R / "runs/yolo26n_v4_neg/weights/best.pt"), 1: YOLO(R / "runs/yolo26n_v2/weights/best.pt")}
test_imgs = sorted((R / "dataset_v2/test/images").iterdir())
clean_imgs = sorted((R / "negatives/heldout").iterdir())
real_imgs = sorted((R / "realtest/aleminaitha/test/images").iterdir())

def predict(paths):
    res = {p.name: [] for p in paths}
    for cls, m in models.items():
        for i in range(0, len(paths), 8):  # small chunks: a list is predicted as one batch, which can run out of memory
            for r in m.predict([str(p) for p in paths[i:i + 8]], imgsz=512, conf=0.05, device=0, verbose=False):
                res[Path(r.path).name] += [(cls, float(b.conf), b.xyxyn[0].tolist()) for b in r.boxes if int(b.cls) == cls]
    return res

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

gt = {}
for p in test_imgs:
    lab = R / "dataset_v2/test/labels" / (p.stem + ".txt")
    rows = [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
    gt[p.name] = [(int(c), [x - w / 2, y - h / 2, x + w / 2, y + h / 2]) for c, x, y, w, h in rows]
tp_pred, cl_pred, rl_pred = predict(test_imgs), predict(clean_imgs), predict(real_imgs)
rows = []
for t in np.round(np.arange(0.10, 0.701, 0.05), 2):
    row = {"conf": float(t)}
    for cls, name in ((0, "stain"), (1, "torn")):
        tp = fp = n_gt = 0
        for img, g in gt.items():
            gs = [b for c, b in g if c == cls]; n_gt += len(gs); used = set()
            for _, cf, b in sorted([d for d in tp_pred[img] if d[0] == cls and d[1] >= t], key=lambda d: -d[1]):
                best = max(((iou(b, gb), i) for i, gb in enumerate(gs) if i not in used), default=(0, None))
                if best[0] >= 0.5: tp += 1; used.add(best[1])
                else: fp += 1
        row[f"{name}_recall"] = round(tp / max(1, n_gt), 3)
        row[f"{name}_precision"] = round(tp / max(1, tp + fp), 3)
    row["clean_imgs_flagged"] = sum(1 for d in cl_pred.values() if any(cf >= t for _, cf, _ in d))
    rows.append(row)
# (tell, ask) pairs: how often a photo gets a confirmed report vs a friend question
def shares(pred, tell, ask):
    n = len(pred)
    told = sum(1 for d in pred.values() if any(cf >= tell for _, cf, _ in d))
    asked = sum(1 for d in pred.values() if any(ask <= cf < tell for _, cf, _ in d))
    return round(told / n, 3), round(asked / n, 3)
by_conf = {r["conf"]: r for r in rows}
pairs = []
for tell in (0.4, 0.5, 0.6, 0.7):
    for ask in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        if ask >= tell:
            continue
        rt, ra = shares(rl_pred, tell, ask); ct, ca = shares(cl_pred, tell, ask)
        lo = by_conf.get(round(ask, 2)) or by_conf[min(by_conf, key=lambda c: abs(c - ask))]
        hi = by_conf.get(round(tell, 2)) or by_conf[min(by_conf, key=lambda c: abs(c - tell))]
        pairs.append({"tell": tell, "ask": ask,
                      "real_told": rt, "real_asked": ra, "clean_told": ct, "clean_asked": ca,
                      "stain_recall_told": hi["stain_recall"], "stain_recall_told_or_asked": lo["stain_recall"],
                      "torn_recall_told": hi["torn_recall"], "torn_recall_told_or_asked": lo["torn_recall"]})
out.write_text(json.dumps({"by_conf": rows, "pairs": pairs}, indent=1))
for r in rows: print(r)
for r in pairs: print(r)
