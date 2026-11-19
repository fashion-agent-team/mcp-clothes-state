"""Real-photo test: photo-level found(>=0.5)/ambiguous(0.2-0.5)/missed + box recall (IoU>=0.5, same class).
usage: python realtest_eval.py <root> <run name> <out json> [device]"""
import json, sys, time
from pathlib import Path
from ultralytics import YOLO
R, RUN, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); DEV = sys.argv[4] if len(sys.argv) > 4 else "mps"
m = YOLO(R / f"runs/{RUN}/weights/best.pt")
SETS = {"aleminaitha32": R / "realtest/aleminaitha/test", "auto2_12": R / "dataset_v7/auto2"}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)
def gt_of(d, p):
    lab = d / "labels" / (p.stem + ".txt")
    rows = [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
    out = []
    for r in rows:  # box rows (c x y w h) or polygon rows (c x1 y1 x2 y2 ...) -> xyxy
        if len(r) == 5: c, x, y, w, h = r; out.append((int(c), [x-w/2, y-h/2, x+w/2, y+h/2]))
        else: xs, ys = r[1::2], r[2::2]; out.append((int(r[0]), [min(xs), min(ys), max(xs), max(ys)]))
    return out
def fin(b):
    b["recall"] = round(b["tp"] / max(1, b["gt"]), 3); b["precision"] = round(b["tp"] / max(1, b["tp"] + b["fp"]), 3)
res = {"run": RUN, "conf": 0.2, "imgsz": 512, "device": DEV, "sets": {}}
allp = {"defect": 0, "found": 0, "ambiguous": 0, "missed": 0, "clean": 0, "clean_flag05": 0, "clean_flag02": 0}
allb = {t: {"gt": 0, "tp": 0, "fp": 0} for t in ("0.2", "0.5")}
t0 = time.time(); n_img = 0
for name, d in SETS.items():
    imgs = sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")); n_img += len(imgs)
    pred = {}
    for i in range(0, len(imgs), 8):
        for r in m.predict([str(p) for p in imgs[i:i+8]], imgsz=512, conf=0.2, device=DEV, verbose=False):
            pred[Path(r.path).name] = [(int(b.cls), float(b.conf), b.xyxyn[0].tolist()) for b in r.boxes]
    ph = {k: 0 for k in allp}; bx = {t: {"gt": 0, "tp": 0, "fp": 0} for t in ("0.2", "0.5")}; per = []
    for p in imgs:
        g = gt_of(d, p); dets = pred[p.name]; mx = max((c for _, c, _ in dets), default=0.0)
        if g:
            ph["defect"] += 1; k = "found" if mx >= 0.5 else "ambiguous" if mx >= 0.2 else "missed"; ph[k] += 1
        else:
            ph["clean"] += 1; ph["clean_flag05"] += mx >= 0.5; ph["clean_flag02"] += mx >= 0.2; k = "clean"
        for t in bx:
            used = set(); bx[t]["gt"] += len(g)
            for c, cf, b in sorted([x for x in dets if x[1] >= float(t)], key=lambda x: -x[1]):
                best = max(((iou(b, gb), j) for j, (gc, gb) in enumerate(g) if gc == c and j not in used), default=(0, None))
                if best[0] >= 0.5: bx[t]["tp"] += 1; used.add(best[1])
                else: bx[t]["fp"] += 1
        per.append({"img": p.name, "gt": [c for c, _ in g], "max_conf": round(mx, 3), "result": k,
                    "dets": [(c, round(cf, 3)) for c, cf, _ in dets]})
    for t in bx: fin(bx[t])
    for k in allp: allp[k] += ph[k]
    for t in bx:
        for k in ("gt", "tp", "fp"): allb[t][k] += bx[t][k]
    res["sets"][name] = {"photo": ph, "box": bx, "per_image": per}
for t in allb: fin(allb[t])
res["total"] = {"photo": allp, "box": allb, "sec_per_img": round((time.time() - t0) / n_img, 3)}
OUT.write_text(json.dumps(res, indent=1))
for k, v in res["sets"].items(): print(k, v["photo"], v["box"])
print("total", res["total"])
