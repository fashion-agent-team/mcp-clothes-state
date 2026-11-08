"""Synthetic stains on people wearing clothes: clean CLOTHES photos (not used as negatives, not held out), pose
keypoints give the torso (shoulders-hips), and feathered stain crops from labelled Clothes-stain
boxes are blended in there (darkening only, like make_synth.py), small relative to the shoulder width."""
import random, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

R = Path(sys.argv[1]); N_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 600
OUT = R / "synth_worn"; (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
used = {p.name for p in (R / "dataset_v5/neg/images").iterdir()}
bases = [p for p in sorted((R / "negatives/pool").iterdir()) if p.name not in used]
rng.shuffle(bases)

crops = []
SRC = R / "raw/stain/train"
for img in sorted((SRC / "images").iterdir()):
    lab = SRC / "labels" / f"{img.stem}.txt"
    rows = [list(map(float, l.split()[1:5])) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
    rows = [b for b in rows if 0.04 < b[2] < 0.6 and 0.04 < b[3] < 0.6]
    if rows: crops.append((img, rng.choice(rows)))
rng.shuffle(crops)

def ratio_patch(img_p, b):
    im = cv2.imread(str(img_p)).astype(np.float32); h, w = im.shape[:2]
    x, y, bw, bh = b
    c = im[max(0, int((y - bh / 2) * h)):int((y + bh / 2) * h), max(0, int((x - bw / 2) * w)):int((x + bw / 2) * w)]
    if c.shape[0] < 12 or c.shape[1] < 12: return None
    border = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    return np.clip(c / (np.median(border, axis=0) + 1), 0.15, 1.0)

def feather(hh, ww):
    yy, xx = np.mgrid[0:hh, 0:ww]
    d = ((xx - ww / 2) / (ww / 2)) ** 2 + ((yy - hh / 2) / (hh / 2)) ** 2
    return np.clip((1.15 - d) / 0.45, 0, 1)[..., None]

pose = YOLO(R / "weights/yolo11n-pose.pt")
ci = made = skipped = 0
for base in bases:
    if made >= N_MAX: break
    im0 = cv2.imread(str(base)); H, W = im0.shape[:2]
    r = pose.predict(im0, imgsz=640, conf=0.5, device=0, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0: continue
    i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.tolist()]))
    kp, kc = r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()
    if not all(kc[j] > 0.5 for j in (5, 6, 11, 12)): continue          # need a visible torso
    sw = np.linalg.norm(kp[5] - kp[6])
    if sw < 40: continue                                                # person too small in the photo
    torso = np.float32([kp[5], kp[6], kp[12], kp[11]])
    im, rows = im0.astype(np.float32), []
    for _ in range(rng.randint(1, 3) * 3):
        if len(rows) >= 3: break
        for _t in range(10):
            p, b = crops[ci % len(crops)]; ci += 1
            ratio = ratio_patch(p, b)
            if ratio is not None: break
        tw = max(8, int(sw * rng.uniform(0.06, 0.18)))
        th = max(8, int(tw * ratio.shape[0] / ratio.shape[1] * rng.uniform(0.8, 1.25)))
        rr = cv2.resize(ratio, (tw, th), interpolation=cv2.INTER_AREA)
        rr = cv2.warpAffine(rr, cv2.getRotationMatrix2D((tw / 2, th / 2), rng.uniform(0, 360), 1.0), (tw, th), borderValue=(1, 1, 1))
        a = feather(th, tw) * rng.uniform(0.8, 1.0)
        # torso only: arms and legs are often bare skin (short sleeves, skirts, shorts)
        u, v = rng.uniform(0.15, 0.85), rng.uniform(0.1, 0.9)
        top = torso[0] + (torso[1] - torso[0]) * u; bot = torso[3] + (torso[2] - torso[3]) * u
        cx, cy = top + (bot - top) * v
        px, py = int(cx - tw / 2), int(cy - th / 2)
        if px < 0 or py < 0 or px + tw > W or py + th > H: continue
        roi = im[py:py + th, px:px + tw]; new = roi * (1 - a + a * rr); core = a[..., 0] > 0.5
        if not core.any() or (roi[core] - new[core]).mean() < 18.0:     # keep only visible stains
            skipped += 1; continue
        im[py:py + th, px:px + tw] = new
        rows.append(f"0 {(px + tw / 2) / W:.6f} {(py + th / 2) / H:.6f} {tw / W:.6f} {th / H:.6f}")
    if not rows: continue
    name = f"synthworn_{base.stem[:50]}"
    cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n")
    made += 1
print(f"bases available {len(bases)}, synthetic worn images {made}, invisible pastes skipped {skipped}")
