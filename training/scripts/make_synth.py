"""Synthetic whole-garment stains: paste feathered stain crops (from labeled Clothes-stain train boxes) inside the
garment box of More clothes 'good' train images. Only the stain's darkening relative to its own fabric is transferred
(ratio to the crop's border median), so the garment's colour and texture show through. Writes YOLO labels (class 0)."""
import random, sys
from pathlib import Path
import cv2, numpy as np

R = Path(sys.argv[1]); VARIANTS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SRC_STAIN, SRC_GOOD, OUT = R / "raw/stain/train", R / "raw/more/train", R / "synth"
(OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(parents=True, exist_ok=True)
rng = random.Random(0)

def boxes(lab, cls):
    return [list(map(float, l.split()[1:5])) for l in lab.read_text().splitlines() if l.strip() and l.split()[0] == cls] if lab.exists() else []

# stain crops: one box per labeled image, not too tiny, not covering most of the photo
crops = []
for img in sorted((SRC_STAIN / "images").iterdir()):
    bs = boxes(SRC_STAIN / "labels" / f"{img.stem}.txt", "0")
    bs = [b for b in bs if 0.04 < b[2] < 0.6 and 0.04 < b[3] < 0.6]
    if bs:
        crops.append((img, rng.choice(bs)))
rng.shuffle(crops)

def stain_patch(img_p, b):
    im = cv2.imread(str(img_p)).astype(np.float32); h, w = im.shape[:2]
    x, y, bw, bh = b
    x0, y0 = max(0, int((x - bw / 2) * w)), max(0, int((y - bh / 2) * h))
    x1, y1 = min(w, int((x + bw / 2) * w)), min(h, int((y + bh / 2) * h))
    c = im[y0:y1, x0:x1]
    if c.shape[0] < 12 or c.shape[1] < 12: return None
    border = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    ratio = np.clip(c / (np.median(border, axis=0) + 1), 0.15, 1.0)        # darkening only
    return ratio

def feather(hh, ww):
    yy, xx = np.mgrid[0:hh, 0:ww]
    d = ((xx - ww / 2) / (ww / 2)) ** 2 + ((yy - hh / 2) / (hh / 2)) ** 2
    return np.clip((1.15 - d) / 0.45, 0, 1)[..., None]

goods = [p for p in sorted((SRC_GOOD / "images").iterdir()) if boxes(SRC_GOOD / "labels" / f"{p.stem}.txt", "0")]
ci, made, skipped, MIN_DROP = 0, 0, 0, 18.0
for g in goods:
    base = cv2.imread(str(g)).astype(np.float32); H, W = base.shape[:2]
    gx, gy, gw, gh = boxes(SRC_GOOD / "labels" / f"{g.stem}.txt", "0")[0]
    GX0, GY0, GW, GH = (gx - gw / 2) * W, (gy - gh / 2) * H, gw * W, gh * H
    for v in range(VARIANTS):
        im, rows = base.copy(), []
        for _ in range(rng.randint(1, 4) * 3):
            if len(rows) >= 4: break
            for _try in range(10):
                p, b = crops[ci % len(crops)]; ci += 1
                ratio = stain_patch(p, b)
                if ratio is not None: break
            tw = max(8, int(GW * rng.uniform(0.02, 0.07)))                  # small: 2-7% of garment width
            th = max(8, int(tw * ratio.shape[0] / ratio.shape[1] * rng.uniform(0.8, 1.25)))
            r = cv2.resize(ratio, (tw, th), interpolation=cv2.INTER_AREA)
            M = cv2.getRotationMatrix2D((tw / 2, th / 2), rng.uniform(0, 360), 1.0)
            r = cv2.warpAffine(r, M, (tw, th), borderValue=(1, 1, 1))
            strength = rng.uniform(0.8, 1.0)
            a = feather(th, tw) * strength
            # keep inside the inner 70% of the garment box (its corners are often background)
            px = int(GX0 + GW * 0.15 + rng.random() * max(1, GW * 0.7 - tw))
            py = int(GY0 + GH * 0.15 + rng.random() * max(1, GH * 0.7 - th))
            px, py = min(max(px, 0), W - tw), min(max(py, 0), H - th)
            roi = im[py:py + th, px:px + tw]
            new = roi * (1 - a + a * r)
            core = a[..., 0] > 0.5
            # only keep stains that are actually visible (mean luminance drop in the core region)
            if not core.any() or (roi[core] - new[core]).mean() < MIN_DROP:
                skipped += 1; continue
            im[py:py + th, px:px + tw] = new
            rows.append(f"0 {(px + tw / 2) / W:.6f} {(py + th / 2) / H:.6f} {tw / W:.6f} {th / H:.6f}")
        if not rows:
            continue
        name = f"synth_{g.stem[:60]}_{v}"
        cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n")
        made += 1
print(f"good images {len(goods)}, stain crops {len(crops)}, synthetic images {made}, invisible pastes skipped {skipped}")
