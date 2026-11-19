"""Sanity sheet for synthetic images: 24 random samples with their auto boxes (green)."""
import random, sys
from pathlib import Path
import cv2, numpy as np
R = Path(sys.argv[1]); d = R / (sys.argv[2] if len(sys.argv) > 2 else "synth"); out = R / "synth_sheets"; out.mkdir(exist_ok=True)
imgs = sorted((d / "images").iterdir()); random.Random(1).shuffle(imgs)
T = 420
for k in range(2):
    tiles = []
    for p in imgs[k * 12:(k + 1) * 12]:
        im = cv2.imread(str(p)); h, w = im.shape[:2]; s = T / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s))); h, w = im.shape[:2]
        for l in (d / "labels" / f"{p.stem}.txt").read_text().split("\n"):
            if not l.strip(): continue
            _, x, y, bw, bh = map(float, l.split())
            cv2.rectangle(im, (int((x - bw / 2) * w) - 3, int((y - bh / 2) * h) - 3), (int((x + bw / 2) * w) + 3, int((y + bh / 2) * h) + 3), (0, 255, 0), 1)
        c = np.full((T, T, 3), 30, np.uint8); c[:h, :w] = im; tiles.append(c)
    cv2.imwrite(str(out / f"synth_check_{k + 1}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
