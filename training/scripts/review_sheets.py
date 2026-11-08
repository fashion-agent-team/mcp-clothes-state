"""Numbered review sheets for low-confidence autolabel images (max conf < 0.5), 16 per sheet, boxes labelled by index."""
import sys
from pathlib import Path
import cv2, numpy as np
R = Path(sys.argv[1]); imgs = R / "autolabel/images"; labs = R / "autolabel/pred/labels"; out = R / "autolabel/review"
out.mkdir(parents=True, exist_ok=True)
items = []
for p in sorted(imgs.iterdir()):
    f = labs / f"{p.stem}.txt"
    rows = [list(map(float, l.split())) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    if max((r[5] for r in rows), default=0) < 0.5:
        items.append((p, rows))
index = []
T, COLS = 470, 4
for s in range(0, len(items), 16):
    tiles = []
    for j, (p, rows) in enumerate(items[s:s + 16]):
        n = s + j + 1
        im = cv2.imread(str(p)); h, w = im.shape[:2]; k = T / max(h, w)
        im = cv2.resize(im, (int(w * k), int(h * k))); h, w = im.shape[:2]
        for b, (c, x, y, bw, bh, cf) in enumerate(rows):
            x0, y0, x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255) if c == 0 else (255, 160, 0), 2)
            cv2.putText(im, f"{b}", (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        canvas = np.full((T + 26, T, 3), 30, np.uint8); canvas[:h, :w] = im
        cv2.putText(canvas, f"#{n}", (4, T + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(canvas); index.append(f"{n}\t{p.name}\t{len(rows)}")
    while len(tiles) % COLS: tiles.append(np.full_like(tiles[0], 30))
    cv2.imwrite(str(out / f"review_{s // 16 + 1:02d}.jpg"), np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
(out / "index.tsv").write_text("\n".join(index) + "\n")
print(len(items), "images,", (len(items) + 15) // 16, "sheets")
