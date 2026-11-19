"""Draw predicted boxes (with conf) on the autolabel images, 24 per review sheet, sorted by max confidence."""
import sys
from pathlib import Path
import cv2, numpy as np
R = Path(sys.argv[1]); imgs = R / "autolabel/images"; labs = R / "autolabel/pred/labels"; out = R / "autolabel/sheets"
out.mkdir(parents=True, exist_ok=True)
items = []
for p in sorted(imgs.iterdir()):
    f = labs / f"{p.stem}.txt"
    rows = [list(map(float, l.split())) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    items.append((max((r[5] for r in rows), default=0), p, rows))
items.sort(key=lambda x: -x[0])
T, COLS = 320, 6
for s in range(0, len(items), 24):
    tiles = []
    for mx, p, rows in items[s:s + 24]:
        im = cv2.imread(str(p)); h, w = im.shape[:2]; k = T / max(h, w)
        im = cv2.resize(im, (int(w * k), int(h * k))); h, w = im.shape[:2]
        for c, x, y, bw, bh, cf in rows:
            col = (0, 0, 255) if cf >= 0.5 else (0, 200, 255)
            if c == 1: col = (255, 160, 0)
            x0, y0, x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
            cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
            cv2.putText(im, f"{cf:.2f}", (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        canvas = np.full((T + 22, T, 3), 30, np.uint8); canvas[:h, :w] = im
        cv2.putText(canvas, f"{p.stem[:40]}", (3, T + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
        tiles.append(canvas)
    while len(tiles) % COLS: tiles.append(np.full_like(tiles[0], 30))
    sheet = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    cv2.imwrite(str(out / f"sheet_{s // 24 + 1:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
hi = sum(m >= 0.5 for m, _, _ in items); mid = sum(0.25 <= m < 0.5 for m, _, _ in items)
print(f"images {len(items)}: >=0.5 {hi}, 0.25-0.5 {mid}, none {len(items) - hi - mid}; sheets {(len(items) + 23) // 24}")
