"""Review sheets for hard negatives, highest model confidence first, 30 per sheet, numbered."""
import ast, sys
from pathlib import Path
import cv2, numpy as np
R = Path(sys.argv[1]); out = R / "negatives/review"; out.mkdir(exist_ok=True)
rows = [l.split("\t") for l in (R / "negatives/hard.tsv").read_text().splitlines() if l]
items = sorted(((n, ast.literal_eval(d)) for n, d in rows), key=lambda x: -max(c for _, c, _ in x[1]))
(out / "order.tsv").write_text("".join(f"{i + 1}\t{n}\n" for i, (n, _) in enumerate(items)))
T, COLS = 300, 6
for s in range(0, len(items), 30):
    tiles = []
    for j, (n, dets) in enumerate(items[s:s + 30]):
        im = cv2.imread(str(R / "negatives/pool" / n)); h, w = im.shape[:2]; k = T / max(h, w)
        im = cv2.resize(im, (int(w * k), int(h * k))); h, w = im.shape[:2]
        for c, cf, (x, y, bw, bh) in dets:
            col = (0, 0, 255) if c == 0 else (255, 160, 0)
            cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), col, 2)
            cv2.putText(im, f"{cf:.2f}", (int((x - bw / 2) * w), max(12, int((y - bh / 2) * h) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        t = np.full((T + 22, T, 3), 255, np.uint8); t[:h, :w] = im
        cv2.putText(t, f"#{s + j + 1}", (4, T + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        tiles.append(t)
    while len(tiles) % COLS: tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(out / f"neg_{s // 30 + 1:02d}.jpg"), np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(items), "items")
