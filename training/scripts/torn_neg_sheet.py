"""Sheet of hard negatives where yolo26n_v2 predicted 'torn', to drop distressed/ripped-design garments."""
import ast, sys
from pathlib import Path
import cv2, numpy as np
R = Path(sys.argv[1]); N = R / "negatives"
order = {name: int(n) for n, name in (l.split("\t") for l in (N / "review/order.tsv").read_text().splitlines() if l)}
rows = [l.split("\t") for l in (N / "hard.tsv").read_text().splitlines() if l]
items = sorted(((order[n], n, ast.literal_eval(d)) for n, d in rows if any(c == 1 for c, _, _ in ast.literal_eval(d))))
T, COLS, tiles = 260, 8, []
for num, n, dets in items:
    im = cv2.imread(str(N / "pool" / n)); h, w = im.shape[:2]; k = T / max(h, w)
    im = cv2.resize(im, (int(w * k), int(h * k)))
    t = np.full((T + 22, T, 3), 255, np.uint8); t[:im.shape[0], :im.shape[1]] = im
    cv2.putText(t, f"#{num}", (4, T + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2); tiles.append(t)
while len(tiles) % COLS: tiles.append(np.full_like(tiles[0], 255))
cv2.imwrite(str(N / "review/torn_called.jpg"), np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(items))
