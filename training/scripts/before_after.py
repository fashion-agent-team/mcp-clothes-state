"""Before/after sheet for reviewed autolabel images: left = model boxes (yellow), right = final labels (green)."""
import sys
from pathlib import Path
import cv2, numpy as np
R = Path(sys.argv[1]); AL = R / "autolabel"; V3 = R / "dataset_v3/auto"; nums = list(map(int, sys.argv[2].split(",")))
idx = {int(n): name for n, name, _ in (l.split("\t") for l in (AL / "review/index.tsv").read_text().splitlines() if l)}
T = 380; rows = []
def draw(im, lines, col, conf):
    h, w = im.shape[:2]
    for l in lines:
        v = l.split(); x, y, bw, bh = map(float, v[1:5])
        cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), col, 2)
    return im
for n in nums:
    name = idx[n]; img = cv2.imread(str(AL / "images" / name)); h, w = img.shape[:2]; k = T / max(h, w)
    img = cv2.resize(img, (int(w * k), int(h * k)))
    pf = AL / "pred/labels" / (Path(name).stem + ".txt"); ff = V3 / "labels" / f"auto_{Path(name).stem}.txt"
    a = draw(img.copy(), pf.read_text().splitlines() if pf.exists() else [], (0, 220, 255), True)
    b = draw(img.copy(), ff.read_text().splitlines() if ff.exists() else [], (0, 255, 0), False)
    pair = np.full((T + 24, 2 * T + 10, 3), 30, np.uint8); pair[:a.shape[0], :a.shape[1]] = a; pair[:b.shape[0], T + 10:T + 10 + b.shape[1]] = b
    cv2.putText(pair, f"#{n} before (model) | after (reviewed)", (4, T + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    rows.append(pair)
while len(rows) % 2: rows.append(np.full_like(rows[0], 30))
cv2.imwrite(sys.argv[3], np.vstack([np.hstack([rows[i], np.full((rows[i].shape[0], 16, 3), 0, np.uint8), rows[i + 1]]) for i in range(0, len(rows), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
