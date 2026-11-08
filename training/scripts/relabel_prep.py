"""Pre-label the low-confidence autolabel train photos skipped in the first review, with yolo26n_v6_tuned (conf 0.15),
and draw numbered review sheets (boxes numbered per image) for a second, accept/reject review."""
import importlib.util, json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

R = Path(sys.argv[1]); AL = R / "autolabel"; out = AL / "review2"; out.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("rd", R / "scripts/review_decisions.py"); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
idx = {int(n): name for n, name, _ in (l.split("\t") for l in (AL / "review/index.tsv").read_text().splitlines() if l)}
todo = [(n, idx[n]) for n in sorted(idx) if idx[n].startswith("train__") and n not in rd.KEEP]
m = YOLO(R / "runs/yolo26n_v6_tuned/weights/best.pt")
preds, tiles = {}, []
T = 470
for n, name in todo:
    im = cv2.imread(str(AL / "images" / name))
    r = m.predict(im, imgsz=512, conf=0.15, device=0, verbose=False)[0]
    boxes = [(int(b.cls), round(float(b.conf), 3), [round(v, 4) for v in b.xywhn[0].tolist()]) for b in r.boxes]
    preds[n] = {"name": name, "boxes": boxes}
    h, w = im.shape[:2]; k = T / max(h, w); im = cv2.resize(im, (int(w * k), int(h * k))); h, w = im.shape[:2]
    for i, (c, cf, (x, y, bw, bh)) in enumerate(boxes):
        col = (0, 255, 255) if c == 0 else (255, 160, 0)
        cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), col, 2)
        cv2.putText(im, f"{i}", (int((x - bw / 2) * w) + 2, int((y - bh / 2) * h) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    t = np.full((T + 26, T, 3), 30, np.uint8); t[:h, :w] = im
    cv2.putText(t, f"#{n}", (4, T + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); tiles.append(t)
(out / "preds.json").write_text(json.dumps(preds))
for s in range(0, len(tiles), 16):
    ts = tiles[s:s + 16]
    while len(ts) % 4: ts.append(np.full_like(tiles[0], 30))
    cv2.imwrite(str(out / f"r2_{s // 16 + 1:02d}.jpg"), np.vstack([np.hstack(ts[i:i + 4]) for i in range(0, len(ts), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(todo), "images")
