"""Real-scenario check without retraining: tiled inference (overlapping tiles + NMS merge) and plain imgsz=1024.
Writes YOLO-format predictions (cls cx cy w h conf, normalized to the full photo) and per-photo timings."""
import sys, time
from pathlib import Path
import cv2, torch, torchvision
from ultralytics import YOLO

R = Path(sys.argv[1])
SRC = R / "realtest/aleminaitha/test/images"
OUT = R / "runs/realtest_tiling"
COLS, ROWS, OVERLAP, CONF = 2, 3, 0.2, 0.25

def tiles(w, h):
    tw, th = w / (COLS - (COLS - 1) * OVERLAP), h / (ROWS - (ROWS - 1) * OVERLAP)
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = int(c * tw * (1 - OVERLAP)), int(r * th * (1 - OVERLAP))
            yield x0, y0, min(w, int(x0 + tw)), min(h, int(y0 + th))

def write(path, boxes, w, h):
    path.write_text("".join(f"{int(c)} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f} {s:.4f}\n"
                            for x0, y0, x1, y1, s, c in boxes))

for m in ("yolo26n_v2", "yolo11s_v2"):
    model = YOLO(R / f"runs/{m}/weights/best.pt")
    model.predict(cv2.imread(str(next(SRC.iterdir()))), imgsz=512, device=0, verbose=False)  # warm-up
    for mode in ("tile2x3", "plain1024"):
        od = OUT / f"{m}_{mode}"; od.mkdir(parents=True, exist_ok=True)
        times = []
        for img_p in sorted(SRC.iterdir()):
            img = cv2.imread(str(img_p)); h, w = img.shape[:2]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            boxes = []
            if mode == "plain1024":
                r = model.predict(img, imgsz=1024, conf=CONF, device=0, verbose=False)[0]
                boxes = [(*b.xyxy[0].tolist(), float(b.conf), int(b.cls)) for b in r.boxes]
            else:
                for x0, y0, x1, y1 in tiles(w, h):
                    r = model.predict(img[y0:y1, x0:x1], imgsz=512, conf=CONF, device=0, verbose=False)[0]
                    boxes += [(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0, float(b.conf), int(b.cls))
                              for b in r.boxes for bx0, by0, bx1, by1 in [b.xyxy[0].tolist()]]
                if boxes:
                    t = torch.tensor(boxes)
                    keep = torchvision.ops.batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), 0.5)
                    boxes = [tuple(t[i].tolist()) for i in keep]
            torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
            write(od / f"{img_p.stem}.txt", boxes, w, h)
        print(f"{m} {mode}: {sum(times)/len(times):.1f} ms/photo, boxes {sum(1 for f in od.glob('*.txt') for l in f.read_text().splitlines())}")
