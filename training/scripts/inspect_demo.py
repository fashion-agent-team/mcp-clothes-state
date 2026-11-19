"""Run clothes_inspect_model on a few photos and save annotated results (keypoints, boxes, where text)."""
import json, sys, base64
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from clothes_inspect_model import ClothesInspector, KP_CONF

R = Path(sys.argv[1]); out = R / "inspect_demo"; out.mkdir(exist_ok=True)
ins = ClothesInspector(R / "runs/yolo26n_v6_tuned/weights/best.pt", R / "weights/yolo11n-pose.pt")
cases = json.loads(sys.argv[2])  # [[path, mode, view, mirrored], ...]
summary = []
for i, (p, mode, view, mirrored) in enumerate(cases):
    img = cv2.imread(str(R / p))
    res = ins.inspect(img, mode, view, mirrored)
    vis = cv2.flip(img, 1) if mirrored else img.copy()
    h, w = vis.shape[:2]
    if mode == "worn":
        person = ins._person(vis)
        if person is not None:
            for j, (x, y) in enumerate(person["kp"]):
                if person["kc"][j] >= KP_CONF:
                    cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 255), -1)
    for d in res["defects"]:
        b = d["box"]; col = (0, 0, 255) if d["status"] == "confirmed" else (0, 200, 255)
        cv2.rectangle(vis, (int(b["x"] * w), int(b["y"] * h)), (int((b["x"] + b["w"]) * w), int((b["y"] + b["h"]) * h)), col, 3)
    cv2.imwrite(str(out / f"case{i}.jpg"), vis)
    if res["askImage"]:
        (out / f"case{i}_ask.jpg").write_bytes(base64.b64decode(res["askImage"]))
    res["askImage"] = bool(res["askImage"])
    summary.append({"case": i, "file": Path(p).name[:40], "mode": mode, "view": view, "mirrored": mirrored, **res})
print(json.dumps(summary, ensure_ascii=False, indent=1))
