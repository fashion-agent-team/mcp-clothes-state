"""B scoring: v2 test mAP50 (all/stain/torn) via Ultralytics val on split=test for each run name.
usage: python eval_B.py <out json> <run> [run...]"""
import json, sys
from pathlib import Path
from ultralytics import YOLO
R = Path.home() / "clothes-state-train"; out = {}
for run in sys.argv[2:]:
    m = YOLO(R / f"runs/{run}/weights/best.pt")
    r = m.val(data=str(R / "exp3/data_v6ft.yaml"), split="test", imgsz=512, batch=8, device=0, plots=False, verbose=False,
              project=str(R / "exp3/val"), name=run, exist_ok=True)
    ap50 = r.box.all_ap[:, 0] if hasattr(r.box, "all_ap") else r.box.ap50
    out[run] = {"map50": round(float(r.box.map50), 3), "stain_ap50": round(float(ap50[0]), 3), "torn_ap50": round(float(ap50[1]), 3),
                "map50_95": round(float(r.box.map), 3)}
    print(run, out[run], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
