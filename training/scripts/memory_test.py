"""Check garment memory and multi-photo filtering on real-scenario photos (CPU, so the GPU training is untouched)."""
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent))
import clothes_inspect_model as cim
R = Path(sys.argv[1]); cim.ASK = 0.2  # lower ask cut-off here only, so the test photos have uncertain spots to filter
ins = cim.ClothesInspector(R / "runs/yolo26n_v5/weights/best.pt", R / "weights/yolo11n-pose.pt", device="cpu")
imgs = sorted((R / "realtest/aleminaitha/test/images").iterdir())
a = cv2.imread(str(imgs[8]))
first = ins.inspect(a, "laid")
asks = [d for d in first["defects"] if d["status"] == "ask"]
print("first look:", [(d["where"], d["confidence"]) for d in asks], "askImage:", bool(first["askImage"]))
dismissed = [{"kind": d["kind"], "where": d["where"], "box": d["box"]} for d in asks]  # friend said "no" to all
again = ins.inspect(a, "laid", dismissed=dismissed)
print("same garment again:", [(d["where"], d["status"]) for d in again["defects"]], "askImage:", bool(again["askImage"]))
views = ins.inspect_views([a, cv2.imread(str(imgs[22]))], "laid")
print("two photos:", [[(d["where"], d["status"]) for d in r["defects"]] for r in views])
