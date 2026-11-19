"""Hyperparameter search (Ultralytics tuner: mutates the best settings so far each iteration) for yolo26n on the
v5 dataset at 512. Short 10-epoch runs, so fitness is only a relative signal for picking settings."""
import sys
from pathlib import Path
from ultralytics import YOLO

R = Path(sys.argv[1]); batch = int(sys.argv[2]) if len(sys.argv) > 2 else 16
YOLO(R / "weights/yolo26n.pt").tune(
    data=str(R / "dataset_v5/data.yaml"), imgsz=512, epochs=10, iterations=30, batch=batch, device=0, workers=2,
    plots=False, save=False, val=True, project=str(R / "runs"), name="tune_v5", exist_ok=True)
