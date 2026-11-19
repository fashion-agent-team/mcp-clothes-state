"""yolo26n from pretrained weights with the hyperparameters found by the overnight tuner.
usage: train_v6.py <root> <batch> [dataset dir] [run name]  (defaults: dataset_v5, yolo26n_v6_tuned)"""
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

R = Path(sys.argv[1]); batch = int(sys.argv[2]) if len(sys.argv) > 2 else 16
data = sys.argv[3] if len(sys.argv) > 3 else "dataset_v5"; name = sys.argv[4] if len(sys.argv) > 4 else "yolo26n_v6_tuned"
hyp = yaml.safe_load((R / "runs/tune_v5/best_hyperparameters.yaml").read_text())  # comment lines are ignored
YOLO(R / "weights/yolo26n.pt").train(
    data=str(R / data / "data.yaml"), imgsz=512, epochs=60, patience=15, batch=batch, device=0, workers=2,
    project=str(R / "runs"), name=name, exist_ok=True, **hyp)
