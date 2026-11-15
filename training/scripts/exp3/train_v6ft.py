"""v6 fine-tune: start from v6 best.pt, v6 tuned hyperparameters/augmentation, new data (exp3/data_v6ft.yaml).
usage: python train_v6ft.py <batch> <run name>"""
import os, sys, yaml
from pathlib import Path
from ultralytics import YOLO
R = Path.home() / "clothes-state-train"; B = int(sys.argv[1]); NAME = sys.argv[2]
(R / "exp3/train.pid").write_text(str(os.getpid()))
v6 = yaml.safe_load((R / "runs/yolo26n_v6_tuned/args.yaml").read_text())
keep = "momentum weight_decay warmup_momentum warmup_bias_lr box cls cls_pw dfl nbs hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr bgr mosaic mixup cutmix copy_paste copy_paste_mode auto_augment erasing close_mosaic amp seed deterministic".split()
hyp = {k: v6[k] for k in keep if k in v6}
YOLO(R / "runs/yolo26n_v6_tuned/weights/best.pt").train(
    data=str(R / "exp3/data_v6ft.yaml"), epochs=20, imgsz=512, batch=B, workers=2, lr0=0.001, lrf=0.1,
    warmup_epochs=0, optimizer="SGD", patience=8, project=str(R / "runs"), name=NAME, exist_ok=True, device=0, **hyp)
(R / "exp3/train.pid").unlink(missing_ok=True)
