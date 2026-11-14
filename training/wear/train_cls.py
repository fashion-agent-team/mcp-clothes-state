import sys
from ultralytics import YOLO
data, name, imgsz, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
m = YOLO('yolo26n-cls.pt')
m.train(data=data, imgsz=imgsz, epochs=epochs, batch=64, device='cpu', workers=4, project='runs',
        name=name, exist_ok=True, patience=5, fliplr=0.5, hsv_v=0.3, hsv_s=0.4, degrees=5, translate=0.1, scale=0.3,
        erasing=0.0, verbose=False, plots=False)
