import sys, glob
from ultralytics import YOLO
m = YOLO(sys.argv[1]); d = sys.argv[2]; pos = sys.argv[3]
res = {}
for cls in sorted(glob.glob(d + '/val/*')):
    name = cls.split('/')[-1]
    fs = sorted(glob.glob(cls + '/*.jpg'))
    ps = []
    for i in range(0, len(fs), 64):
        for r in m.predict(fs[i:i+64], verbose=False, device='cpu'):
            ps.append(float(r.probs.data[r.names_inv[pos]] if hasattr(r,'names_inv') else r.probs.data[[k for k,v in r.names.items() if v==pos][0]]))
    res[name] = ps
for t in (0.5, 0.7, 0.8, 0.9):
    print(t, {k: round(sum(p >= t for p in v) / len(v), 3) for k, v in res.items()})
