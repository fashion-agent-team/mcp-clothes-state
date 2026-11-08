"""Clean-clothing negatives (CLOTHES / clothes-upt5u-lutml, public domain): dedupe, hold out 200 for a false-alarm
test, run the current model over the rest; images where it fires are hard negatives (reviewed by eye before use)."""
import hashlib, random, re, shutil, sys
from pathlib import Path
from ultralytics import YOLO

R = Path(sys.argv[1]); src = R / "raw2/clothes_quoc"; out = R / "negatives"
for d in ("pool", "heldout"):
    (out / d).mkdir(parents=True, exist_ok=True)
seen_md5, seen_stem = set(), set()
for p in list((R / "raw").rglob("*.jpg")) + list((R / "raw2").rglob("*.jpg")):
    if "clothes_quoc" not in str(p):
        seen_stem.add(re.sub(r"\.rf\..*$", "", p.name).lower())
imgs = []
for p in sorted(src.glob("*/images/*")):
    h = hashlib.md5(p.read_bytes()).hexdigest()
    stem = re.sub(r"\.rf\..*$", "", p.name).lower()
    if h in seen_md5 or stem in seen_stem:
        continue
    seen_md5.add(h); seen_stem.add(stem); imgs.append(p)
random.Random(0).shuffle(imgs)
held, pool = imgs[:200], imgs[200:]
for d, lst in (("heldout", held), ("pool", pool)):
    for p in lst:
        dst = out / d / f"neg_{p.name}"
        if not dst.exists(): dst.symlink_to(p.resolve())
model = YOLO(R / "runs/yolo26n_v2/weights/best.pt")
hard = []
for r in model.predict(source=str(out / "pool"), imgsz=512, conf=0.2, device=0, verbose=False, stream=True):
    if len(r.boxes):
        hard.append((Path(r.path).name, [(int(b.cls), round(float(b.conf), 3), [round(v, 4) for v in b.xywhn[0].tolist()]) for b in r.boxes]))
(out / "hard.tsv").write_text("".join(f"{n}\t{d}\n" for n, d in hard))
print(f"unique {len(imgs)}, heldout {len(held)}, pool {len(pool)}, hard (fires at 0.2) {len(hard)}")
