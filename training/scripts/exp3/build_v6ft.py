"""Build exp3 train extras for v6ft: auto2 12 photos x5 (renamed copies) + 400 negatives/pool photos
not in heldout or dataset_v5/neg (checked by md5 and by the original name before Roboflow's .rf. hash)."""
import hashlib, random, shutil
from pathlib import Path
R = Path.home() / "clothes-state-train"; E = R / "exp3"
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
base = lambda p: p.name.split("_jpg.rf.")[0].split(".rf.")[0]
img = lambda d: sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
a = E / "auto2x5"; (a / "images").mkdir(parents=True, exist_ok=True); (a / "labels").mkdir(exist_ok=True)
for p in img(R / "dataset_v7/auto2/images"):
    lab = R / "dataset_v7/auto2/labels" / (p.stem + ".txt")
    for k in range(5):
        shutil.copy(p, a / "images" / f"rep{k}_{p.name}"); shutil.copy(lab, a / "labels" / f"rep{k}_{p.stem}.txt")
excl = img(R / "negatives/heldout") + img(R / "dataset_v5/neg/images")
ex_md5 = {md5(p) for p in excl}; ex_base = {base(p) for p in excl}
cand = [p for p in img(R / "negatives/pool") if base(p) not in ex_base and md5(p) not in ex_md5]
held_md5 = {md5(p) for p in img(R / "negatives/heldout")}
random.Random(0).shuffle(cand); pick = cand[:400]
assert not ({md5(p) for p in pick} & held_md5)
n = E / "neg400"; (n / "images").mkdir(parents=True, exist_ok=True); (n / "labels").mkdir(exist_ok=True)
for p in pick:
    shutil.copy(p, n / "images" / p.name); (n / "labels" / (p.stem + ".txt")).write_text("")
print("auto2x5", len(img(a / "images")), "pool candidates", len(cand), "picked", len(pick))
(E / "data_v6ft.yaml").write_text(f"""path: {R}
train:
  - {R}/dataset_v2/train/images
  - {R}/dataset_v3/auto/images
  - {R}/synth/images
  - {R}/dataset_v5/neg/images
  - {R}/dataset_v5/torn2x/images
  - {E}/auto2x5/images
  - {E}/neg400/images
val: {R}/dataset_v2/valid/images
test: {R}/dataset_v2/test/images
nc: 2
names: ['stain', 'torn']
""")
