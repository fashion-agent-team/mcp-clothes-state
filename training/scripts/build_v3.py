"""dataset_v3 = dataset_v2 + reviewed autolabel (train split only, so the v2 valid/test stay unchanged and free of
model-made labels) + synthetic whole-garment stains. Train list is written as a txt so v2 files are not copied."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from review_decisions import KEEP, ADD

R = Path(sys.argv[1]); AL = R / "autolabel"; OUT = R / "dataset_v3"
(OUT / "auto/images").mkdir(parents=True, exist_ok=True); (OUT / "auto/labels").mkdir(parents=True, exist_ok=True)
review_idx = {int(n): name for n, name, _ in (l.split("\t") for l in (AL / "review/index.tsv").read_text().splitlines() if l)}
low = set(review_idx.values())
stats = {"high_imgs": 0, "high_boxes": 0, "low_imgs": 0, "low_boxes": 0, "low_added": 0, "skipped_not_train": 0}

def preds(name):
    f = AL / "pred/labels" / (Path(name).stem + ".txt")
    return [l.split() for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

def emit(name, rows):
    src = AL / "images" / name
    dst = OUT / "auto/images" / f"auto_{name}"
    if not dst.exists(): dst.symlink_to(src.resolve())
    (OUT / "auto/labels" / f"auto_{Path(name).stem}.txt").write_text("".join(f"0 {x} {y} {w} {h}\n" for x, y, w, h in rows))

for img in sorted((AL / "images").iterdir()):
    name = img.name
    if not name.startswith("train__"):
        stats["skipped_not_train"] += 1; continue
    if name in low: continue
    rows = [r[1:5] for r in preds(name) if r[0] == "0" and float(r[5]) >= 0.5]
    if rows:
        emit(name, rows); stats["high_imgs"] += 1; stats["high_boxes"] += len(rows)
for n, keep in KEEP.items():
    name = review_idx[n]
    if not name.startswith("train__"):
        stats["skipped_not_train"] += 1; continue
    p = preds(name)
    rows = [p[i][1:5] for i in keep if p[i][0] == "0"] + [tuple(f"{v:.4f}" for v in b) for b in ADD.get(n, [])]
    if rows:
        emit(name, rows); stats["low_imgs"] += 1; stats["low_boxes"] += len(rows); stats["low_added"] += len(ADD.get(n, []))
(OUT / "train.txt").write_text("".join(f"{d}\n" for d in (R / "dataset_v2/train/images", OUT / "auto/images", R / "synth/images")))
(OUT / "data.yaml").write_text(
    f"path: {R}\ntrain:\n  - {R}/dataset_v2/train/images\n  - {OUT}/auto/images\n  - {R}/synth/images\n"
    f"val: {R}/dataset_v2/valid/images\ntest: {R}/dataset_v2/test/images\nnc: 2\nnames: ['stain', 'torn']\n")
print(stats)
