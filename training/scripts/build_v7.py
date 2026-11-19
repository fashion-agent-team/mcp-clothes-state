"""dataset_v7 = dataset_v5 + second-review autolabel photos (whole-garment phone photos) + synthetic stains on worn
photos. Any new image whose bytes match an image in the v2 test split, the clean held-out set or the real-scenario
photos is skipped."""
import hashlib, importlib.util, json, shutil, sys
from pathlib import Path

R = Path(sys.argv[1]); OUT = R / "dataset_v7"; AL = R / "autolabel"
spec = importlib.util.spec_from_file_location("rd2", R / "scripts/review2_decisions.py"); rd2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd2)
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
protected = {md5(p) for d in (R / "dataset_v2/test/images", R / "negatives/heldout", R / "realtest/aleminaitha/test/images") for p in d.iterdir()}
(OUT / "auto2/images").mkdir(parents=True, exist_ok=True); (OUT / "auto2/labels").mkdir(parents=True, exist_ok=True)
preds = json.loads((AL / "review2/preds.json").read_text())
n_img = n_box = n_skip = 0
for n, keep in rd2.KEEP.items():
    p = preds[str(n)]; src = AL / "images" / p["name"]
    if md5(src) in protected: n_skip += 1; continue
    rows = [p["boxes"][i][2] for i in keep if p["boxes"][i][0] == 0] + [list(b) for b in rd2.ADD.get(n, [])]
    dst = OUT / "auto2/images" / f"auto2_{p['name']}"
    if not dst.exists(): dst.symlink_to(src.resolve())
    (OUT / "auto2/labels" / f"auto2_{Path(p['name']).stem}.txt").write_text("".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in rows))
    n_img += 1; n_box += len(rows)
n_syn = sum(1 for p in (R / "synth_worn/images").iterdir() if md5(p) not in protected)
(OUT / "data.yaml").write_text(
    f"path: {R}\ntrain:\n  - {R}/dataset_v2/train/images\n  - {R}/dataset_v3/auto/images\n  - {R}/synth/images\n"
    f"  - {R}/dataset_v5/neg/images\n  - {R}/dataset_v5/torn2x/images\n  - {OUT}/auto2/images\n  - {R}/synth_worn/images\n"
    f"val: {R}/dataset_v2/valid/images\ntest: {R}/dataset_v2/test/images\nnc: 2\nnames: ['stain', 'torn']\n")
print(f"second-review photos {n_img} ({n_box} boxes, {n_skip} skipped as duplicates of eval photos), synthetic worn {n_syn}")
