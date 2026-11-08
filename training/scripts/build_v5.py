"""dataset_v5 = dataset_v4 minus distressed/ripped-design negatives, with every torn training image from dataset_v2
added a second time (as differently named links, since the label cache is keyed by path) so tears weigh 2x."""
import sys
from pathlib import Path

R = Path(sys.argv[1]); V4 = R / "dataset_v4"; OUT = R / "dataset_v5"
DROP_NUMS = {285, 326, 321, 144}  # hard-negative review numbers: ripped jeans, frayed denim, distressed skirt
order = {int(n): name for n, name in (l.split("\t") for l in (R / "negatives/review/order.tsv").read_text().splitlines() if l)}
drop = {order[n] for n in DROP_NUMS}
for sub in ("neg", "torn2x"):
    (OUT / sub / "images").mkdir(parents=True, exist_ok=True); (OUT / sub / "labels").mkdir(parents=True, exist_ok=True)
n_neg = 0
for img in sorted((V4 / "neg/images").iterdir()):
    if img.name in drop: continue
    dst = OUT / "neg/images" / img.name
    if not dst.exists(): dst.symlink_to(img.resolve())
    (OUT / "neg/labels" / (img.stem + ".txt")).write_text("")
    n_neg += 1
n_torn = 0
for lab in sorted((R / "dataset_v2/train/labels").glob("*.txt")):
    rows = lab.read_text().splitlines()
    if not any(r.split()[0] == "1" for r in rows if r.strip()): continue
    img = next((R / "dataset_v2/train/images").glob(lab.stem + ".*"))
    dst = OUT / "torn2x/images" / f"dup_{img.name}"
    if not dst.exists(): dst.symlink_to(img.resolve())
    (OUT / "torn2x/labels" / f"dup_{lab.name}").write_text(lab.read_text())
    n_torn += 1
(OUT / "data.yaml").write_text(
    f"path: {R}\ntrain:\n  - {R}/dataset_v2/train/images\n  - {R}/dataset_v3/auto/images\n  - {R}/synth/images\n"
    f"  - {OUT}/neg/images\n  - {OUT}/torn2x/images\nval: {R}/dataset_v2/valid/images\ntest: {R}/dataset_v2/test/images\n"
    f"nc: 2\nnames: ['stain', 'torn']\n")
print(f"negatives {n_neg} (dropped {len(drop)}), torn images duplicated {n_torn}")
