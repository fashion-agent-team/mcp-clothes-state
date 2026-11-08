"""dataset_v4 = dataset_v3 train (v2 + reviewed autolabel + synth) + clean-clothing negatives with empty labels:
all reviewed hard negatives (images where yolo26n_v2 fired) + 200 random easy ones. The 200 held-out clean images
are never trained on and serve as a false-alarm test. valid/test stay the v2 ones."""
import random, sys
from pathlib import Path

R = Path(sys.argv[1]); NEG = R / "negatives"; OUT = R / "dataset_v4"
DROP = {285}  # review sheet numbers that are not clearly clean (distressed fringe skirt)
order = {int(n): name for n, name in (l.split("\t") for l in (NEG / "review/order.tsv").read_text().splitlines() if l)}
hard = {name for n, name in order.items() if n not in DROP}
easy = sorted(p.name for p in (NEG / "pool").iterdir() if p.name not in set(order.values()))
easy = random.Random(1).sample(easy, 200)
(OUT / "neg/images").mkdir(parents=True, exist_ok=True); (OUT / "neg/labels").mkdir(parents=True, exist_ok=True)
for name in sorted(hard) + easy:
    dst = OUT / "neg/images" / name
    if not dst.exists(): dst.symlink_to((NEG / "pool" / name).resolve())
    (OUT / "neg/labels" / (Path(name).stem + ".txt")).write_text("")
(OUT / "data.yaml").write_text(
    f"path: {R}\ntrain:\n  - {R}/dataset_v2/train/images\n  - {R}/dataset_v3/auto/images\n  - {R}/synth/images\n  - {OUT}/neg/images\n"
    f"val: {R}/dataset_v2/valid/images\ntest: {R}/dataset_v2/test/images\nnc: 2\nnames: ['stain', 'torn']\n")
print(f"hard negatives {len(hard)}, easy negatives {len(easy)}, held-out {len(list((NEG / 'heldout').iterdir()))}")
