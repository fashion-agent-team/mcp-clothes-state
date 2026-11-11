"""Build the 8-class garment pattern dataset (Ultralytics classify layout) on the Jetson.
 - Fashionpedia (CC BY 4.0 labels): crops listed in pattern_crops.json (made by pattern_list.py), box + 8% padding.
 - Clothing Dataset for Second-Hand Fashion v3 front-only (CC BY 4.0): laid-flat whole-garment photos with a `pattern`
   field. Empty pattern = plain. 'Other', 'Embroidered', 'Lace', 'Transparent', 'Glitter', 'Metallic' are dropped
   (texture or unclear, not a print). 'Logo print' (often one small chest logo) and 'Animal print' (mostly pictures of
   animals, not leopard/zebra skin like Fashionpedia's 'animal') are dropped because they do not match our classes. Official test parquet -> separate laid test set, 5% of train -> valid.
Images are saved with the long side at 320 px.
usage: python -I pattern_build.py <crop json> <fashionpedia raw dir> <second-hand parquet dir> <out dir> [--no-sh]"""
import io, json, random, sys
from collections import Counter
from pathlib import Path
from PIL import Image

CJ, FP, SH, OUT = (Path(a) for a in sys.argv[1:5])
USE_SH = "--no-sh" not in sys.argv
NAMES = json.load(open(CJ))["names"]
SH_MAP = {None: "plain", "Striped": "stripe", "Checkered print": "check", "Floral print": "floral", "Dots": "dot",
          "Geometric print": "other"}
SH_PLAIN_CAP = 5000
LONG = 320
stats = Counter()


def save(im, split, cls, name):
    im = im.convert("RGB"); s = LONG / max(im.size)
    if s < 1: im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    d = OUT / split / cls; d.mkdir(parents=True, exist_ok=True)
    im.save(d / f"{name}.jpg", quality=90); stats[(split, cls)] += 1


# --- Fashionpedia crops
index = {p.name: p for p in FP.rglob("*.jpg")}
crops = json.load(open(CJ))
for split in ("train", "valid", "test"):
    tgt = {"valid": "val", "test": "test_fp"}.get(split, split)
    by_file = {}
    for r in crops[split]: by_file.setdefault(r["file"], []).append(r)
    for f, rs in by_file.items():
        if f not in index: stats[("missing", f)] += 1; continue
        im = Image.open(index[f])
        for r in rs:
            x, y, w, h = r["box"]; px, py = 0.08 * w, 0.08 * h
            box = (max(0, x - px), max(0, y - py), min(im.width, x + w + px), min(im.height, y + h + py))
            save(im.crop(tuple(round(v) for v in box)), tgt, NAMES[r["cls"]], f"fp{r['ann']}")

# --- Second-Hand laid photos
if USE_SH:
    import pyarrow.parquet as pq
    rng = random.Random(3)
    for pf in sorted(SH.glob("*.parquet")):
        is_test = pf.name.startswith("test")
        t = pq.read_table(pf, columns=["image", "pattern"]).to_pylist()
        for k, row in enumerate(t):
            cls = SH_MAP.get(row["pattern"], "drop")
            if cls == "drop": continue
            if is_test: split = "test_sh"
            else:
                if cls == "plain":
                    if stats[("sh_plain_seen", "")] >= SH_PLAIN_CAP: continue
                    stats[("sh_plain_seen", "")] += 1
                split = "val" if rng.random() < 0.05 else "train"
            b = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            save(Image.open(io.BytesIO(b)), split, cls, f"sh_{pf.stem}_{k}")

for split in ("train", "val", "test_fp", "test_sh"):
    print(split, {c: stats[(split, c)] for c in NAMES})
print("missing fashionpedia files:", sum(v for k, v in stats.items() if k[0] == "missing"))
