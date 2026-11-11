"""Fashionpedia (CC BY 4.0 annotations) textile-pattern attributes -> garment crop list for an 8-class pattern classifier.
Only garments (category 0-11) with exactly one pattern attribute are kept; herringbone is dropped (reads as plain).
Splits: Fashionpedia val -> valid; fashion12's 2,000 held-out train images (seed 0) + 3,000 more (seed 1) -> test.
Plain is subsampled in train (seed 2) so it does not swamp the rare patterns.
usage: python pattern_list.py <dir with instances_attributes_*.json> <out.json>"""
import json, random, sys
from collections import Counter
from pathlib import Path

NAMES = ["plain", "stripe", "check", "floral", "dot", "graphic", "animal", "other"]
GROUP = {"plain (pattern)": 0, "stripe": 1, "check": 2, "houndstooth (pattern)": 2, "argyle": 2,
         "floral": 3, "plant": 3, "dot": 4, "letters, numbers": 5, "cartoon": 5,
         "leopard": 6, "cheetah": 6, "snakeskin (pattern)": 6, "zebra": 6, "giraffe": 6, "peacock": 6,
         "abstract": 7, "geometric": 7, "fair isle": 7, "paisley": 7, "chevron": 7, "camouflage": 7, "toile de jouy": 7}
GARMENTS = set(range(12))
MIN_SIDE, MIN_AREA = 48, 0.01
PLAIN_TRAIN = 9000

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def crops(ann_file):
    d = json.load(open(ann_file))
    A = {a["id"]: a for a in d["attributes"]}
    imgs = {im["id"]: im for im in d["images"]}
    rows = {}
    for a in d["annotations"]:
        if a["category_id"] not in GARMENTS or a.get("iscrowd"):
            continue
        p = [A[x]["name"] for x in a["attribute_ids"] if A[x]["supercategory"] in ("textile pattern", "animal")]
        if len(p) != 1 or p[0] not in GROUP:
            continue
        im = imgs[a["image_id"]]; x, y, w, h = a["bbox"]
        if min(w, h) < MIN_SIDE or w * h < MIN_AREA * im["width"] * im["height"]:
            continue
        rows.setdefault(a["image_id"], []).append({"file": im["file_name"], "box": [round(v, 1) for v in a["bbox"]],
                                                   "cls": GROUP[p[0]], "cat": a["category_id"], "ann": a["id"]})
    return sorted(imgs), rows


ids, tr = crops(SRC / "instances_attributes_train2020.json")
test12 = set(random.Random(0).sample(ids, 2000))            # same draw as fashionpedia_to_yolo.py
rest = [i for i in ids if i not in test12]
extra = set(random.Random(1).sample(rest, 3000))
test_ids = test12 | extra
out = {"names": NAMES, "train": [], "valid": [], "test": []}
for i, rs in tr.items():
    out["test" if i in test_ids else "train"] += rs
plain = [r for r in out["train"] if r["cls"] == 0]
keep = set(id(r) for r in random.Random(2).sample(plain, min(PLAIN_TRAIN, len(plain))))
out["train"] = [r for r in out["train"] if r["cls"] != 0 or id(r) in keep]
_, va = crops(SRC / "instances_attributes_val2020.json")
out["valid"] = [r for rs in va.values() for r in rs]
OUT.write_text(json.dumps(out))
for s in ("train", "valid", "test"):
    c = Counter(r["cls"] for r in out[s]); print(s, len(out[s]), {NAMES[k]: c[k] for k in range(len(NAMES))})
