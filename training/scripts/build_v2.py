"""Build dataset_v2 = dataset (v1) + new stain/hole sets, deduped against v1 and each other."""
import hashlib, re, shutil, sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1]).expanduser()
V1, RAW, RAW2, OUT = R / "dataset", R / "raw", R / "raw2", R / "dataset_v2"
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
stats = Counter()

def stem(name):
    s = re.sub(r"\.rf\..*$", "", name)
    while True:
        t = re.sub(r"_(jpe?g|png)$", "", s, flags=re.I)
        if t == s:
            return s.lower()
        s = t

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def images(d, split):
    p = d / split / "images"
    return sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT) if p.exists() else []

def rows(img):
    lab = img.parent.parent / "labels" / (img.stem + ".txt")
    return [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []

def to_box(r):
    v = list(map(float, r[1:]))
    if len(v) > 4:  # polygon -> bbox
        xs, ys = v[0::2], v[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return v

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V1, OUT)
(OUT / "data.yaml").write_text(
    f"path: {OUT}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ['stain', 'torn']\n")

# everything already in v1 (all raw images, any split, incl. ones v1 skipped) counts as seen
seen_stems, seen_md5 = set(), set()
for d in (RAW / "stain", RAW / "more"):
    for s in SPLITS:
        for img in images(d, s):
            seen_stems.add(stem(img.name)); seen_md5.add(md5(img))

def add(key, prefix, cls_map, split_map, drop_name=None):
    d = RAW2 / key
    batch_stems = set()
    for s in SPLITS:
        for img in images(d, s):
            st, h = stem(img.name), md5(img)
            if h in seen_md5:
                stats[f"{key}/drop_dup_file"] += 1; continue
            if st in seen_stems:
                stats[f"{key}/drop_seen_name"] += 1; continue
            if drop_name and drop_name(img.name):
                stats[f"{key}/drop_nonclothing"] += 1; continue
            out_rows = []
            for r in rows(img):
                c = cls_map.get(r[0])
                if c is None:
                    stats[f"{key}/drop_box_cls{r[0]}"] += 1; continue
                out_rows.append(f"{c} " + " ".join(f"{v:.6f}" for v in to_box(r)))
            if not out_rows:
                stats[f"{key}/drop_no_boxes"] += 1; continue
            seen_md5.add(h); batch_stems.add(st)
            ds = split_map[s]
            shutil.copy2(img, OUT / ds / "images" / f"{prefix}_{img.name}")
            (OUT / ds / "labels" / f"{prefix}_{img.stem}.txt").write_text("\n".join(out_rows) + "\n")
            stats[f"{ds}/{key}_imgs"] += 1
            stats[f"{ds}/{key}_boxes"] += len(out_rows)
    # augmented copies share a stem, so only mark stems seen after the whole set is added
    seen_stems.update(batch_stems)

ALL_TRAIN = {s: "train" for s in SPLITS}
SAME = {s: s for s in SPLITS}
add("stain_v2i", "stainB", {"0": 0}, ALL_TRAIN)                 # 'sss' (1) dropped
add("stain_v1i", "stainC", {"0": 0}, SAME)
add("hole_v1i", "holeB", {"0": 1}, SAME)                          # added before hole_h so its overlap is removed from hole_h
add("hole_h", "holeA", {"0": 1}, ALL_TRAIN, drop_name=lambda n: "car-cover" in n)

for k in sorted(stats):
    print(k, stats[k])
tot = Counter()
for s in SPLITS:
    for f in (OUT / s / "labels").glob("*.txt"):
        cs = {l.split()[0] for l in f.read_text().splitlines() if l.strip()}
        tot[(s, "imgs")] += 1
        if not cs: tot[(s, "negative")] += 1
        for c in cs: tot[(s, "imgs_with_" + ("stain" if c == "0" else "torn"))] += 1
        for l in f.read_text().splitlines():
            if l.strip(): tot[(s, "boxes_" + ("stain" if l.split()[0] == "0" else "torn"))] += 1
print("TOTAL")
for s in SPLITS:
    print(s, {k[1]: v for k, v in sorted(tot.items()) if k[0] == s})
