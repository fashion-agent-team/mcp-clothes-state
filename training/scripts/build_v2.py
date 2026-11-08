"""2차(v2) 데이터 만들기: v1 + 새 얼룩·구멍 데이터. v1과도, 서로 간에도 겹치는 사진을 뺀다.

사용: python build_v2.py <프로젝트 루트>   (merge_dataset.py 를 먼저 돌려 <루트>/dataset 이 있어야 함)
  <루트>/raw2/stain_v2i = stain.v2i  (s-workspace-okn3h/stain-uq7kn)
  <루트>/raw2/stain_v1i = stain.v1i  (devadharshini/stain-sdji5)
  <루트>/raw2/hole_v1i  = Hole.v1i   (s-workspace-wsqvz/hole-950bj-lmgem)
  <루트>/raw2/hole_h    = hole.v1-h  (swee/hole-inskr)
결과: <루트>/dataset_v2
"""
import hashlib, re, shutil, sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1]).expanduser()
V1, RAW, RAW2, OUT = R / "dataset", R / "raw", R / "raw2", R / "dataset_v2"
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
stats = Counter()

def stem(name):
    """Roboflow 파일 이름에서 원본 이름만 남긴다(증강본은 같은 원본 이름을 공유)."""
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
    if len(v) > 4:  # 폴리곤 -> 박스
        xs, ys = v[0::2], v[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return v

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V1, OUT)
(OUT / "data.yaml").write_text(
    f"path: {OUT}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ['stain', 'torn']\n")

# v1 원본(어느 split이든, v1에서 뺀 사진 포함)은 모두 "이미 본 사진"으로 친다.
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
    # 증강본은 원본 이름을 공유하므로, 한 데이터를 다 넣은 뒤에야 "본 이름"에 더한다.
    seen_stems.update(batch_stems)

ALL_TRAIN = {s: "train" for s in SPLITS}   # 증강본이 섞인 데이터는 학습에만
SAME = {s: s for s in SPLITS}
add("stain_v2i", "stainB", {"0": 0}, ALL_TRAIN)                 # 'sss'(1)는 같은 얼룩 중복 박스라 버림
add("stain_v1i", "stainC", {"0": 0}, SAME)
add("hole_v1i", "holeB", {"0": 1}, SAME)                          # hole_h보다 먼저 넣어 겹치는 사진을 hole_h에서 뺌
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
