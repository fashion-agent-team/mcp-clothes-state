#!/usr/bin/env python3
"""학습·평가 목록(jsonl) 만들기. 사용:
  python3 make_lists.py fashionpedia ANN.json IMG_DIR SPLIT OUT.jsonl [CAP]   (입은 옷, CC BY 4.0)
  python3 make_lists.py cc0 CLOTHING_DATASET_DIR OUT.jsonl                     (내려놓은 옷, CC0)
  python3 make_lists.py walls OUT_DIR OUT.jsonl N                              (만든 빈 벽·바닥)
  python3 make_lists.py df2 CONSUMER_DIR OUT.jsonl                             (DeepFashion2 소비자 사진, 평가만)
"""
import csv, json, os, random, sys

random.seed(7)
FP = {"shirt, blouse": "shirt", "top, t-shirt, sweatshirt": "tshirt", "sweater": "sweater", "cardigan": "cardigan",
      "jacket": "jacket", "vest": "vest", "pants": "pants", "shorts": "shorts", "skirt": "skirt", "coat": "coat",
      "dress": "dress", "shoe": "shoes"}
CC0 = {"T-Shirt": "tshirt", "Shoes": "shoes", "Shorts": "shorts", "Shirt": "shirt", "Pants": "pants", "Skirt": "skirt",
       "Dress": "dress", "Polo": "tshirt", "Blouse": "shirt", "Hoodie": "sweater", "Blazer": "jacket", "Top": "tshirt",
       "Undershirt": "tshirt", "Longsleeve": "longsleeve", "Outwear": "outer"}


def fashionpedia(ann, img_dir, split, out, cap=1500):
    d = json.load(open(ann))
    cats = {c["id"]: FP.get(c["name"]) for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    by = {}
    garments = {}
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        garments.setdefault(a["image_id"], []).append((x, y, w, h))
        lab = cats.get(a["category_id"])
        if not lab or a.get("iscrowd"):
            continue
        min_side = 0.06 if lab == "shoes" else 0.12
        if w < im["width"] * min_side or h < im["height"] * min_side:
            continue
        by.setdefault(lab, []).append({"img": os.path.join(img_dir, im["file_name"]),
                                       "box": [x / im["width"], y / im["height"], w / im["width"], h / im["height"]],
                                       "label": lab, "split": split, "src": "fashionpedia"})
    rows = []
    for lab, lst in by.items():
        random.shuffle(lst)
        rows += lst[:int(cap)]
        print(lab, len(lst), "→", min(len(lst), int(cap)))
    # 옷이 거의 안 걸친 배경 조각 = none
    ids = list(garments)
    random.shuffle(ids)
    n_none = 0
    for iid in ids:
        if n_none >= int(cap):
            break
        im = imgs[iid]
        W, H = im["width"], im["height"]
        for _ in range(10):
            w = random.uniform(0.2, 0.45) * W
            h = random.uniform(0.2, 0.45) * H
            x = random.uniform(0, W - w)
            y = random.uniform(0, H - h)
            inter = 0
            for gx, gy, gw, gh in garments[iid]:
                ix = max(0, min(x + w, gx + gw) - max(x, gx))
                iy = max(0, min(y + h, gy + gh) - max(y, gy))
                inter += ix * iy
            if inter < 0.05 * w * h:
                rows.append({"img": os.path.join(img_dir, im["file_name"]), "box": [x / W, y / H, w / W, h / H],
                             "label": "none", "split": split, "src": "fashionpedia-bg"})
                n_none += 1
                break
    print("none", n_none)
    write(rows, out)


def cc0(root, out):
    rows = []
    with open(os.path.join(root, "images.csv")) as f:
        for r in csv.DictReader(f):
            lab = CC0.get(r["label"])
            p = os.path.join(root, "images", r["image"] + ".jpg")
            if lab and os.path.exists(p):
                split = "test" if random.random() < 0.2 else "train"
                rows.append({"img": p, "box": [0.02, 0.02, 0.96, 0.96], "label": lab, "split": split, "src": "cc0"})
    write(rows, out)


def walls(out_dir, out, n):
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i in range(int(n)):
        base = np.array([random.randint(60, 250)] * 3) + np.random.randint(-25, 25, 3)
        a = np.ones((320, 320, 3)) * base
        gx = np.linspace(-1, 1, 320)[None, :, None] * random.uniform(-40, 40)
        gy = np.linspace(-1, 1, 320)[:, None, None] * random.uniform(-40, 40)
        a = a + gx + gy + np.random.normal(0, random.uniform(1, 9), a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
        if random.random() < 0.4:  # 바닥 나뭇결·타일 줄
            d = ImageDraw.Draw(im)
            step = random.randint(30, 90)
            c = tuple(int(v) for v in np.clip(base - random.randint(10, 40), 0, 255))
            for k in range(0, 320, step):
                if random.random() < 0.5:
                    d.line([(0, k), (320, k + random.randint(-20, 20))], fill=c, width=random.randint(1, 3))
                else:
                    d.line([(k, 0), (k + random.randint(-20, 20), 320)], fill=c, width=random.randint(1, 3))
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.5)))
        p = os.path.join(out_dir, f"wall{i:04d}.jpg")
        im.save(p, quality=85)
        rows.append({"img": p, "box": [0.05, 0.05, 0.9, 0.9], "label": "none",
                     "split": "test" if i % 5 == 0 else "train", "src": "walls"})
    write(rows, out)


def df2(root, out):
    rows = []
    with open(os.path.join(root, "index.csv")) as f:
        r0 = list(csv.DictReader(f))
    print("columns:", list(r0[0].keys()))
    for r in r0:
        rows.append({"img": os.path.join(root, r.get("path") or r.get("file") or r.get("image")),
                     "box": [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])] if "x" in r else [0, 0, 1, 1],
                     "label": r.get("category_name") or r.get("category") or r.get("label"), "split": "test",
                     "src": "df2"})
    write(rows, out)


def write(rows, out):
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("썼어요", out, len(rows))


if __name__ == "__main__":
    k = sys.argv[1]
    {"fashionpedia": fashionpedia, "cc0": cc0, "walls": walls, "df2": df2}[k](*sys.argv[2:])
