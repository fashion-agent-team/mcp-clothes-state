"""단추 어긋남 합성 데이터: Fashionpedia(CC BY 4.0) 단추 여밈(single breasted) 셔츠·가디건의 오른쪽(사진 기준) 절반을
단추 한 칸만큼(옷 높이 4~9%) 아래로 내려 밑단·깃 높이가 어긋나 보이게 만듭니다(shifted). 원본은 ok.
서버와 같은 자르기: 옷 박스를 사방 5% 넓힘.

python3 build_button_dataset.py ANN.json IMG_DIR OUT_DIR [max_images]
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

SHIRTS = {0, 3}  # shirt/blouse, cardigan (자켓은 열고 입는 일이 많아 뺌)
SINGLE_BREASTED = 225


def ann_mask(a, h, w):
    seg = a['segmentation']
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg['counts'], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def shift_half(img, m, box, rng, vertical=True):
    """옷 가운데 세로선 기준 한쪽 절반(옷 픽셀만)을 d만큼 아래로. vertical=False면 옆으로만 2~4% 밀어
    자른 자국은 같지만 높이는 맞는 사진(ok)을 만듦: 모델이 자른 자국이 아니라 높이 차이를 보게."""
    arr = np.array(img)
    x, y, w, h = box
    xs = np.where(m.any(axis=0))[0]
    cx = int((xs.min() + xs.max()) / 2 + w * rng.uniform(-0.04, 0.04))
    d = max(3, int(h * rng.uniform(0.04, 0.09)))
    dx = 0
    if not vertical:
        d = 0
        dx = max(2, int(w * rng.uniform(0.02, 0.04))) * rng.choice([-1, 1])
    out = arr.copy()
    side = np.zeros_like(m)
    if rng.random() < 0.5:
        side[:, cx:] = True
    else:
        side[:, :cx] = True
    region = m & side
    ys, xs2 = np.where(region)
    keep = (ys + d < arr.shape[0]) & (xs2 + dx >= 0) & (xs2 + dx < arr.shape[1])
    out[ys[keep] + d, xs2[keep] + dx] = arr[ys[keep], xs2[keep]]
    return Image.fromarray(out)


def crop(img, box):
    x, y, w, h = box
    b = (int(x - w * 0.05), int(y - h * 0.05), int(x + w * 1.05), int(y + h * 1.05))
    b = (max(0, b[0]), max(0, b[1]), min(img.width, b[2]), min(img.height, b[3]))
    return img.crop(b)


def main():
    ann_path, img_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    rng = random.Random(7)
    d = json.load(open(ann_path))
    imgs = {i['id']: i for i in d['images']}
    cands = [a for a in d['annotations'] if a['category_id'] in SHIRTS and SINGLE_BREASTED in a['attribute_ids']
             and a['bbox'][2] * a['bbox'][3] > 150 * 150 and a['bbox'][3] > a['bbox'][2] * 0.7]
    rng.shuffle(cands)
    seen = set()
    n = 0
    for a in cands:
        if n >= limit:
            break
        if a['image_id'] in seen:
            continue
        seen.add(a['image_id'])
        info = imgs[a['image_id']]
        path = os.path.join(img_dir, info['file_name'])
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert('RGB')
        try:
            m = ann_mask(a, img.height, img.width)
        except Exception:
            continue
        if m.sum() < 5000:
            continue
        split = 'val' if a['image_id'] % 10 == 0 else 'train'
        ok = img if rng.random() < 0.5 else shift_half(img, m, a['bbox'], rng, vertical=False)
        for cls, src in (('ok', ok), ('shifted', shift_half(img, m, a['bbox'], rng))):
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)
            crop(src, a['bbox']).save(os.path.join(out_dir, split, cls, f"{a['id']}.jpg"), quality=rng.randint(60, 95))
        n += 1
        if n % 500 == 0:
            print(n, flush=True)
    print('done', n)


if __name__ == '__main__':
    main()
