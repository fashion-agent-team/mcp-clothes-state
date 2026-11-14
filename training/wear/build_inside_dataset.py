"""뒤집어 입음 합성 데이터: Fashionpedia(CC BY 4.0) 티셔츠·니트·셔츠·원피스를 안팎 뒤집은 것처럼 바꿉니다(inside_out). 원본은 ok.
뒤집어 입으면 보이는 것을 흉내 냅니다.
- 솔기 시접: 소매가 붙는 선(소매 마스크와 몸판이 닿는 곳)과 옆선·어깨선 안쪽에 도드라진 띠 + 오버록 바늘땀
- 프린트: 좌우가 뒤집히고 흐리게 비침
서버와 같은 자르기: 옷 박스를 사방 5% 넓힘.

python3 build_inside_dataset.py ANN.json IMG_DIR OUT_DIR [max_images]
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_button_dataset import ann_mask, crop  # noqa: E402

BODIES = {0, 1, 2, 10}  # shirt, t-shirt, sweater, dress
SLEEVE = 31


def seam_band(mask, inner, width):
    """mask 안쪽에서 경계로부터 inner~inner+width 픽셀 거리의 띠."""
    dist = ndimage.distance_transform_edt(mask)
    return (dist > inner) & (dist <= inner + width)


def inside_out(img, body, sleeves, rng):
    arr = np.array(img).astype(np.float32)
    h, w = body.shape
    ys, xs = np.where(body)
    bw = xs.max() - xs.min()
    out = arr.copy()
    # 1) 프린트 좌우 뒤집고 흐리게: 옷 안쪽만
    cx = (xs.min() + xs.max()) / 2
    flip = arr[:, ::-1]
    shift = int(round(2 * cx - (w - 1)))
    flipped = np.roll(flip, shift, axis=1)
    fbody = np.roll(body[:, ::-1], shift, axis=1)
    fmask = fbody & body
    mean = arr[body].mean(axis=0)
    fade = rng.uniform(0.25, 0.6)
    # 좌우 모양이 비슷한 옷만 뒤집음(자세가 기울면 몸 모양이 어긋나 가짜 티가 남). 아니면 흐리게만.
    sym = fmask.sum() / max(1, (fbody | body).sum())
    src = flipped if sym > 0.88 else arr
    faded = mean + (src - mean) * fade
    inner_body = ndimage.binary_erosion(fmask if sym > 0.88 else body, iterations=3)
    out[inner_body] = faded[inner_body]
    # 2) 솔기 시접 띠: 옆선·어깨선(옷 바깥 경계 안쪽) + 소매가 붙는 선
    width = max(2, int(bw * rng.uniform(0.008, 0.018)))
    inner = max(1, int(bw * rng.uniform(0.0, 0.02)))
    band = seam_band(body, inner, width)
    # 밑단·목둘레보다 옆선·어깨를 남김: 위·아래 10%는 일부만
    top, bottom = ys.min(), ys.max()
    rows = np.arange(h)[:, None]
    band &= ~((rows > bottom - (bottom - top) * 0.06) & (rng.random() < 0.6))
    if sleeves is not None:
        dil = ndimage.binary_dilation(sleeves, iterations=width)
        band |= dil & body & ~ndimage.binary_erosion(dil, iterations=max(1, width // 2))
    tone = rng.choice([0.88, 0.92, 1.08, 1.12])
    out[band] = np.clip(out[band] * tone + rng.uniform(-8, 8), 0, 255)
    # 오버록 바늘땀: 띠 가장자리 점선
    edge = band & ~ndimage.binary_erosion(band)
    yy, xx = np.where(edge)
    keep = ((yy + xx) // max(2, width // 2)) % 2 == 0
    stitch = np.clip(mean * rng.choice([0.75, 1.25]), 0, 255)
    a = rng.uniform(0.3, 0.7)
    out[yy[keep], xx[keep]] = out[yy[keep], xx[keep]] * (1 - a) + stitch * a
    # 띠만 살짝 흐리게(천이 접혀 생긴 도톰함)
    blurred = ndimage.gaussian_filter(out, sigma=(1, 1, 0))
    out[band] = blurred[band]
    out = ndimage.uniform_filter(out, size=(1, 1, 1))
    return Image.fromarray(out.astype(np.uint8))


def main():
    ann_path, img_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    rng = random.Random(21)
    d = json.load(open(ann_path))
    imgs = {i['id']: i for i in d['images']}
    by_img = defaultdict(list)
    for a in d['annotations']:
        by_img[a['image_id']].append(a)
    ids = sorted(by_img)
    rng.shuffle(ids)
    n = 0
    for iid in ids:
        if n >= limit:
            break
        anns = by_img[iid]
        bodies = [a for a in anns if a['category_id'] in BODIES and a['bbox'][2] * a['bbox'][3] > 150 * 150]
        if not bodies:
            continue
        a = max(bodies, key=lambda a: a['bbox'][2] * a['bbox'][3])
        info = imgs[iid]
        path = os.path.join(img_dir, info['file_name'])
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert('RGB')
        try:
            body = ann_mask(a, img.height, img.width)
            sl = [ann_mask(s, img.height, img.width) for s in anns if s['category_id'] == SLEEVE]
        except Exception:
            continue
        if body.sum() < 8000:
            continue
        sleeves = np.any(sl, axis=0) if sl else None
        split = 'val' if iid % 10 == 0 else 'train'
        for cls, src in (('ok', img), ('inside_out', inside_out(img, body, sleeves, rng))):
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)
            crop(src, a['bbox']).save(os.path.join(out_dir, split, cls, f"{a['id']}.jpg"), quality=rng.randint(60, 95))
        n += 1
        if n % 500 == 0:
            print(n, flush=True)
    print('done', n)


if __name__ == '__main__':
    main()
