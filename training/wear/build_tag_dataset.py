"""택(옷 라벨) 합성 데이터 만들기: Fashionpedia(CC BY 4.0) 사람이 입은 상의의 목 부분을 잘라
반은 그대로(none), 반은 합성 택을 붙여(tag) 분류 데이터셋을 만듭니다.

서버에서 쓸 때와 같은 자르기: 상의 박스에서 가로 가운데 60%, 세로 위 -10%~+30%.
택 위치: 목둘레(neckline) 박스 위쪽 가운데(뒷목 위로 삐져나옴) 또는 아래쪽 가운데(앞뒤 바꿔 입어 앞 목에 보임).

python3 build_tag_dataset.py ANN.json IMG_DIR OUT_DIR [max_images]
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

TOPS = {0, 1, 2, 3, 4, 5, 9, 10}  # shirt, top/t-shirt, sweater, cardigan, jacket, vest, coat, dress
NECKLINE, COLLAR, HOOD = 33, 28, 27
# 택이 목 가운데에 오는 목둘레(크루·라운드·오벌·U·보트·스쿱·하이·터틀). 깊은 V넥 등은 택이 살 위에 붙어 어색해서 none만.
TAG_NECKS = {181, 182, 185, 186, 189, 190, 197, 198}
FONTS = [os.path.join(dp, f) for dp, _, fs in os.walk('/usr/share/fonts') for f in fs
         if f.endswith('.ttf') and ('Sans' in f or 'Serif' in f or 'Liberation' in f)]
WORDS = ['M', 'L', 'S', 'XL', '95', '100', '105', 'FREE', 'COTTON 100%', 'MADE IN KOREA', 'MADE IN VIETNAM',
         'UNIQ', 'ZARO', 'SPAO', 'H&N', 'BASIC', 'ESSENTIAL', 'STUDIO', 'NORTH', 'SIZE', 'EU 38', 'US 6']


def tag_sprite(w, h, rng):
    """택 한 장(RGBA). 흰색이 가장 많고 검정·베이지·회색·색 택도."""
    palette = [((245, 245, 242), (20, 20, 20))] * 5 + [((25, 25, 25), (235, 235, 235))] * 2 + \
        [((225, 210, 185), (60, 40, 30)), ((170, 170, 170), (20, 20, 20)), ((200, 30, 40), (250, 250, 250)),
         ((30, 50, 120), (240, 240, 240))]
    bg, fg = rng.choice(palette)
    bg = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in bg)
    scale = 4
    W, H = max(8, w * scale), max(8, h * scale)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = rng.randint(0, min(W, H) // 6)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=bg + (255,))
    # 글자 몇 줄
    lines = rng.randint(1, 4)
    y = H * 0.08
    for _ in range(lines):
        word = rng.choice(WORDS)
        size = max(6, int(H / (lines + 1.5) * rng.uniform(0.5, 1.0)))
        try:
            font = ImageFont.truetype(rng.choice(FONTS), size)
        except Exception:
            font = ImageFont.load_default()
        tw = d.textlength(word, font=font)
        if tw > W * 0.9:
            size = max(6, int(size * W * 0.9 / tw))
            try:
                font = ImageFont.truetype(rng.choice(FONTS), size)
            except Exception:
                font = ImageFont.load_default()
            tw = d.textlength(word, font=font)
        d.text(((W - tw) / 2, y), word, fill=fg + (255,), font=font)
        y += size * 1.25
        if y > H * 0.9:
            break
    if rng.random() < 0.4:  # 세탁 기호 줄
        for i in range(rng.randint(2, 5)):
            x = W * (0.1 + i * 0.18)
            d.rectangle([x, H * 0.82, x + W * 0.1, H * 0.92], outline=fg + (255,), width=max(1, scale))
    if rng.random() < 0.5:  # 꿰맨 선
        d.line([(0, 2 * scale), (W, 2 * scale)], fill=tuple(int(c * 0.8) for c in bg) + (255,), width=scale)
    im = im.resize((w, h), Image.LANCZOS)
    # 접힘·음영: 한쪽으로 어두워지게
    shade = Image.linear_gradient('L').resize((w, h)).rotate(rng.choice([0, 90, 180, 270]))
    dark = ImageEnhance.Brightness(im.convert('RGB')).enhance(rng.uniform(0.6, 0.9))
    im = Image.composite(dark.convert('RGBA'), im, shade.point(lambda v: int(v * rng.uniform(0.2, 0.8))))
    im.putalpha(im.getchannel('A'))
    return im


def perspective(im, rng):
    """살짝 기울이고 사다리꼴로(접히거나 들린 택)."""
    w, h = im.size
    dx = w * rng.uniform(0, 0.25)
    dy = h * rng.uniform(0, 0.15)
    coeffs = find_coeffs([(0, 0), (w, 0), (w, h), (0, h)],
                         [(rng.uniform(0, dx), rng.uniform(0, dy)), (w - rng.uniform(0, dx), rng.uniform(0, dy)),
                          (w, h), (0, h)])
    im = im.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    return im.rotate(rng.uniform(-30, 30), resample=Image.BICUBIC, expand=True)


def find_coeffs(pa, pb):
    import numpy as np
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(pb, dtype=float).reshape(8)
    return np.linalg.solve(A, B).tolist()


def paste_tag(img, neck, rng):
    """neck = (x, y, w, h) 목둘레 박스. 위쪽 가운데(뒤 택) 또는 아래쪽 가운데(앞 목 택)에 붙임."""
    nx, ny, nw, nh = neck
    base = max(nw, 20)
    tw = int(base * rng.uniform(0.10, 0.26))
    th = int(tw * rng.uniform(0.6, 1.6))
    sprite = perspective(tag_sprite(max(tw, 6), max(th, 6), rng), rng)
    cx = nx + nw / 2 + nw * rng.uniform(-0.12, 0.12)
    sw, sh = sprite.size
    if rng.random() < 0.5:
        cy = ny + nh * rng.uniform(-0.1, 0.3)  # 뒷목 위로 삐져나옴
    else:
        cy = ny + nh - sh * rng.uniform(0.15, 0.45)  # 앞 목 가장자리 위로 보임(앞뒤 바꿔 입음)
    pos = (int(cx - sw / 2), int(cy - sh / 2))
    # 주변 밝기에 맞추기
    region = img.crop((max(0, pos[0]), max(0, pos[1]), min(img.width, pos[0] + sw), min(img.height, pos[1] + sh)))
    if region.width > 0 and region.height > 0:
        mean = sum(region.convert('L').get_flattened_data()) / (region.width * region.height)
        factor = 0.75 + 0.35 * mean / 255
        rgb = ImageEnhance.Brightness(sprite.convert('RGB')).enhance(factor)
        rgb.putalpha(sprite.getchannel('A'))
        sprite = rgb
    if rng.random() < 0.6:  # 그림자
        shadow = Image.new('RGBA', sprite.size, (0, 0, 0, 0))
        shadow.putalpha(sprite.getchannel('A').point(lambda v: int(v * 0.35)))
        shadow = shadow.filter(ImageFilter.GaussianBlur(2))
        img.paste(shadow, (pos[0] + 2, pos[1] + 2), shadow)
    sprite = sprite.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))
    img.paste(sprite, pos, sprite)
    return img


def neck_crop(img, top):
    """서버와 같은 자르기: 상의 박스 가로 가운데 60%, 세로 -10%~+30%."""
    x, y, w, h = top
    box = (int(x + w * 0.2), int(y - h * 0.1), int(x + w * 0.8), int(y + h * 0.3))
    box = (max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3]))
    return img.crop(box), box


def main():
    ann_path, img_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    rng = random.Random(13)
    d = json.load(open(ann_path))
    by_img = defaultdict(list)
    for a in d['annotations']:
        by_img[a['image_id']].append(a)
    files = {i['id']: i['file_name'] for i in d['images']}
    ids = sorted(by_img)
    rng.shuffle(ids)
    n = 0
    for iid in ids:
        if n >= limit:
            break
        anns = by_img[iid]
        tops = [a for a in anns if a['category_id'] in TOPS and a['bbox'][2] * a['bbox'][3] > 120 * 120]
        necks = [a for a in anns if a['category_id'] == NECKLINE]
        if not tops or not necks:
            continue
        top = max(tops, key=lambda a: a['bbox'][2] * a['bbox'][3])
        tx, ty, tw, th = top['bbox']
        neck = None
        for a in necks:
            nx, ny, nw, nh = a['bbox']
            if tx <= nx + nw / 2 <= tx + tw and ty - th * 0.1 <= ny + nh / 2 <= ty + th * 0.3:
                neck = a['bbox']
                break
        if neck is None:
            continue
        path = os.path.join(img_dir, files[iid])
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert('RGB')
        split = 'val' if (iid % 10) == 0 else 'train'
        # 같은 사진에서 none 하나, tag 하나 (모델이 사진이 아니라 택을 보게)
        crop, _ = neck_crop(img, top['bbox'])
        if crop.width < 48 or crop.height < 32:
            continue
        can_tag = any(
            TAG_NECKS & set(a.get('attribute_ids', [])) for a in necks + [top])
        for cls in (('none', 'tag') if can_tag else ('none',)):
            src = img.copy()
            if cls == 'tag':
                src = paste_tag(src, neck, rng)
            c, _ = neck_crop(src, top['bbox'])
            q = rng.randint(55, 95)
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)
            c.save(os.path.join(out_dir, split, cls, f'{iid}.jpg'), quality=q)
        n += 1
        if n % 500 == 0:
            print(n, flush=True)
    print('done', n)


if __name__ == '__main__':
    main()
