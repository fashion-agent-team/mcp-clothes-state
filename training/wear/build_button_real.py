"""단추 6차: 5차에 같은 셔츠를 바르게 잠근 사진 5장(ok6~10, 11-16 사용자)을 더함. 5차 설명:
단추 5차: 사용자가 찍은 실제 사진(어긋남 4장: 줄무늬 셔츠 한 벌, 정상 1장)을 서버와 같은 박스(fashion12 옷 박스)로
자르되 박스를 흔들고(±6%), 좌우 뒤집고, 밝기·색을 바꿔 장마다 80장으로 늘림. 합성 3차 세트(ds_btn3)에서 ok·shifted 4000장씩 섞음.
다른 셔츠 실제 사진 2장(웹·거울)은 시험용으로만."""
import os, random, shutil, glob, sys
from PIL import Image, ImageEnhance, ImageOps
from build_button_dataset import crop
rng = random.Random(11)
R = sys.argv[1] if len(sys.argv) > 1 else 'real/'  # 사용자 사진은 레포에 안 넣음(개인 사진). R/에 1~5.jpg, R/../shirt_ok/ok6~10.jpg
real = {'shifted': [('1.jpg', (0.01, 0.00, 0.99, 0.82)), ('2.jpg', (0.10, 0.00, 0.85, 0.84)),
                    ('3.jpg', (0.01, 0.00, 0.99, 0.85)), ('5.jpg', (0.03, 0.02, 0.96, 0.79))],
        'ok': [('4.jpg', (0.03, 0.00, 0.95, 0.85)), ('../shirt_ok/ok6.jpg', (0.18, 0.00, 0.82, 0.90)),
               ('../shirt_ok/ok7.jpg', (0.05, 0.03, 0.95, 0.89)), ('../shirt_ok/ok8.jpg', (0.08, 0.01, 0.92, 0.93)),
               ('../shirt_ok/ok9.jpg', (0.21, 0.05, 0.79, 0.89)), ('../shirt_ok/ok10.jpg', (0.35, 0.00, 0.42, 0.76))]}
OUT = 'ds_btn6'
shutil.rmtree(OUT, ignore_errors=True)
for split in ('train', 'val'):
    for cls in ('ok', 'shifted'):
        os.makedirs(f'{OUT}/{split}/{cls}')
        src = sorted(glob.glob(f'ds_btn3/{split}/{cls}/*.jpg'))
        rng.shuffle(src)
        for f in src[:4000 if split == 'train' else 600]:
            os.symlink(os.path.abspath(f), f'{OUT}/{split}/{cls}/{os.path.basename(f)}')
for cls, items in real.items():
    n = 80 if cls == 'shifted' else 60  # 어긋남 320장, 정상 360장
    for name, (x, y, w, h) in items:
        img = Image.open(R + name).convert('RGB')
        W, H = img.size
        for k in range(n):
            j = lambda: rng.uniform(-0.06, 0.06)
            bx, by = x + j() * w, y + j() * h
            bw, bh = w * (1 + j()), h * (1 + j())
            c = crop(img, (bx * W, by * H, bw * W, bh * H))
            if rng.random() < 0.5:
                c = ImageOps.mirror(c)
            c = ImageEnhance.Brightness(c).enhance(rng.uniform(0.7, 1.3))
            c = ImageEnhance.Color(c).enhance(rng.uniform(0.6, 1.4))
            c = c.rotate(rng.uniform(-6, 6), fillcolor=(114, 114, 114))
            c.save(f'{OUT}/train/{cls}/real_{os.path.basename(name)[:-4]}_{k}.jpg', quality=rng.randint(70, 95))
print('done')
