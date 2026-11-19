# K-Fashion 정답 상자로 옷 조각을 잘라 작게(긴 변 96px) 저장. 사진 전체는 저장하지 않음.
# 실행: python crop.py <원천데이터.zip> <시작 줄> <끝 줄> <출력 폴더>   (labels.csv는 lab.py 결과)
# 조각과 사진은 AI Hub 데이터라 레포·공유 폴더에 올리지 않습니다(재배포 금지).
import zipfile, csv, io, os, sys
from PIL import Image
src, start, end, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows = list(csv.DictReader(open(os.path.expanduser('~/kf/labels.csv'))))
z = zipfile.ZipFile(src)
byname = {os.path.basename(n): n for n in z.namelist() if not n.endswith('/')}
os.makedirs(out, exist_ok=True)
cur = None; im = None; done = 0; miss = 0
meta = open(os.path.join(out, f'meta_{start}.csv'), 'w', newline=''); w = csv.writer(meta)
for i in range(start, min(end, len(rows))):
    r = rows[i]
    dst = os.path.join(out, f'{i}.jpg')
    if r['img'] != cur:
        cur = r['img']; n = byname.get(cur)
        im = Image.open(io.BytesIO(z.read(n))).convert('RGB') if n else None
    if im is None: miss += 1; continue
    W, H = im.size
    x, y, bw, bh = (float(r[k]) for k in ('x', 'y', 'bw', 'bh'))
    box = (max(0, int(x)), max(0, int(y)), min(W, int(x + bw)), min(H, int(y + bh)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8: miss += 1; continue
    c = im.crop(box); c.thumbnail((96, 96)); c.save(dst, quality=92)
    w.writerow([i, r['img'], r['style'], r['cat'], r['color'], r['subcolor'], r['print'], r['kind']])
    done += 1
meta.close(); print('done', done, 'miss', miss)
