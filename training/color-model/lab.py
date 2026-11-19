# K-Fashion(AI Hub) Validation 라벨 zip에서 옷마다 색상·서브색상·프린트·카테고리·정답 상자를 CSV로 뽑음.
# 실행: python lab.py <시작> <끝> <라벨링데이터.zip>  →  labels_<시작>.csv (합쳐서 labels.csv, 헤더
# img,style,w,h,cat,color,subcolor,print,kind,x,y,bw,bh)
import zipfile, json, csv, sys
start, end = int(sys.argv[1]), int(sys.argv[2])
z = zipfile.ZipFile(sys.argv[3])
names = sorted(n for n in z.namelist() if n.endswith('.json'))
with open(f'labels_{start}.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    for n in names[start:end]:
        root = json.loads(z.read(n).decode('utf-8'))
        img = root['이미지 정보']; d = root['데이터셋 정보']['데이터셋 상세설명']
        # zip 안 폴더 이름(스타일)은 cp949인데 zipfile이 cp437로 읽음
        style = n.split('/')[0].encode('cp437').decode('cp949', 'replace')
        for cat in ('상의', '하의', '아우터', '원피스'):
            labs = d['라벨링'].get(cat, []); rects = d['렉트좌표'].get(cat, [])
            for j, it in enumerate(labs):
                if not it or '색상' not in it: continue
                r = rects[j] if j < len(rects) else {}
                if not r or '가로' not in r: continue
                # 이미지 파일 이름은 라벨 JSON 이름과 같음('이미지 파일명' 값과 다를 때가 있음)
                w.writerow([n.split('/')[-1].replace('.json', '.jpg'), style, img['이미지 너비'], img['이미지 높이'], cat, it.get('색상'), it.get('서브색상', ''),
                            '|'.join(it.get('프린트', []) or []), it.get('카테고리', ''), r['X좌표'], r['Y좌표'], r['가로'], r['세로']])
print('ok', start, end, len(names))
