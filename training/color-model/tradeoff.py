import csv, numpy as np, sys
CLASSES = ['화이트','블랙','베이지','핑크','그레이','스카이블루','네이비','브라운','블루','카키','레드','그린','라벤더','실버','옐로우','와인','민트','퍼플','오렌지','골드']
MAP = {'화이트':'white','블랙':'black','베이지':'beige','핑크':'pink','그레이':'gray','스카이블루':'blue','네이비':'navy',
 '브라운':'brown','블루':'blue','카키':'khaki','레드':'red','그린':'green','라벤더':'purple','실버':'gray','옐로우':'yellow',
 '와인':'red','민트':'green','퍼플':'purple','오렌지':'orange','골드':'yellow'}
NEUTRAL = {'white','black','gray','navy','beige','brown','khaki'}
# 색 조합에서 "강한 색"으로 보는 클래스(연한 색·와인은 차분한 색이라 뺌)
STRONG = [i for i, c in enumerate(CLASSES) if MAP[c] not in NEUTRAL and c not in ('스카이블루','라벤더','민트','와인')]
NEU = [i for i, c in enumerate(CLASSES) if MAP[c] in NEUTRAL]
rows = list(csv.reader(open('crops/meta.csv')))
P = np.load(sys.argv[1]); te = np.load(sys.argv[2])[:, 0].astype(int)
gt = np.array([CLASSES.index(rows[i][4]) for i in te])
gt_strong = np.isin(gt, STRONG); gt_fam = np.array([MAP[CLASSES[g]] for g in gt])
for tau in [1.1, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    pred = P.argmax(1).copy()
    pneu = P[:, NEU].sum(1)
    sw = np.isin(pred, STRONG) & (pneu >= tau)
    pred[sw] = np.array(NEU)[P[sw][:, NEU].argmax(1)]
    fam = np.array([MAP[CLASSES[p]] for p in pred])
    ps = np.isin(pred, STRONG)
    print(f'τ={tau}: family {np.mean(fam==gt_fam):.3f}  강한 색 아닌데 강한 색으로 {np.mean(ps[~gt_strong]):.3f}  강한 색 놓침 {np.mean(~ps[gt_strong]):.3f}')
