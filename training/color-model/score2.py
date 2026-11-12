import csv, sys, numpy as np, collections
CLASSES = ['화이트','블랙','베이지','핑크','그레이','스카이블루','네이비','브라운','블루','카키','레드','그린','라벤더','실버','옐로우','와인','민트','퍼플','오렌지','골드']
MAP = {'화이트':'white','블랙':'black','베이지':'beige','핑크':'pink','그레이':'gray','스카이블루':'blue','네이비':'navy',
 '브라운':'brown','블루':'blue','카키':'khaki','레드':'red','그린':'green','라벤더':'purple','실버':'gray','옐로우':'yellow',
 '와인':'red','민트':'green','퍼플':'purple','오렌지':'orange','골드':'yellow'}
NEUTRAL = {'white','black','gray','navy','beige','brown','khaki'}
rows = list(csv.reader(open('crops/meta.csv')))
P = np.load(sys.argv[1])
tot=ok=0; per=collections.Counter(); perok=collections.Counter(); conf=collections.Counter()
n2s=s2n=nn=ss=0; bp=collections.Counter(); bpo=collections.Counter()
thr = float(sys.argv[2]) if len(sys.argv)>2 else 0
cov=0
for idx, pc, pr in P.astype(int):
    r = rows[idx]; gt = MAP[r[4]]; p = MAP[CLASSES[pc]]
    if pr < thr*1000: continue
    cov+=1
    tot+=1; per[gt]+=1; conf[(gt,p)]+=1; bp[r[6]]+=1
    if p==gt: ok+=1; perok[gt]+=1; bpo[r[6]]+=1
    if gt in NEUTRAL: nn+=1; n2s += p not in NEUTRAL
    else: ss+=1; s2n += p in NEUTRAL
print(f'남긴 비율 {cov/len(P):.3f}  family 정확도 {ok}/{tot} = {ok/tot:.3f}')
print(f'기본색인데 강한 색으로: {n2s}/{nn} = {n2s/nn:.3f}   강한 색인데 기본색으로: {s2n}/{ss} = {s2n/ss:.3f}')
for g in sorted(per, key=lambda g:-per[g]):
    top = [(p,c) for (gg,p),c in conf.most_common() if gg==g and p!=g][:3]
    print(f'  {g:7s} {per[g]:6d}  {perok[g]/per[g]:.2f}  틀림: {top}')
for k in sorted(bp, key=lambda k:-bp[k])[:8]: print(f'  {k:8s} {bp[k]:6d} {bpo[k]/bp[k]:.2f}')
