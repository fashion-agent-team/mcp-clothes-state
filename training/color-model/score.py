import csv, sys, collections
MAP = {'화이트':'white','블랙':'black','베이지':'beige','핑크':'pink','그레이':'gray','스카이블루':'blue','네이비':'navy',
 '브라운':'brown','블루':'blue','카키':'khaki','레드':'red','그린':'green','라벤더':'purple','실버':'gray','옐로우':'yellow',
 '와인':'red','민트':'green','퍼플':'purple','오렌지':'orange','골드':'yellow'}
NEUTRAL = {'white','black','gray','navy','beige','brown','khaki'}
meta = {r[0]: r for r in csv.reader(open(sys.argv[1]))}
pred = {r[0]: r for r in csv.reader(open(sys.argv[2]))}
tot=ok=0; conf=collections.Counter(); per=collections.Counter(); perok=collections.Counter()
n2s=s2n=nn=ss=0; byprint=collections.Counter(); byprintok=collections.Counter()
for id, m in meta.items():
    gt = MAP.get(m[4])
    if gt is None or id not in pred: continue
    p = pred[id][1]
    tot+=1; per[gt]+=1; byprint[m[6]]+=1
    if p==gt: ok+=1; perok[gt]+=1; byprintok[m[6]]+=1
    conf[(gt,p)]+=1
    g_n = gt in NEUTRAL; p_n = p in NEUTRAL or p=='unknown'
    if g_n: nn+=1; n2s += (not p_n)
    else: ss+=1; s2n += p_n
print(f'family 정확도 {ok}/{tot} = {ok/tot:.3f}')
print(f'기본색인데 강한 색으로: {n2s}/{nn} = {n2s/nn:.3f}   강한 색인데 기본색으로: {s2n}/{ss} = {s2n/ss:.3f}')
for g in sorted(per, key=lambda g:-per[g]):
    top = [(p,c) for (gg,p),c in conf.most_common() if gg==g and p!=g][:3]
    print(f'  {g:7s} {per[g]:6d}  {perok[g]/per[g]:.2f}  틀림: {top}')
print('프린트별:')
for k in sorted(byprint, key=lambda k:-byprint[k])[:12]:
    print(f'  {k:8s} {byprint[k]:6d} {byprintok[k]/byprint[k]:.2f}')
