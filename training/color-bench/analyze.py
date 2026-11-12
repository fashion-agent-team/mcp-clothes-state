import csv, sys, random, collections
bench, labels = sys.argv[1], sys.argv[2]
lab = {r['img']: r for r in csv.DictReader(open(labels))}
rows = list(csv.DictReader(open(bench), delimiter='\t'))
BODY = {'top', 'bottom', 'outer', 'dress'}
stats = collections.Counter()
use = []
for r in rows:
    stats['photos'] += 1
    if r['lighting'] != 'ok':
        stats['not_ok:' + r['lighting']] += 1; continue
    g = [x.split(':') for x in r['garments'].split(';') if x]
    cats = {x[0] for x in g if x[0] in BODY and x[2] != 'unknown'}
    if len(cats) < 2:
        stats['under2_body'] += 1; continue
    l = lab.get(r['img'])
    if not l:
        stats['no_label'] += 1; continue
    use.append((r, l, g))
print(dict(stats), 'used', len(use))
fam = collections.Counter(x[2] for _, _, g in use for x in g if x[0] in BODY)
print('색 분포', fam.most_common())

def rate(items):
    n = sum(int(l['n']) for _, l, _ in items); k = sum(int(l['q5_1']) for _, l, _ in items)
    return k, n

def boot(items, reps=2000):
    vals = []
    for _ in range(reps):
        s = [random.choice(items) for _ in items]
        k, n = rate(s); vals.append(k / n if n else 0)
    vals.sort(); return vals[int(reps * .025)], vals[int(reps * .975)]

random.seed(0)
print('\n== 일상(casual) 판정별 Q5=1 비율 ==')
for v in ['good', 'ok', 'bad']:
    it = [u for u in use if u[0]['casual'] == v]
    if not it: continue
    k, n = rate(it); lo, hi = boot(it)
    print(f'{v:5s} 사진 {len(it):4d} 응답 {n:4d}  Q5=1 {k/n:.3f}  (95% {lo:.3f}~{hi:.3f})')
print('\n== 스타일별 ==')
for st in sorted({l['style'] for _, l, _ in use}):
    line = st
    for v in ['good', 'ok', 'bad']:
        it = [u for u in use if u[0]['casual'] == v and u[1]['style'] == st]
        if it:
            k, n = rate(it); line += f'  {v} {k/n:.3f} ({len(it)}장)'
    print(line)
print('\n== 규칙별 ==')
rc = collections.defaultdict(list)
for u in use:
    for rule in filter(None, u[0]['casual_rules'].split(',')):
        rc[rule].append(u)
for rule, it in sorted(rc.items(), key=lambda x: -len(x[1])):
    k, n = rate(it); print(f'{rule:22s} 사진 {len(it):4d}  Q5=1 {k/n:.3f}')
print('\n면접(formal) 판정 분포', collections.Counter(u[0]['formal'] for u in use))
