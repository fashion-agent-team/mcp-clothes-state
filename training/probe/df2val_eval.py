import json, sys, glob, numpy as np, collections
from train_probe import OK, text_matrix
K = {"신발":"shoes","원피스":"dress","바지":"pants","티셔츠":"tshirt","자켓":"jacket","치마":"skirt","셔츠":"shirt","반바지":"shorts","코트":"coat","조끼":"vest","니트":"sweater","가디건":"cardigan","겉옷":"jacket","맨투맨":"sweater","후드":"sweater","블라우스":"shirt","청바지":"pants","패딩":"coat","점퍼":"jacket"}
BIG = {"dress":"dress","shoes":"shoes","tshirt":"top","jacket":"outer","shorts":"bottom","pants":"bottom","skirt":"bottom","shirt":"top","coat":"outer","sweater":"top","cardigan":"outer","vest":"outer"}
FINE = {"1":"short sleeve top","2":"long sleeve top","3":"short sleeve outwear","4":"long sleeve outwear","5":"vest","6":"sling","7":"shorts","8":"trousers","9":"skirt","10":"short sleeve dress","11":"long sleeve dress","12":"vest dress","13":"sling dress"}
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if not r.get("empty") and r.get("v") and ":" in r["label"]]
def okbig(p, r):
    big, cid = r["label"].split(":")
    allowed = {big} | ({"top", "outer"} if cid in ("5", "6") else set())
    return BIG.get(p) in allowed
def okfine(p, r): return p in OK[FINE[r["label"].split(":")[1]]]
def decide(r, names, M, b0, b1, keep=0.9):
    det = K.get(r["serverName"], r["serverName"]); conf = r.get("conf") or 0
    if conf >= keep or det == "shoes": return det
    s = np.array(r["v"], dtype=np.float32) @ M.T; best = {}
    for i, n in enumerate(names):
        if n == "none": continue
        sc = s[i] + ((b0 + b1 * conf) if BIG.get(n) == r["serverCat"] else 0); best[n] = max(best.get(n, -9), sc)
    return max(best, key=best.get)
print(f"옷 {len(rows)}개 (사진 {len({r["img"] for r in rows})}장), 확신도 0.9 미만 {np.mean([(r.get("conf") or 0) < .9 for r in rows])*100:.0f}%")
files = [sys.argv[2]] + sorted(glob.glob("probe_v2.json") + glob.glob("probe_v3_l0.3.json") + glob.glob("probe_v4_*.json") + glob.glob("probe_v5_*.json"))
print("| 벡터 | 가산점 | 큰 분류 | 세부(DF2 13종) |"); print("|---|---|---|---|")
for f in files:
    names, M = text_matrix(f); names = list(names)
    for b0, b1 in [(0, 0), (0.01, 0.005), (0.02, 0), (0.03, 0)]:
        p = [decide(r, names, M, b0, b1) for r in rows]
        print(f"| {f.split("/")[-1]} | {b0}+{b1}c | {np.mean([okbig(a, r) for a, r in zip(p, rows)])*100:.1f}% | {np.mean([okfine(a, r) for a, r in zip(p, rows)])*100:.1f}% |", flush=True)
# detector only
p = [K.get(r["serverName"], r["serverName"]) for r in rows]
print(f"| 검출기만 | - | {np.mean([okbig(a, r) for a, r in zip(p, rows)])*100:.1f}% | {np.mean([okfine(a, r) for a, r in zip(p, rows)])*100:.1f}% |")
