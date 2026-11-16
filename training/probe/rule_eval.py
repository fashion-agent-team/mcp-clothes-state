import json, sys, collections, numpy as np
from train_probe import OK, text_matrix
names, M = text_matrix(sys.argv[1]); names = list(names)
K = {"신발":"shoes","원피스":"dress","바지":"pants","티셔츠":"tshirt","자켓":"jacket","치마":"skirt","셔츠":"shirt","반바지":"shorts","코트":"coat","조끼":"vest","니트":"sweater","가디건":"cardigan","겉옷":"jacket","맨투맨":"sweater","후드":"sweater","블라우스":"shirt","청바지":"pants","패딩":"coat","점퍼":"jacket"}
rows = {}
for f, keep in [("nt_fp_val.jsonl", 0.9), ("nt_cc0.jsonl", 0.7)]:
    rs = [json.loads(l) for l in open(f)]
    rs = [r for r in rs if not r.get("empty") and r.get("v") and r["label"] != "none" and (f != "nt_cc0.jsonl" or r["split"] == "test")]
    rows[f] = (rs, keep)
# big category of each fine name, learned from server output (name -> serverCat)
big = collections.defaultdict(collections.Counter)
for f in rows:
    for r in rows[f][0]: big[K.get(r["serverName"], r["serverName"])][r["serverCat"]] += 1
BIG = {k: c.most_common(1)[0][0] for k, c in big.items()}
BIG.update({"longsleeve": "top", "outer": "outer"})
print("BIG", dict(BIG))
gnames = [i for i, n in enumerate(names) if n != "none"]
def decide(r, keep, b0, b1):
    det = K.get(r["serverName"], r["serverName"]); conf = r.get("conf") or 0
    if conf >= keep or det == "shoes": return det
    s = np.array(r["v"], dtype=np.float32) @ M.T
    best = {}
    for i in gnames:
        n = names[i]; sc = s[i] + ((b0 + b1 * conf) if BIG.get(n) == r["serverCat"] else 0)
        best[n] = max(best.get(n, -9), sc)
    return max(best, key=best.get)
def okfine(p, l): return p in OK.get(l, {l})
def okbig(p, l):
    lb = BIG.get(l) or {"short sleeve top":"top"}.get(l); allowed = {BIG.get(x) for x in OK.get(l, {l})}
    return BIG.get(p) in allowed
print("| 규칙 | FP val 큰 분류 | FP val 이름 | CC0 큰 분류 | CC0 이름 |"); print("|---|---|---|---|---|")
for nm, b0, b1 in [("지금(글 벡터 1등)", 0, 0), ("새 0.01+0.005×확신도", 0.01, 0.005), ("0+0.01×확신도", 0, 0.01), ("0.01+0.005 (재확인)", 0.01, 0.005), ("0.02+0", 0.02, 0), ("0.01+0", 0.01, 0), ("0.03+0", 0.03, 0), ("0.05+0", 0.05, 0), ("0.02+0.01×확신도", 0.02, 0.01)]:
    cells = []
    for f in ["nt_fp_val.jsonl", "nt_cc0.jsonl"]:
        rs, keep = rows[f]; p = [decide(r, keep, b0, b1) for r in rs]
        cells += [f"{np.mean([okbig(a, r["label"]) for a, r in zip(p, rs)])*100:.1f}%", f"{np.mean([okfine(a, r["label"]) for a, r in zip(p, rs)])*100:.1f}%"]
    print(f"| {nm} | " + " | ".join(cells) + " |")
for f in rows: print(f, "수", len(rows[f][0]), "확신도 낮음 비율", f"{np.mean([(r.get("conf") or 0) < rows[f][1] for r in rows[f][0]])*100:.0f}%")
