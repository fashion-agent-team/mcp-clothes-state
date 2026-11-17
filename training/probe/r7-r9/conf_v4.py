import json, sys, collections, numpy as np
sys.argv = ["x", "df2val2000.jsonl", "probe_v4_l0.3_df2x1.json"]
src = open("df2val_eval.py").read().split("print(f\"옷")[0]
exec(src)
names, M = text_matrix(sys.argv[2]); names = list(names)
P = [decide(r, names, M, 0, 0) for r in rows]
err = collections.Counter(); tot = collections.Counter(); by_src = collections.Counter()
for p, r in zip(P, rows):
    f = FINE[r["label"].split(":")[1]]; tot[f] += 1
    if not okfine(p, r):
        det = K.get(r["serverName"], r["serverName"]); conf = r.get("conf") or 0
        err[(f, p)] += 1; by_src["detector(conf>=0.9)" if conf >= 0.9 or det == "shoes" else "probe"] += 1
print("오답", sum(err.values()), "/", len(rows), dict(by_src))
print("| 정답(DF2) | 예측 | 수 |"); print("|---|---|---|")
for (f, p), c in err.most_common(15): print(f"| {f} | {p} | {c} |")
print("클래스별 정답률")
for f in tot: print(f, tot[f], f"{1 - sum(c for (a, b), c in err.items() if a == f) / tot[f]:.1%}")
