s = open("r7.py").read(); exec(s[s.index("import json"):s.index("# --- data")])
rs, V, S, F = load("df2val2000.jsonl", 0.9, lambda r: ":" in r["label"])
out = open("r9_rows.jsonl", "w")
for r, s in zip(rs, S):
    if r["conf"] >= 0.9 or r["det"] == "shoes": continue
    o = np.argsort(-s[:11]); out.write(json.dumps({"img": r["img"].split("/")[-1], "box": r["box"], "label": r["label"], "fine": FINE[r["label"].split(":")[1]],
        "conf": r["conf"], "det": r["det"], "base": r["base"], "top": [G[i] for i in o[:3]], "gap": float(s[o[0]] - s[o[1]]), "ok": r["base"] in labelset(r)}) + "\n")
