# r7.py: rerankers over probe_v4 for confused kinds. Train on DF2 train only; eval on DF2 val 1841 + FP val + CC0 test.
import json, sys, collections, numpy as np, torch
from train_probe import OK, text_matrix
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = {"신발":"shoes","원피스":"dress","바지":"pants","티셔츠":"tshirt","자켓":"jacket","치마":"skirt","셔츠":"shirt","반바지":"shorts","코트":"coat","조끼":"vest","니트":"sweater","가디건":"cardigan","겉옷":"jacket","맨투맨":"sweater","후드":"sweater","블라우스":"shirt","청바지":"pants","패딩":"coat","점퍼":"jacket"}
BIG = {"dress":"dress","shoes":"shoes","tshirt":"top","jacket":"outer","shorts":"bottom","pants":"bottom","skirt":"bottom","shirt":"top","coat":"outer","sweater":"top","cardigan":"outer","vest":"outer"}
FINE = {"1":"short sleeve top","2":"long sleeve top","3":"short sleeve outwear","4":"long sleeve outwear","5":"vest","6":"sling","7":"shorts","8":"trousers","9":"skirt","10":"short sleeve dress","11":"long sleeve dress","12":"vest dress","13":"sling dress"}
G = ["shirt","tshirt","sweater","cardigan","jacket","coat","vest","pants","shorts","skirt","dress","shoes"]
names, M = text_matrix("probe_v4_l0.3_df2x1.json"); names = list(names)
idx = [[i for i, n in enumerate(names) if n == g] for g in G]
def scores(V): s = V @ M.T; return np.stack([s[:, ix].max(1) for ix in idx], 1)
def load(path, keep, filt=None):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if r.get("empty") or not r.get("v") or not r.get("box"): continue
        if filt and not filt(r): continue
        rows.append({"img": r["img"], "box": r["box"], "label": r["label"], "conf": r.get("conf") or 0, "det": K.get(r["serverName"], r["serverName"]), "v": np.array(r["v"], np.float32)})
    V = np.stack([r["v"] for r in rows]); S = scores(V)
    for r, s in zip(rows, S):
        r["s"] = s; r["base"] = r["det"] if (r["conf"] >= keep or r["det"] == "shoes") else G[int(s[:11].argmax())]
        del r["v"]
    by = collections.defaultdict(list)
    for i, r in enumerate(rows): by[r["img"]].append(i)
    F = []
    for i, r in enumerate(rows):
        x1, y1, x2, y2 = r["box"]; w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3); cy = (y1 + y2) / 2; cx = (x1 + x2) / 2
        top_above = bot_below = dress_o = n = 0; ov_v = 0
        for j in by[r["img"]]:
            if j == i: continue
            o = rows[j]; b = BIG.get(o["base"]); a1, b1, a2, b2 = o["box"]; ocy = (b1 + b2) / 2
            hov = max(0, min(x2, a2) - max(x1, a1)) / w
            if hov < 0.2: continue
            n += 1
            if b in ("top", "outer") and ocy < cy: top_above = 1
            if b == "bottom" and ocy > cy: bot_below = 1
            if b == "dress": dress_o = 1
            ov_v = max(ov_v, max(0, min(y2, b2) - max(y1, b1)) / h)
        F.append([r["conf"], x1, y1, x2, y2, w, h, np.log(h / w), w * h, cy, cx, n, top_above, bot_below, dress_o, ov_v])
    return rows, V, S, np.array(F, np.float32)
def labelset(r):
    l = r["label"]
    if ":" in l: return OK[FINE[l.split(":")[1]]]
    return set(l.split("|")) if "|" in l else OK.get(l, {l})
# --- data
tr = []
for f in ["srv_df2_train.jsonl", "srv_df2_train2.jsonl"]:
    rs, V, S, F = load(f, 0.9); keep = [i for i, r in enumerate(rs) if r["label"] != "none" and labelset(r) <= set(G)]
    tr.append(([rs[i] for i in keep], V[keep], S[keep], F[keep])); print(f, len(keep), flush=True)
trR = sum([t[0] for t in tr], []); trV = np.concatenate([t[1] for t in tr]); trS = np.concatenate([t[2] for t in tr]); trF = np.concatenate([t[3] for t in tr])
low = np.array([r["conf"] < 0.9 and r["det"] != "shoes" for r in trR])
mask = np.array([[g in labelset(r) for g in G] for r in trR])
print("train rows", len(trR), "low-conf", low.sum(), flush=True)
EV = {}
rs, V, S, F = load("df2val2000.jsonl", 0.9, lambda r: ":" in r["label"]); EV["df2val"] = (rs, V, S, F, 0.9)
rs, V, S, F = load("nt_fp_val.jsonl", 0.9, lambda r: r["label"] != "none"); EV["fpval"] = (rs, V, S, F, 0.9)
rs, V, S, F = load("nt_cc0.jsonl", 0.7, lambda r: r["label"] != "none" and r.get("split") == "test"); EV["cc0"] = (rs, V, S, F, 0.7)

import itertools
gi = {g: i for i, g in enumerate(G)}
def repick(rs, S, F, keep, t1, h1, t2):
    P = []
    for r, s, f in zip(rs, S, F):
        p = r["base"]
        if not (r["conf"] >= keep or r["det"] == "shoes"):
            h, top_above, bot_below = f[6], f[12], f[13]
            best = s[:11].max()
            # top picked but no bottom below and tall box -> dress if close
            if BIG.get(p) == "top" and not bot_below and h >= h1 and best - s[gi["dress"]] < t1: p = "dress"
            # dress picked but a top is above -> skirt if close
            elif p == "dress" and top_above and best - s[gi["skirt"]] < t2: p = "skirt"
            # skirt picked but no top above and tall -> dress if close
        P.append(p)
    return P
def acc(rs, P):
    fine = np.mean([p in labelset(r) for p, r in zip(P, rs)])
    big = np.mean([BIG.get(p) in {BIG.get(x) for x in labelset(r)} | ({"top", "outer"} if r["label"].endswith((":5", ":6")) else set()) for p, r in zip(P, rs)])
    return big, fine
trRl = trR
best = None
for t1, h1, t2 in itertools.product([0, 0.01, 0.02, 0.03, 0.05, 0.08], [0.3, 0.4, 0.5, 0.6, 0.7], [0, 0.01, 0.02, 0.03, 0.05]):
    a = acc(trR, repick(trR, trS, trF, 0.9, t1, h1, t2))[1]
    if best is None or a > best[0]: best = (a, t1, h1, t2)
print("train base", acc(trR, [r["base"] for r in trR]), "best", best, flush=True)
for nm, (t1, h1, t2) in [("base", (0, 9, 0)), ("tuned", best[1:]), ("dress only", (best[1], best[2], 0)), ("skirt only", (0, 9, best[3]))]:
    cells = []
    for k, (rs, V, S, F, keep) in EV.items():
        cells.append("%.1f%% / %.1f%%" % tuple(100 * x for x in acc(rs, repick(rs, S, F, keep, t1, h1, t2))))
    print("|", nm, (t1, h1, t2), "|", " | ".join(cells), "|", flush=True)
