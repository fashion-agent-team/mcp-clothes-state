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
        bx = r["box"]; rows.append({"img": r["img"], "box": [bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]], "label": r["label"], "conf": r.get("conf") or 0, "det": K.get(r["serverName"], r["serverName"]), "v": np.array(r["v"], np.float32)})
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
mu, sd = trF.mean(0), trF.std(0) + 1e-6
T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
class Net(torch.nn.Module):
    def __init__(s, use_emb, use_ctx, hid):
        super().__init__(); s.use_emb, s.use_ctx = use_emb, use_ctx; d = (768 if use_emb else 0) + (16 if use_ctx else 0)
        s.a = torch.nn.Parameter(torch.tensor(30.0)); s.b = torch.nn.Parameter(torch.zeros(12))
        s.f = torch.nn.Sequential(torch.nn.Linear(d, hid), torch.nn.ReLU(), torch.nn.Linear(hid, 12)) if hid else torch.nn.Linear(d, 12)
        last = s.f[-1] if hid else s.f; torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
    def forward(s, V, S, F):
        x = torch.cat(([V] if s.use_emb else []) + ([(F - T(mu)) / T(sd)] if s.use_ctx else []), 1)
        return s.a * S + s.b + s.f(x)
def fit(use_emb, use_ctx, hid, lam, cw_boost=None, ep=30):
    net = Net(use_emb, use_ctx, hid).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    sel = np.where(low)[0]; Vt, St, Ft, Mt = T(trV[sel]), T(trS[sel]), T(trF[sel]), torch.tensor(mask[sel], device=dev)
    labs = [trR[i]["label"] for i in sel]; cnt = collections.Counter(labs)
    w = torch.tensor([(1 / cnt[l]) ** 0.5 for l in labs], device=dev); w = w / w.mean()
    if cw_boost:
        for g, k in cw_boost.items(): w = torch.where(Mt[:, G.index(g)], w * k, w)
    for e in range(ep):
        perm = torch.randperm(len(sel), device=dev)
        for i in range(0, len(sel), 256):
            ix = perm[i:i+256]; z = net(Vt[ix], St[ix], Ft[ix])
            loss = ((torch.logsumexp(z, 1) - z.masked_fill(~Mt[ix], -1e4).amax(1)) * w[ix]).sum() / w[ix].sum()
            reg = sum((p ** 2).sum() for n_, p in net.f.named_parameters() if "weight" in n_)
            opt.zero_grad(); (loss + lam * reg).backward(); opt.step()
    return net
def evaluate(net):
    out = {}
    for k, (rs, V, S, F, keep) in EV.items():
        if net is None: P = [r["base"] for r in rs]
        else:
            with torch.no_grad(): z = net(T(V), T(S), T(F)).cpu().numpy()
            z[:, G.index("shoes")] = -1e9
            P = [r["det"] if (r["conf"] >= keep or r["det"] == "shoes") else G[int(zz.argmax())] for r, zz in zip(rs, z)]
        fine = np.mean([p in labelset(r) for p, r in zip(P, rs)])
        big = np.mean([BIG.get(p) in {BIG.get(x) for x in labelset(r)} | ({"top", "outer"} if r["label"].endswith((":5", ":6")) else set()) for p, r in zip(P, rs)])
        out[k] = (big, fine)
    return out
def line(nm, o): print(f"| {nm} | " + " | ".join(f"{o[k][0]*100:.1f}% / {o[k][1]*100:.1f}%" for k in ["df2val", "fpval", "cc0"]) + " |", flush=True)
print("| 방식 | DF2 val 1841 큰/세부 | FP val 큰/세부 | CC0 test 큰/세부 |\n|---|---|---|---|")
line("probe_v4 (배포)", evaluate(None))
res = {}
for nm, args in [("(a) 박스·주변 옷 선형", (False, True, 0, 1e-3)), ("(a) 박스·주변 옷 MLP64", (False, True, 64, 1e-3)),
                 ("(c) 임베딩 선형 잔차 λ1e-2", (True, False, 0, 1e-2)), ("(c) 임베딩 MLP256 λ1e-3", (True, False, 256, 1e-3)),
                 ("(c) 임베딩 MLP256 λ1e-2", (True, False, 256, 1e-2)),
                 ("(a+c) 임베딩+박스 MLP256 λ1e-3", (True, True, 256, 1e-3)), ("(a+c) 임베딩+박스 MLP256 λ1e-2", (True, True, 256, 1e-2))]:
    net = fit(*args); o = evaluate(net); line(nm, o); res[nm] = (net, o)
for nm, args, cb in [("(b) 박스·주변 MLP64 + dress·skirt×2", (False, True, 64, 1e-3), {"dress": 2, "skirt": 2}),
                     ("(b) 임베딩+박스 MLP256 + dress·skirt×2", (True, True, 256, 1e-3), {"dress": 2, "skirt": 2})]:
    net = fit(*args, cw_boost=cb); o = evaluate(net); line(nm, o); res[nm] = (net, o)
best = max(res, key=lambda k: res[k][1]["df2val"][1]); print("best", best, flush=True)
torch.save({k: v[0].state_dict() for k, v in res.items()}, "r7e_models.pt")
print("R7E DONE", flush=True)
