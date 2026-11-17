# r8: (1) per-class bias tuned by accuracy (coordinate ascent) on DF2 train; (2) residual MLP trained on ALL train sources (FP+CC0+DF2)
s = open("r7.py").read(); exec(s[s.index("import json"):s.index("mu, sd =")])
from train_probe import load as nload
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
gi = {g: i for i, g in enumerate(G)}
def acc(rs, P):
    fine = np.mean([p in labelset(r) for p, r in zip(P, rs)])
    big = np.mean([BIG.get(p) in {BIG.get(x) for x in labelset(r)} | ({"top", "outer"} if r["label"].endswith((":5", ":6")) else set()) for p, r in zip(P, rs)])
    return big, fine
def pick(rs, Z, keep):
    Z = Z.copy(); Z[:, gi["shoes"]] = -1e9
    return [r["det"] if (r["conf"] >= keep or r["det"] == "shoes") else G[int(z.argmax())] for r, z in zip(rs, Z)]
def line(nm, f):
    print("| " + nm + " | " + " | ".join("%.1f%% / %.1f%%" % tuple(100 * x for x in acc(rs, pick(rs, f(V, S), keep))) for k, (rs, V, S, F, keep) in EV.items()) + " |", flush=True)
print("| 방식 | DF2 val 1841 | FP val | CC0 test |\n|---|---|---|---|")
line("probe_v4", lambda V, S: S)
# (1) bias by accuracy
sel = np.where(low)[0]; Str = trS[sel]; Rtr = [trR[i] for i in sel]; Mtr = mask[sel]
def tacc(b): z = Str + b; z[:, gi["shoes"]] = -1e9; return Mtr[np.arange(len(z)), z.argmax(1)].mean()
b = np.zeros(12, np.float32)
for it in range(3):
    for c in range(11):
        best = (tacc(b), b[c])
        for d in np.arange(-0.04, 0.0401, 0.005):
            bb = b.copy(); bb[c] = d; a = tacc(bb)
            if a > best[0] + 1e-4: best = (a, d)
        b[c] = best[1]
print("bias", dict(zip(G, np.round(b, 3))), "train", tacc(np.zeros(12, np.float32)), "->", tacc(b), flush=True)
line("(b) 종류별 가산점(정답률로 맞춤)", lambda V, S: S + b)
bd = np.zeros(12, np.float32); bd[gi["dress"]] = b[gi["dress"]]
line("(b) dress 가산점만", lambda V, S: S + bd)
# (2) MLP on all sources
Va, La, Sa, Ra = nload(["srv_fp_train.npz", "srv_cc0.npz", "srv_fp_train_rare.npz", "srv_df2_train.npz", "srv_df2_train2.npz"])
sets = [set(l.split("|")) if "|" in l else OK.get(l, {l}) for l in La]
m = (Sa == "train") & np.array([len(x) > 0 and x <= set(G) for x in sets])
Xa = Va[m]; Ma = np.array([[g in sets[i] for g in G] for i in np.where(m)[0]]); Rs = Ra[m]; Ls = La[m]
cnt = collections.Counter(zip(Rs, Ls)); srcn = collections.Counter(Rs)
w = np.array([(1 / cnt[(r, l)]) ** 0.5 for r, l in zip(Rs, Ls)]); 
for src in srcn: w[Rs == src] /= w[Rs == src].sum()
w = w / w.mean(); print("rows", len(Xa), dict(srcn), flush=True)
Sx = scores(Xa)
def fit(hid, lam, ep=30):
    f = torch.nn.Sequential(torch.nn.Linear(768, hid), torch.nn.ReLU(), torch.nn.Linear(hid, 12)).to(dev)
    torch.nn.init.zeros_(f[2].weight); torch.nn.init.zeros_(f[2].bias)
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    X, Sg, Mm, W = T(Xa), T(Sx), torch.tensor(Ma, device=dev), T(w)
    for e in range(ep):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 256):
            ix = perm[i:i+256]; z = 30 * Sg[ix] + f(X[ix])
            loss = ((torch.logsumexp(z, 1) - z.masked_fill(~Mm[ix], -1e4).amax(1)) * W[ix]).sum() / W[ix].sum()
            loss = loss + lam * (f[0].weight.pow(2).sum() + f[2].weight.pow(2).sum())
            opt.zero_grad(); loss.backward(); opt.step()
    return f
nets = {}
for hid, lam in [(256, 1e-3), (256, 3e-4), (128, 1e-3), (256, 3e-3)]:
    f = fit(hid, lam); nets[(hid, lam)] = f
    def run(V, S, f=f):
        with torch.no_grad(): return 30 * S + f(T(V)).cpu().numpy()
    line(f"(c) 전체 소스 MLP{hid} λ{lam}", run)
torch.save({str(k): v.state_dict() for k, v in nets.items()}, "r8_models.pt"); np.save("r8_bias.npy", b)
print("R8 DONE", flush=True)
