# train_probe3.py: start from server text prototypes (35 prompts, 13 names), fine-tune with pull-back to text (lambda),
# class-balanced, server-box vectors. Select lambda on 20% held-out train images. Eval on srv_fp_val + cc0 test.
import sys, json, os, numpy as np, torch
from train_probe import CLASSES, load, text_matrix, predict, accuracy
text_json, out_json = sys.argv[1], sys.argv[2]
names, M0 = text_matrix(text_json); names = list(names)
V, L, S, R = load(["srv_fp_train.npz", "srv_cc0.npz", "walls.npz"] + os.environ.get("EXTRA", "").split() ); EV, EL, ES, ER = load(["srv_fp_val.npz"])
m = (S == "train") & np.isin(L, CLASSES)
X, Y = V[m], np.array([CLASSES.index(l) for l in L[m]])
rng = np.random.default_rng(0); hold = rng.random(len(X)) < 0.2
cls_of = torch.tensor([CLASSES.index(n) for n in names]).cuda()
def fit(Xt, yt, lam, epochs=40):
    Xt = torch.tensor(Xt).cuda(); yt = torch.tensor(yt).cuda()
    W0 = torch.tensor(M0).cuda(); W = W0.clone().requires_grad_(True); logs = torch.tensor(3.4, device="cuda", requires_grad=True)
    cnt = torch.bincount(yt, minlength=len(CLASSES)).float().clamp(min=1); cw = (1 / cnt) ** 0.5; cw = cw / cw[yt].mean()
    opt = torch.optim.Adam([W, logs], lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(Xt), 256):
            idx = perm[i:i+256]; Wn = torch.nn.functional.normalize(W, dim=1); s = Xt[idx] @ Wn.T
            sc = torch.full((len(idx), len(CLASSES)), -9.0, device="cuda").scatter_reduce(1, cls_of.expand(len(idx), -1), s, "amax")
            loss = torch.nn.functional.cross_entropy(sc * logs.exp(), yt[idx], weight=cw) + lam * ((W - W0) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return torch.nn.functional.normalize(W, dim=1).detach().cpu().numpy()
def acc(Mx, Xe, Le):
    keep = Le != "none"; g, _ = predict(Xe[keep], names, Mx); return accuracy(g, Le[keep])
Lh = np.array([CLASSES[i] for i in Y[hold]])
print("held-out 글 벡터", f"{acc(M0, X[hold], Lh)*100:.1f}%", flush=True)
best = None
for lam in ([float(os.environ["LAM"])] if os.environ.get("LAM") else [1.0, 0.3, 0.1, 0.03, 0.01]):
    Mx = fit(X[~hold], Y[~hold], lam); a = acc(Mx, X[hold], Lh); print(f"lambda {lam}: held-out {a*100:.1f}%", flush=True)
    if not best or a > best[0]: best = (a, lam)
lam = best[1]; Mx = fit(X, Y, lam)
cc0t = (S == "test") & (R == "cc0-server")
print(f"\n최종 lambda {lam}")
for nm, Xe, Le in [("fp-server val", EV, EL), ("cc0-server test", V[cc0t], L[cc0t])]:
    print(f"{nm:16s} {int((Le!="none").sum()):5d}  글 {acc(M0, Xe, Le)*100:.1f}%  학습 {acc(Mx, Xe, Le)*100:.1f}%", flush=True)
json.dump({"model": f"Marqo/marqo-fashionSigLIP 글 벡터에서 시작해 서버 박스 벡터로 다듬음(lambda {lam}; Fashionpedia CC BY 4.0 + clothing-dataset CC0 + 만든 벽)",
           "template": "learned-from-text", "prompts": [{"name": n, "text": "tuned:" + n, "vector": [round(float(x), 6) for x in Mx[i]]} for i, n in enumerate(names)]},
          open(out_json, "w"))
print("저장", out_json, flush=True)
