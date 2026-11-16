# train_probe4.py TEXT_JSON OUT_JSON LAM DF2_WEIGHT : text-initialized prototypes, partial labels ("a|b"), DF2 rows weighted.
import sys, json, os, numpy as np, torch
from train_probe import CLASSES, load, text_matrix, predict, accuracy
text_json, out_json, lam, dfw = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
names, M0 = text_matrix(text_json); names = list(names)
V, L, S, R = load(["srv_fp_train.npz", "srv_cc0.npz", "walls.npz", "srv_fp_train_rare.npz", "srv_df2_train.npz"])
EV, EL, ES, ER = load(["srv_fp_val.npz"])
sets = [set(l.split("|")) for l in L]
m = (S == "train") & np.array([s <= set(CLASSES) for s in sets])
X = torch.tensor(V[m]).cuda()
mask = torch.tensor(np.array([[c in sets[i] for c in CLASSES] for i in np.where(m)[0]])).cuda()
first = torch.tensor([CLASSES.index(sorted(sets[i])[0]) if len(sets[i]) == 1 else -1 for i in np.where(m)[0]]).cuda()
srcw = torch.tensor(np.where(R[m] == "df2-server", dfw, 1.0), dtype=torch.float32).cuda()
# class balance on single-label rows (multi-label rows weight 1)
cnt = torch.bincount(first[first >= 0], minlength=len(CLASSES)).float().clamp(min=1); cw = (1 / cnt) ** 0.5; cw = cw / cw.mean()
roww = torch.where(first >= 0, cw[first.clamp(min=0)], torch.ones_like(srcw)) * srcw
cls_of = torch.tensor([CLASSES.index(n) for n in names]).cuda()
W0 = torch.tensor(M0).cuda(); W = W0.clone().requires_grad_(True); logs = torch.tensor(3.4, device="cuda", requires_grad=True)
opt = torch.optim.Adam([W, logs], lr=1e-3); torch.manual_seed(0)
print("train rows", len(X), "df2 rows", int((srcw != 1).sum()) if dfw != 1 else int(torch.tensor(R[m] == "df2-server").sum()), flush=True)
for ep in range(40):
    perm = torch.randperm(len(X), device="cuda")
    for i in range(0, len(X), 256):
        idx = perm[i:i+256]; s = X[idx] @ torch.nn.functional.normalize(W, dim=1).T
        sc = torch.full((len(idx), len(CLASSES)), -9.0, device="cuda").scatter_reduce(1, cls_of.expand(len(idx), -1), s, "amax") * logs.exp()
        lse = torch.logsumexp(sc, 1); pos = torch.logsumexp(sc.masked_fill(~mask[idx], -1e4), 1)
        loss = ((lse - pos) * roww[idx]).sum() / roww[idx].sum() + lam * ((W - W0) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
Mx = torch.nn.functional.normalize(W, dim=1).detach().cpu().numpy()
def acc(Mm, Xe, Le):
    k = Le != "none"; g, _ = predict(Xe[k], names, Mm); return accuracy(g, Le[k])
cc0t = (S == "test") & (R == "cc0-server")
for nm, Xe, Le in [("fp-server val", EV, EL), ("cc0-server test", V[cc0t], L[cc0t])]:
    print(f"{nm:16s} {int(sum(Le != "none")):5d}  글 {acc(M0, Xe, Le)*100:.1f}%  학습 {acc(Mx, Xe, Le)*100:.1f}%", flush=True)
json.dump({"model": f"Marqo/marqo-fashionSigLIP 글 벡터에서 시작해 서버 박스 벡터로 다듬음(lambda {lam}, DF2 x{dfw}; Fashionpedia CC BY 4.0 + clothing-dataset CC0 + 만든 벽 + DeepFashion2 train 일반인 사진(비상업 연구·교육용))",
           "template": "learned-from-text", "prompts": [{"name": n, "text": "tuned:" + n, "vector": [round(float(x), 6) for x in Mx[i]]} for i, n in enumerate(names)]}, open(out_json, "w"))
print("저장", out_json, flush=True)
