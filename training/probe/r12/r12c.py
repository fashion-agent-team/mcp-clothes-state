# r12c: assist path = v4 first; re-check with So400m+probe only when v4 is unsure. Offline on r11 embeddings.
import sys, json, numpy as np, torch
sys.argv = ["r11.py", "NONE"]
src = open("r11.py").read(); src = src[:src.index("res = {}")].replace("dev = \"cuda\"", "dev = \"cpu\"")
exec(src)
from train_probe import text_matrix
A = np.load("r11_Marqo_marqo-fashionSigLIP.npz"); B = np.load("r11_ViT-SO400M-14-SigLIP2.npz")
n4, M4 = text_matrix("probe_v4_l0.3_df2x1.json"); n4 = list(n4)
Mt = np.load("r12_so400m_text.npy"); Mp, pn = probe(B["tr"], [r["label"] for r in TR], Mt, PN)
json.dump({"model": "SigLIP2 So400m/14-224 webli + probe(λ0.3, DF2 5k + FP 5k) — 평가용, 배포 안 함", "prompts": [{"name": n, "vector": [round(float(x), 5) for x in Mp[i]]} for i, n in enumerate(pn)]}, open("r12_so400m_probe.json", "w"))
def decide(rows, keep, S4, SB, t, c):
    P = []; asked = 0
    for r, a, b in zip(rows, S4, SB):
        if r["conf"] >= keep or r["det"] == "shoes": P.append(r["det"]); continue
        a = a.copy(); a[G.index("shoes")] = -9; o = np.sort(a)[::-1]; p = G[int(a.argmax())]
        if (o[0] - o[1]) < t and r["conf"] < c:
            b = b.copy(); b[G.index("shoes")] = -9; p = G[int(b.argmax())]; asked += 1
        P.append(p)
    fine = np.mean([p in labelset(r["label"]) for p, r in zip(P, rows)])
    big = np.mean([BIG.get(p) in {BIG.get(x) for x in labelset(r["label"])} | ({"top", "outer"} if r["label"].endswith((":5", ":6")) else set()) for p, r in zip(P, rows)])
    return big, fine, asked / len(rows)
S = {k: (gscores(A[k], M4, n4), gscores(B[k], Mp, pn)) for k in EV}
res = []
print("| v4 1·2등 차이 < t | 검출기 확신도 < c | DF2 val 다시 묻는 비율 | DF2 val 큰/세부 | FP val 다시 묻는 비율 | FP val 큰/세부 |\n|---|---|---|---|---|---|")
for t in [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 9]:
    for c in [0.9, 0.7, 0.5]:
        if t == 0 and c != 0.9: continue
        row = {"t": t, "c": c}; cells = []
        for k, (rows, keep) in EV.items():
            big, fine, asked = decide(rows, keep, *S[k], t, c); row[k] = [round(100 * big, 1), round(100 * fine, 1), round(100 * asked, 1)]
            cells += [f"{asked:.0%}", f"{big*100:.1f}% / {fine*100:.1f}%"]
        res.append(row); print(f"| {t} | {c} | " + " | ".join(cells) + " |", flush=True)
json.dump(res, open("r12c_result.json", "w"), indent=1)
