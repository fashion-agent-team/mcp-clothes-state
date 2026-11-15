#!/usr/bin/env python3
"""옷 벡터 → 종류 분류기(편향 없는 선형, 클래스마다 벡터 하나). 서버 GarmentTextCheck가 그대로 읽는 JSON으로 저장.
사용: python3 train_probe.py TEXT_JSON OUT_JSON train1.npz [train2.npz ...] --eval eval1.npz [eval2.npz ...]
 - 학습: split=train 인 줄(평가 파일은 학습에 안 씀). 'longsleeve','outer' 같은 애매한 라벨은 학습에서 뺌.
 - 평가: 글 벡터(지금 서버) vs 학습한 벡터, 소스별 정답률. none 기준값(margin)도 고름.
"""
import json, sys
import numpy as np

CLASSES = ["shirt", "tshirt", "sweater", "cardigan", "jacket", "coat", "vest", "pants", "shorts", "skirt", "dress",
           "shoes", "none"]
# 평가 때 맞다고 볼 이름(데이터 라벨 → 허용 이름)
OK = {"longsleeve": {"tshirt", "sweater", "shirt"}, "outer": {"jacket", "coat", "cardigan"},
      # DeepFashion2 라벨
      "short sleeve top": {"tshirt", "shirt"}, "long sleeve top": {"tshirt", "shirt", "sweater"},
      "short sleeve outwear": {"jacket", "shirt", "cardigan"}, "long sleeve outwear": {"jacket", "coat", "cardigan"},
      "vest": {"vest", "tshirt"}, "sling": {"tshirt", "vest"}, "shorts": {"shorts"}, "trousers": {"pants"},
      "skirt": {"skirt"}, "short sleeve dress": {"dress"}, "long sleeve dress": {"dress"}, "vest dress": {"dress"},
      "sling dress": {"dress"}}


def load(paths):
    V, L, S, R = [], [], [], []
    for p in paths:
        z = np.load(p)
        V.append(z["v"]); L += list(z["label"]); S += list(z["split"]); R += list(z["src"])
    return np.concatenate(V), np.array(L), np.array(S), np.array(R)


def text_matrix(path):
    d = json.load(open(path))
    names = [p["name"] for p in d["prompts"]]
    M = np.array([p["vector"] for p in d["prompts"]], dtype=np.float32)
    return names, M


def predict(V, names, M):
    s = V @ M.T
    uniq = sorted(set(names), key=names.index)
    best = np.stack([s[:, [i for i, n in enumerate(names) if n == u]].max(1) for u in uniq], 1)
    garment = [i for i, u in enumerate(uniq) if u != "none"]
    gi = best[:, garment].argmax(1)
    gname = np.array([uniq[garment[k]] for k in gi])
    margin = best[:, uniq.index("none")] - best[:, garment].max(1)
    return gname, margin


def accuracy(gname, labels):
    ok = [g in OK.get(l, {l}) for g, l in zip(gname, labels)]
    return float(np.mean(ok)) if ok else float("nan")


def train(X, y, epochs=300, lr=0.5, l2=1e-4):
    n, d = X.shape
    k = len(CLASSES)
    W = np.zeros((k, d), np.float32)
    Y = np.zeros((n, k), np.float32); Y[np.arange(n), y] = 1
    cw = 1.0 / np.maximum(np.bincount(y, minlength=k), 1); cw = cw / cw[y].mean()
    sw = cw[y][:, None]
    scale = 30.0  # 벡터 길이 1이라 점수가 작아서 키움(모양만 바꿈, 저장 때 되돌림)
    for e in range(epochs):
        Z = scale * X @ W.T
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = scale * ((P - Y) * sw).T @ X / n + l2 * W
        W -= lr * G
        if e % 50 == 0:
            print(f"epoch {e} loss {-(np.log(P[np.arange(n), y] + 1e-9) * sw[:, 0]).mean():.4f}", flush=True)
    return W


def main():
    a = sys.argv[1:]
    text_json, out_json = a[0], a[1]
    ei = a.index("--eval")
    tr_paths, ev_paths = a[2:ei], a[ei + 1:]
    V, L, S, R = load(tr_paths)
    m = (S == "train") & np.isin(L, CLASSES)
    X, y = V[m], np.array([CLASSES.index(l) for l in L[m]])
    print("학습", len(X), {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
    W = train(X, y)
    # 길이 맞추기: 글 벡터 점수와 비슷한 크기(행 길이 1)로
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    names_t, M_t = text_matrix(text_json)
    names_p, M_p = CLASSES, W
    # 평가: 학습 파일의 test 줄 + 평가 파일
    EV, EL, ES, ER = load(ev_paths) if ev_paths else (V[:0], L[:0], S[:0], R[:0])
    T = np.concatenate([V[S == "test"], EV]); TL = np.concatenate([L[S == "test"], EL])
    TR = np.concatenate([R[S == "test"], ER])
    print(f"\n{'소스':16s} {'수':>5s} {'글 벡터':>8s} {'학습':>8s}")
    res = {}
    for src in sorted(set(TR)):
        mm = (TR == src) & (TL != "none")
        if mm.sum() == 0:
            continue
        gt, _ = predict(T[mm], names_t, M_t)
        gp, _ = predict(T[mm], names_p, M_p)
        res[src] = (int(mm.sum()), accuracy(gt, TL[mm]), accuracy(gp, TL[mm]))
        print(f"{src:16s} {mm.sum():5d} {res[src][1]*100:7.1f}% {res[src][2]*100:7.1f}%")
    # none 기준값: 진짜 옷이 옷 아님으로 빠지는 비율 0.5% 이하에서 none을 가장 많이 잡는 값
    gm = TL != "none"; nm = TL == "none"
    for nm_, (names, M) in {"글 벡터": (names_t, M_t), "학습": (names_p, M_p)}.items():
        _, mg = predict(T, names, M)
        thr = float(np.quantile(mg[gm], 0.995)) if gm.any() else 0.0
        caught = float((mg[nm] > thr).mean()) if nm.any() else float("nan")
        print(f"{nm_}: none 기준 {thr:.3f} → 옷 빠짐 0.5%, 벽·배경 잡음 {caught*100:.1f}% (none {nm.sum()}개)")
        if nm_ == "학습":
            best_thr = thr
    json.dump({"model": "Marqo/marqo-fashionSigLIP vision, 선형 분류기(Fashionpedia CC BY 4.0 + clothing-dataset CC0 + 만든 벽)",
               "template": "learned", "noneMargin": round(best_thr, 4),
               "prompts": [{"name": c, "text": "learned:" + c, "vector": [round(float(x), 6) for x in W[i]]}
                           for i, c in enumerate(CLASSES)]},
              open(out_json, "w"))
    print("저장", out_json)


if __name__ == "__main__":
    main()
