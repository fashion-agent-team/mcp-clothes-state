"""Pick threshold t, mix weight w (LoRA vs pattern B) and when to follow GPT (never / if in model top-2 / top-3 / always)
on the Fashionpedia-style val split only (pattern/ds/val, 2,376 photos), then score test_fp and test_sh once with that choice.
GPT answers come from / go to the same cache as c1_gpt.py. Stops before asking if the estimated cost passes MAX_USD.
usage: OPENAI_API_KEY=... PATTERN_DS=... MAX_USD=0.5 python c1_select.py <ens dir>"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], "gpt-4.1-mini", "0.7"]
src = (Path(__file__).parent / "c1_gpt.py").read_text(); ns = {"__file__": __file__}
exec(src[:src.index("CFGS = {")].replace("MW = float", "_MW = float"), ns)  # reuse ask(), cache, NAMES, classes, D, E
ask, cache, cache_f, NAMES, classes, D, E, USAGE = (ns[k] for k in ("ask", "cache", "cache_f", "NAMES", "classes", "D", "E", "USAGE"))
WS = [float(x) for x in os.environ.get("W_LIST", "0.5,0.6,0.7,0.8,1.0").split(",")]
plain = classes.index("plain"); MAX = float(os.environ.get("MAX_USD", 0.5)); USD_PER_CALL = float(os.environ.get("USD_PER_CALL", 0.00013))

def load(s, w): return w * np.load(E / f"{s}_lora_f0.npy") + (1 - w) * np.load(E / f"{s}_B_f0.npy")
def paths(s): return sorted((D / s).glob("*/*.jpg"))
def fill(s, idx):
    ps = paths(s); todo = [str(ps[i]) for i in idx if str(ps[i]) not in cache]
    est = len(todo) * USD_PER_CALL; print(s, "new GPT calls", len(todo), f"~${est:.3f}")
    if est > MAX: sys.exit(f"stop: estimated ${est:.2f} > ${MAX}")
    with ThreadPoolExecutor(4) as ex:
        for p, a, sec in ex.map(ask, todo): cache[p] = [a, sec]
    cache_f.write_text(json.dumps(cache))
def final(s, t, w, rule):
    pr = load(s, w); top = pr.argmax(1); ps = paths(s); f = top.copy()
    if rule == "never": return f, 0
    idx = np.where((top != plain) & (pr.max(1) < t))[0]; order = np.argsort(-pr, 1)
    for i in idx:
        a = cache.get(str(ps[i]), ["x"])[0]
        if a not in NAMES: continue
        g = classes.index(a)
        if rule == "always" or (rule == "top2" and g in order[i, :2]) or (rule == "top3" and g in order[i, :3]): f[i] = g
    return f, len(idx)

ys = {s: np.load(E / f"{s}_y.npy") for s in ("val", "test_fp", "test_sh")}
need = set()
for w in WS:
    pr = load("val", w); top = pr.argmax(1); need |= set(np.where((top != plain) & (pr.max(1) < 0.8))[0].tolist())
fill("val", sorted(need))
grid = []
for w in WS:
    for rule in ("never", "top2", "top3", "always"):
        for t in ((0.7,) if rule == "never" else (0.5, 0.6, 0.7, 0.8)):
            f, n = final("val", t, w, rule); grid.append(((f == ys["val"]).mean(), n / len(f), w, rule, t))
grid.sort(key=lambda g: (-g[0], g[1]))  # best val acc, then fewer GPT calls
print("val top 8:"); [print(f"  acc {a:.4f} asked {q:.3f} w={w} rule={r} t={t}") for a, q, w, r, t in grid[:8]]
acc, q, w, rule, t = grid[0]
res = {"chosen": {"w": w, "rule": rule, "t": t, "val_acc": round(float(acc), 4), "val_asked": round(float(q), 3)}, "test": {}}
for s in ("test_fp", "test_sh"):
    pr = load(s, w); top = pr.argmax(1); fill(s, np.where((top != plain) & (pr.max(1) < t))[0])
    f, n = final(s, t, w, rule); a = ys[s] == classes.index("animal")
    res["test"][s] = {"acc": round(float((f == ys[s]).mean()), 4), "asked_share": round(n / len(f), 3),
                      "model_only_top1": round(float((top == ys[s]).mean()), 4),
                      "animal": round(float((f[a] == ys[s][a]).mean()), 3) if a.any() else None}
    print(s, res["test"][s])
res["usage_this_run"] = {"prompt_tokens": USAGE[0], "completion_tokens": USAGE[1], "usd": round(USAGE[0] * 0.4e-6 + USAGE[1] * 1.6e-6, 4)}
print("usage", res["usage_this_run"]); (E / os.environ.get("SELECT_OUT", "select.json")).write_text(json.dumps(res, indent=1))
