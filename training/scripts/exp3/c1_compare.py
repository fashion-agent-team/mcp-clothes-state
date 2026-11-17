"""Compare GPT set-ups for unsure pattern crops on the Fashionpedia-style val split (pattern/ds/val) only, then score the
best one once on test_fp / test_sh. Mix fixed: 224 K-Fashion LoRA x W + pattern8 B x (1-W).
Variants: model (gpt-4.1-mini / gpt-5.4-mini), detail high with long side <= 768 (never upscaled), hint (the model's top-3
candidates in the prompt) on/off; ask set = top1 pattern with p < t, optionally also top1 plain with p < PT; follow GPT
never / if in top2 / top3 / always. Stops when the spend estimate passes MAX_USD (conservative prices below).
usage: OPENAI_API_KEY=... PATTERN_DS=... python c1_compare.py <ens dir>"""
import base64, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

E = Path(sys.argv[1]); D = Path(os.environ["PATTERN_DS"]); KEY = os.environ["OPENAI_API_KEY"]
W = float(os.environ.get("MIX_W", 0.7)); MAX = float(os.environ.get("MAX_USD", 0.85))
PRICE = {"gpt-4.1-mini": (0.4e-6, 1.6e-6), "gpt-5.4-mini": (2e-6, 8e-6)}  # $/token in, out (5.4-mini: deliberately high guess)
classes = json.loads((E / "classes.json").read_text()); plain = classes.index("plain")
NAMES = ["plain", "stripe", "check", "floral", "dot", "graphic", "animal", "other"]
BASE = ("This photo shows one garment. What is the main print or pattern of its fabric? Choose one: "
        "plain (solid color, no print; a small logo or texture still counts as plain), stripe, check (plaid, gingham, "
        "houndstooth, argyle), floral (flowers or leaves), dot (polka dots or small spots), graphic (letters, numbers, "
        "cartoon or picture print), animal (leopard, zebra, snake skin), other (any other print: abstract, geometric, "
        "paisley, camouflage). ")
END = 'Answer only JSON: {"pattern": "<one of the words>", "confidence": 0.0-1.0}'
SPENT = [0.0]

def cache_path(model, hint): return E / f"gpt_cache_{model}_high768{'_hint' if hint else ''}.json"
def ask(job):
    path, model, hint = job
    im = Image.open(path).convert("RGB"); im.thumbnail((768, 768))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); b64 = base64.b64encode(buf.getvalue()).decode()
    text = BASE + (f"An image classifier's top guesses, most likely first: {', '.join(hint)}. They may be wrong. " if hint else "") + END
    body = {"model": model, "max_completion_tokens": 60 if model == "gpt-4.1-mini" else 600, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": "high"}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time(); err = "error"
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                full = json.loads(resp.read())
            u = full.get("usage", {}); pi, po = PRICE[model]; SPENT[0] += u.get("prompt_tokens", 0) * pi + u.get("completion_tokens", 0) * po
            p = str(json.loads(full["choices"][0]["message"]["content"]).get("pattern", "")).lower().strip()
            return path, (p if p in NAMES else "bad"), round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            err = "error:" + type(e).__name__; time.sleep(2)
    return path, err, round(time.time() - t0, 2)

def load(s): return W * np.load(E / f"{s}_lora_f0.npy") + (1 - W) * np.load(E / f"{s}_B_f0.npy")
P = {s: sorted((D / s).glob("*/*.jpg")) for s in ("val", "test_fp", "test_sh")}
Y = {s: np.load(E / f"{s}_y.npy") for s in P}
def ask_set(s, t, pt):
    pr = load(s); top = pr.argmax(1); mx = pr.max(1)
    return np.where(((top != plain) & (mx < t)) | ((top == plain) & (mx < pt)))[0]
def fill(s, idx, model, hint):
    cf = cache_path(model, hint); cache = json.loads(cf.read_text()) if cf.exists() else {}
    pr = load(s); order = np.argsort(-pr, 1)
    jobs = [(str(P[s][i]), model, [classes[k] for k in order[i, :3]] if hint else None) for i in idx if str(P[s][i]) not in cache]
    print(f"{s} {model} hint={hint}: {len(jobs)} new calls, spent so far ${SPENT[0]:.3f}", flush=True)
    for k in range(0, len(jobs), 40):
        if SPENT[0] > MAX: sys.exit(f"stop: spent ${SPENT[0]:.2f} > ${MAX}")
        with ThreadPoolExecutor(4) as ex:
            for p, a, sec in ex.map(ask, jobs[k:k + 40]): cache[p] = [a, sec]
        cf.write_text(json.dumps(cache))
    return cache
def final(s, cache, t, pt, rule):
    pr = load(s); top = pr.argmax(1); order = np.argsort(-pr, 1); f = top.copy(); idx = ask_set(s, t, pt)
    if rule != "never":
        for i in idx:
            a = cache.get(str(P[s][i]), ["x"])[0]
            if a not in NAMES: continue
            g = classes.index(a)
            if rule == "always" or (rule == "top2" and g in order[i, :2]) or (rule == "top3" and g in order[i, :3]): f[i] = g
    return f, idx

VARIANTS = [("gpt-4.1-mini", False), ("gpt-4.1-mini", True), ("gpt-5.4-mini", False)]
grid = []
for model, hint in VARIANTS:
    cache = fill("val", ask_set("val", 0.8, 0.7), model, hint)
    for pt in (0.0, 0.6, 0.7):
        for t in (0.5, 0.6, 0.7, 0.8):
            for rule in ("top2", "top3", "always"):
                f, idx = final("val", cache, t, pt, rule)
                grid.append(((f == Y["val"]).mean(), len(idx) / len(f), model, hint, t, pt, rule))
f0 = load("val").argmax(1); print("val no GPT", round(float((f0 == Y["val"]).mean()), 4))
grid.sort(key=lambda g: (-g[0], g[1]))
print("val top 10:"); [print(f"  acc {a:.4f} asked {q:.3f} {m} hint={h} t={t} plain<{pt} rule={r}") for a, q, m, h, t, pt, r in grid[:10]]
for m, h in VARIANTS:
    b = max((g for g in grid if g[2] == m and g[3] == h), key=lambda g: g[0]); print(f"  best {m} hint={h}: {b[0]:.4f} (t={b[4]} plain<{b[5]} {b[6]})")
acc, q, model, hint, t, pt, rule = grid[0]
res = {"chosen": {"model": model, "hint": hint, "t": t, "plain_t": pt, "rule": rule, "val_acc": round(float(acc), 4), "val_asked": round(float(q), 3)}, "test": {}}
for s in ("test_fp", "test_sh"):
    idx = ask_set(s, t, pt); cache = fill(s, idx, model, hint); f, idx = final(s, cache, t, pt, rule)
    top = load(s).argmax(1); g = [cache.get(str(P[s][i]), ["x"])[0] for i in idx]; secs = sorted(cache[str(P[s][i])][1] for i in idx if str(P[s][i]) in cache)
    res["test"][s] = {"acc": round(float((f == Y[s]).mean()), 4), "mix_only": round(float((top == Y[s]).mean()), 4), "asked_share": round(len(idx) / len(f), 3),
                      "gpt_right_on_asked": round(float(np.mean([a == classes[Y[s][i]] for a, i in zip(g, idx)])), 3),
                      "model_right_on_asked": round(float((top[idx] == Y[s][idx]).mean()), 3),
                      "gpt_median_s": secs[len(secs) // 2] if secs else None}
    print(s, res["test"][s], flush=True)
res["spent_usd_estimate"] = round(SPENT[0], 3); print("spent (estimate, conservative for 5.4-mini)", res["spent_usd_estimate"])
(E / "compare.json").write_text(json.dumps(res, indent=1))
