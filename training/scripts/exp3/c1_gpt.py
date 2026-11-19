"""Model + "unsure -> GPT" on both pattern tests (test_fp worn, test_sh laid), using probabilities saved by c1_ensemble.py.
Rule (same as pattern_threshold.py / gpt_pattern_eval.py): top1 plain -> plain; top1 pattern with p >= t -> that pattern;
else unsure -> ask GPT (gpt-4.1-mini, same prompt as gpt_pattern_eval.py). Public test photos only (Fashionpedia, Second-Hand).
Answers are cached per photo in <ens dir>/gpt_cache_<model>.json.
usage: OPENAI_API_KEY=... python c1_gpt.py <ens dir> [model] [t]"""
import base64, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

E = Path(sys.argv[1]); MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1-mini"; T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
D = Path(os.environ.get("PATTERN_DS", Path.home() / "clothes-state-train/pattern/ds")); KEY = os.environ["OPENAI_API_KEY"]
DETAIL = os.environ.get("GPT_DETAIL", "low"); GMAX = int(os.environ.get("GPT_MAX", 512))  # high: long side up to GMAX, never upscaled
classes = json.loads((E / "classes.json").read_text()); plain = classes.index("plain")
NAMES = ["plain", "stripe", "check", "floral", "dot", "graphic", "animal", "other"]
PROMPT = ("This photo shows one garment. What is the main print or pattern of its fabric? Choose one: "
          "plain (solid color, no print; a small logo or texture still counts as plain), stripe, check (plaid, gingham, "
          "houndstooth, argyle), floral (flowers or leaves), dot (polka dots or small spots), graphic (letters, numbers, "
          "cartoon or picture print), animal (leopard, zebra, snake skin), other (any other print: abstract, geometric, "
          'paisley, camouflage). Answer only JSON: {"pattern": "<one of the words>", "confidence": 0.0-1.0}')
USAGE = [0, 0]  # prompt, completion tokens of calls made in this run
cache_f = E / (f"gpt_cache_{MODEL}.json" if DETAIL == "low" else f"gpt_cache_{MODEL}_{DETAIL}{GMAX}.json"); cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}

def ask(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((GMAX, GMAX))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"model": MODEL, "max_completion_tokens": 60, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": DETAIL}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                full = json.loads(resp.read()); out = json.loads(full["choices"][0]["message"]["content"])
            u = full.get("usage", {}); USAGE[0] += u.get("prompt_tokens", 0); USAGE[1] += u.get("completion_tokens", 0)
            p = str(out.get("pattern", "")).lower().strip()
            return str(path), (p if p in NAMES else "bad"), round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            err = "error:" + type(e).__name__; time.sleep(2)
    return str(path), err, round(time.time() - t0, 2)

def table(final, ys):
    pl, pa = ys == plain, ys != plain; f = lambda m, c: round(100 * float(c[m].mean()), 1)
    return {"acc": round(float((final == ys).mean()), 4), "plain_to_pattern": f(pl, (final != plain) & (final != -1)),
            "pattern_right": f(pa, final == ys), "pattern_wrong_kind": f(pa, (final != plain) & (final != -1) & (final != ys)),
            "pattern_to_plain": f(pa, final == plain), "unsure": round(100 * float((final == -1).mean()), 1)}

MW = float(os.environ.get("MIX_W", 0.6))
CFGS = {"lora": lambda s: np.load(E / f"{s}_lora_f0.npy"),
        f"lora{MW}+B": lambda s: MW * np.load(E / f"{s}_lora_f0.npy") + (1 - MW) * np.load(E / f"{s}_B_f0.npy")}
res = {}
for s in ("test_fp", "test_sh"):
    paths = sorted((D / s).glob("*/*.jpg")); ys = np.load(E / f"{s}_y.npy"); assert len(paths) == len(ys)
    assert all(classes.index(p.parent.name) == y for p, y in zip(paths, ys))
    for name, get in CFGS.items():
        pr = get(s); top = pr.argmax(1); said = np.where(top == plain, plain, np.where(pr.max(1) >= T, top, -1))
        todo = [str(paths[i]) for i in np.where(said == -1)[0] if str(paths[i]) not in cache]
        with ThreadPoolExecutor(4) as ex:
            for path, a, sec in ex.map(ask, todo): cache[path] = [a, sec]
        cache_f.write_text(json.dumps(cache))
        final = said.copy()
        for i in np.where(said == -1)[0]:
            a = cache[str(paths[i])][0]
            if a in NAMES: final[i] = classes.index(a)
        un = np.where(said == -1)[0]; secs = sorted(cache[str(paths[i])][1] for i in un)
        res[f"{s} {name}"] = {"model_only": table(said, ys), "model_gpt": table(final, ys), "asked": int(len(un)),
                              "asked_share": round(len(un) / len(ys), 3), "gpt_right_on_asked": round(float((final[un] == ys[un]).mean()), 3),
                              "errors": sum(cache[str(paths[i])][0] not in NAMES for i in un),
                              "gpt_median_s": secs[len(secs) // 2] if secs else None,
                              "animal_recall_model_gpt": round(float((final[ys == classes.index("animal")] == classes.index("animal")).mean()), 3) if (ys == classes.index("animal")).any() else None}
        print(s, name, json.dumps(res[f"{s} {name}"]), flush=True)
res["usage_this_run"] = {"prompt_tokens": USAGE[0], "completion_tokens": USAGE[1],
                         "usd_gpt41mini": round(USAGE[0] * 0.4e-6 + USAGE[1] * 1.6e-6, 4)}
print("usage", res["usage_this_run"])
(E / f"gpt_{MODEL}.json").write_text(json.dumps(res, indent=1))
