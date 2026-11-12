"""When pattern8 B is unsure (top1 is a pattern with prob < t), how well does GPT name the pattern?
Public data only (Second-Hand Fashion official test, CC BY 4.0, laid-flat garments). Images are sent to OpenAI;
nothing is written except the answers JSON.
Rows of thr_B.json[test_sh] follow sorted(<test_sh dir>/*/*.jpg), the same order pattern_threshold.py used.
usage: OPENAI_API_KEY=... python -I gpt_pattern_eval.py <thr_B.json> <test_sh dir> <out dir> [model] [t]"""
import base64, io, json, os, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

THR, D, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MODEL = sys.argv[4] if len(sys.argv) > 4 else "gpt-4.1-mini"
T = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
KEY = os.environ["OPENAI_API_KEY"]
NAMES = ["plain", "stripe", "check", "floral", "dot", "graphic", "animal", "other"]
PROMPT = ("This photo shows one garment. What is the main print or pattern of its fabric? Choose one: "
          "plain (solid color, no print; a small logo or texture still counts as plain), stripe, check (plaid, gingham, "
          "houndstooth, argyle), floral (flowers or leaves), dot (polka dots or small spots), graphic (letters, numbers, "
          "cartoon or picture print), animal (leopard, zebra, snake skin), other (any other print: abstract, geometric, "
          'paisley, camouflage). Answer only JSON: {"pattern": "<one of the words>", "confidence": 0.0-1.0}')

rows = json.loads(THR.read_text())["test_sh"]
paths = sorted(D.glob("*/*.jpg"))
assert len(paths) == len(rows), (len(paths), len(rows))
for p, r in zip(paths, rows):
    assert p.parent.name == r["gt"], (p, r)


def said(r):
    return "plain" if r["top"] == "plain" else (r["top"] if r["p"] >= T else None)


unsure = [i for i, r in enumerate(rows) if said(r) is None]
print(f"t={T}: unsure {len(unsure)} / {len(rows)}", Counter(rows[i]["gt"] for i in unsure))


def ask(i):
    im = Image.open(paths[i]).convert("RGB"); im.thumbnail((512, 512))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"model": MODEL, "max_completion_tokens": 60, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": "low"}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(json.loads(resp.read())["choices"][0]["message"]["content"])
        p = str(out.get("pattern", "")).lower().strip()
        return i, (p if p in NAMES else "bad"), float(out.get("confidence", 0)), round(time.time() - t0, 2)
    except Exception as e:  # noqa: BLE001
        return i, "error:" + type(e).__name__, 0.0, round(time.time() - t0, 2)


with ThreadPoolExecutor(4) as ex:
    ans = {i: (p, c, s) for i, p, c, s in ex.map(ask, unsure)}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"pattern_answers_{MODEL}.json").write_text(json.dumps(
    [{"file": paths[i].name, "gt": rows[i]["gt"], "model_top": rows[i]["top"], "model_p": rows[i]["p"],
      "gpt": a[0], "gpt_conf": a[1], "sec": a[2]} for i, a in ans.items()], indent=1))

ok = sum(ans[i][0] == rows[i]["gt"] for i in unsure)
model_ok = sum(rows[i]["top"] == rows[i]["gt"] for i in unsure)
secs = sorted(a[2] for a in ans.values())
print(f"{MODEL}: unsure set right {ok}/{len(unsure)} (model top1 would be {model_ok}/{len(unsure)}), "
      f"errors {sum(a[0].startswith('error') or a[0] == 'bad' for a in ans.values())}, "
      f"median {secs[len(secs) // 2]}s, p90 {secs[int(len(secs) * 0.9)]}s, over 3s {sum(s > 3 for s in secs)}")


def summary(final):
    pl = [i for i, r in enumerate(rows) if r["gt"] == "plain"]; pa = [i for i, r in enumerate(rows) if r["gt"] != "plain"]
    f = lambda xs, cond: round(100 * sum(cond(i) for i in xs) / len(xs), 1)
    return {"plain->pattern%": f(pl, lambda i: final[i] not in ("plain", None)),
            "pattern right%": f(pa, lambda i: final[i] == rows[i]["gt"]),
            "pattern wrong-kind%": f(pa, lambda i: final[i] not in ("plain", None, rows[i]["gt"])),
            "pattern->plain%": f(pa, lambda i: final[i] == "plain"),
            "unsure%": f(range(len(rows)), lambda i: final[i] is None),
            "acc%": f(range(len(rows)), lambda i: final[i] == rows[i]["gt"])}


base = [said(r) for r in rows]
withgpt = [ans[i][0] if i in ans and ans[i][0] in NAMES else base[i] for i in range(len(rows))]
print("model only (t=%.1f):" % T, summary(base))
print("model + GPT          :", summary(withgpt))
print("confusion on unsure set (gt->gpt):", sorted(Counter(f"{rows[i]['gt']}->{ans[i][0]}" for i in unsure).items()))
