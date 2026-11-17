# kind_gpt.py <images dir> <rows.jsonl> <out dir>: ask GPT the garment kind of each low-confidence box, then compare policies vs probe_v4.
import sys, os, io, json, base64, time, collections, urllib.request, concurrent.futures as cf
from PIL import Image
imgdir, src, out = sys.argv[1:4]; KEY = os.environ["OPENAI_API_KEY"]
G = ["shirt", "tshirt", "sweater", "cardigan", "jacket", "coat", "vest", "pants", "shorts", "skirt", "dress"]
OK = {"short sleeve top": {"tshirt", "shirt"}, "long sleeve top": {"tshirt", "shirt", "sweater"},
      "short sleeve outwear": {"jacket", "shirt", "cardigan"}, "long sleeve outwear": {"jacket", "coat", "cardigan"},
      "vest": {"vest", "tshirt"}, "sling": {"tshirt", "vest"}, "shorts": {"shorts"}, "trousers": {"pants"},
      "skirt": {"skirt"}, "short sleeve dress": {"dress"}, "long sleeve dress": {"dress"}, "vest dress": {"dress"}, "sling dress": {"dress"}}
rows = [json.loads(l) for l in open(src)]
PROMPT = ("The photo is a crop around one garment detected in a photo. Which kind is the main garment in the box? "
          "Choose exactly one: shirt (collared/button shirt or blouse), tshirt (t-shirt, tank, camisole, sleeveless top), sweater (knit, sweatshirt, hoodie), "
          "cardigan, jacket, coat, vest (sleeveless outer/waistcoat), pants, shorts, skirt, dress (one-piece dress or jumpsuit covering top and bottom). "
          "If the garment continues past the crop edge, judge the whole garment. Answer only JSON: {\"kind\": \"...\", \"confidence\": 0-1}")
def crop(x):
    im = Image.open(imgdir + "/" + x["img"]).convert("RGB"); W, H = im.size; x1, y1, bw, bh = x["box"]; x2, y2 = x1 + bw, y1 + bh
    pw, ph = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
    c = im.crop((int(max(0, x1 - pw) * W), int(max(0, y1 - ph) * H), int(min(1, x2 + pw) * W), int(min(1, y2 + ph) * H)))
    c.thumbnail((384, 384)); b = io.BytesIO(); c.save(b, "JPEG", quality=85); return base64.b64encode(b.getvalue()).decode()
cp = out + "/cache.json"; cache = json.load(open(cp)) if os.path.exists(cp) else {}
key = lambda x: x["img"] + str(x["box"])
def ask(x):
    if key(x) in cache: return
    body = {"model": "gpt-5.4-mini", "reasoning_effort": "none", "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + crop(x)}}]}]}
    for a in range(4):
        try:
            t = time.time()
            r = json.loads(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode(), headers={"content-type": "application/json", "authorization": "Bearer " + KEY}), timeout=60).read())
            ans = json.loads(r["choices"][0]["message"]["content"]); u = r.get("usage", {})
            cache[key(x)] = {"kind": ans.get("kind"), "conf": float(ans.get("confidence", 0)), "sec": time.time() - t,
                             "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}; return
        except Exception: time.sleep(2 * (a + 1))
t0 = time.time()
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(ask, rows))
json.dump(cache, open(cp, "w")); print(f"GPT {len(cache)}건 {time.time()-t0:.0f}초", flush=True)
ok = lambda p, x: p in OK[x["fine"]]
N_ALL = 1908  # 전체 옷 수(확신도 높은 행 포함); 확신도 높은 쪽 정답 수는 그대로라 차이만 계산
base_ok = sum(x["ok"] for x in rows)
L = [f"확신도 0.9 미만 {len(rows)}개, v4 정답 {base_ok} ({base_ok/len(rows):.1%})", "| 정책 | GPT 호출 비율 | 낮은 확신도 정답률 | 1,908개 전체 기준 변화 |", "|---|---|---|---|"]
for t in [0.005, 0.01, 0.02, 0.03, 0.05, 1]:
    for cmin in [0, 0.7]:
        n = c = 0
        for x in rows:
            g = cache.get(key(x), {}); p = x["base"]
            if x["gap"] < t:
                n += 1
                if g.get("kind") in G and g.get("conf", 0) >= cmin: p = g["kind"]
            c += ok(p, x)
        L.append(f"| 1·2등 차이 < {t}, GPT 확신도 ≥ {cmin} | {n/len(rows):.0%} | {c/len(rows):.1%} | {(c-base_ok)/N_ALL*100:+.1f}%p |")
ga = [ok(cache.get(key(x), {}).get("kind"), x) for x in rows]; L.append(f"GPT 단독: {sum(ga)/len(ga):.1%}")
conf = collections.Counter((x["fine"], cache.get(key(x), {}).get("kind")) for x in rows if not ok(cache.get(key(x), {}).get("kind"), x))
L.append("GPT 오답 상위: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in conf.most_common(8)))
sec = sorted(v["sec"] for v in cache.values()); L.append(f"GPT 중간값 {sec[len(sec)//2]:.2f}초, 토큰 입력 {sum(v['in'] for v in cache.values())} 출력 {sum(v['out'] for v in cache.values())}")
txt = "\n".join(L); open(out + "/result.md", "w").write(txt + "\n"); print(txt)
