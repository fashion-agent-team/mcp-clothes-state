"""K-Fashion (AI Hub) Validation -> pattern8 crops for training only (never commit: AI Hub data, no redistribution).
Per garment box (아우터/상의/하의/원피스) with exactly one mapped 프린트 class. All patterned garments + plain at 1.5x their count.
Split by photo: train 90% / val 10%. Crops saved with long side 320 as <out>/<split>/<class>/<photo>_<part><i>.jpg
usage: python kf_pattern.py <K-Fashion Validation dir> <out dir> [workers]"""
import collections, hashlib, io, json, multiprocessing as mp, os, random, sys, zipfile
from pathlib import Path
from PIL import Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); NW = int(sys.argv[3]) if len(sys.argv) > 3 else 6
MAP = {"무지": "plain", "스트라이프": "stripe", "체크": "check", "깅엄": "check", "하운즈투스": "check", "아가일": "check",
       "플로럴": "floral", "도트": "dot", "그래픽": "graphic", "레터링": "graphic", "캐릭터": "graphic",
       "호피": "animal", "지브라": "animal", "뱀피": "animal", "레오파드": "animal",
       "페이즐리": "other", "카무플라쥬": "other", "카모플라쥬": "other", "지그재그": "other", "하트": "other", "타이다이": "other",
       "해골": "other", "패치워크": "other"}
SKIP = {"믹스", "그라데이션"}  # mixed / gradient: not one pattern

def norm(p): return p.replace("\xa0", "").replace(" ", "")

def scan():
    z = zipfile.ZipFile(SRC / "라벨링데이터.zip"); rows = []; seen = collections.Counter(); bad = collections.Counter()
    for n in z.namelist():
        if not n.endswith(".json"): continue
        d = json.loads(z.read(n)); img = f"{d['이미지 정보']['이미지 식별자']}.jpg"; ds = d["데이터셋 정보"]["데이터셋 상세설명"]
        for part, items in ds.get("라벨링", {}).items():
            if not isinstance(items, list): continue
            boxes = ds.get("렉트좌표", {}).get(part, [])
            for i, it in enumerate(items):
                if not isinstance(it, dict) or not it.get("프린트"): continue
                ps = {norm(p) for p in it["프린트"]}; [seen.update([p]) for p in ps]
                cls = {MAP.get(p) for p in ps if p not in SKIP}
                if ps & SKIP or None in cls or len(cls) != 1: bad[",".join(sorted(ps))] += 1; continue
                b = boxes[i] if i < len(boxes) else {}
                if not b or b.get("가로", 0) < 48 or b.get("세로", 0) < 48: bad["small_or_no_box"] += 1; continue
                rows.append((img, part, i, cls.pop(), b["X좌표"], b["Y좌표"], b["가로"], b["세로"]))
    return rows, seen, bad

def work(args):
    wid, jobs = args; z = zipfile.ZipFile(SRC / "원천데이터.zip")
    byname = {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}
    done = miss = 0
    for img, items in jobs:
        n = byname.get(img)
        if not n: miss += len(items); continue
        try: im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        except Exception: miss += len(items); continue
        W, H = im.size
        for split, part, i, cls, x, y, w, h in items:
            c = im.crop((max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h)))); c.thumbnail((320, 320))
            d = OUT / split / cls; d.mkdir(parents=True, exist_ok=True)
            c.save(d / f"kf_{Path(img).stem}_{part}{i}.jpg", quality=92); done += 1
    return done, miss

if __name__ == "__main__":
    rows, seen, bad = scan()
    print("print values seen:", seen.most_common()); print("skipped:", bad.most_common(15))
    random.seed(0); pat = [r for r in rows if r[3] != "plain"]; pl = [r for r in rows if r[3] == "plain"]
    random.shuffle(pl); keep = pat + pl[:int(len(pat) * 1.5)]
    split = lambda img: "val" if int(hashlib.md5(img.encode()).hexdigest(), 16) % 10 == 0 else "train"
    by = collections.defaultdict(list)
    for r in keep: by[r[0]].append((split(r[0]),) + r[1:])
    print("counts", {s: collections.Counter(c for r in keep if split(r[0]) == s for c in [r[3]]) for s in ("train", "val")})
    jobs = list(by.items()); chunks = [(k, jobs[k::NW]) for k in range(NW)]
    with mp.Pool(NW) as p: res = p.map(work, chunks)
    print("saved", sum(r[0] for r in res), "missing", sum(r[1] for r in res))
