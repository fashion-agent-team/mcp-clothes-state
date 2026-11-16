# collect.py fp|cc0 ... : send images to local clothes-state (clothes_inspect), keep server box+embedding, label by IoU with GT.
# fp:  collect.py fp ANN.json IMG_DIR SPLIT N OUT.jsonl
# cc0: collect.py cc0 ROOT N OUT.jsonl
import sys, os, io, json, random, base64, time, threading, urllib.request, csv, concurrent.futures as cf
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_lists import FP, CC0
URL = "http://100.65.166.69:8093/mcp"
loc = threading.local()
def rpc(body, sid=None):
    h = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
    if sid: h["mcp-session-id"] = sid
    r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h), timeout=120)
    t = r.read().decode(); sid = r.headers.get("mcp-session-id") or sid
    if not t.strip(): return None, sid
    if t.lstrip().startswith("{"): return json.loads(t), sid
    return json.loads([l[5:] for l in t.splitlines() if l.startswith("data:")][-1]), sid
def inspect(path, mode):
    if not getattr(loc, "sid", None):
        _, loc.sid = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe-collect", "version": "1"}}})
        rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, loc.sid)
    im = Image.open(path).convert("RGB"); im.thumbnail((1280, 1280)); b = io.BytesIO(); im.save(b, "JPEG", quality=90)
    args = {"imageBase64": base64.b64encode(b.getvalue()).decode(), "mode": mode, "allowGpt": False}
    if mode == "worn": args["view"] = "front"
    t = time.time(); res, _ = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "clothes_inspect", "arguments": args}}, loc.sid)
    return json.loads(res["result"]["content"][0]["text"]).get("garments") or [], time.time() - t
def iou(a, b):
    ix = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); iy = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = ix*iy; u = a[2]*a[3] + b[2]*b[3] - i; return i/u if u > 0 else 0
def mem_ok():
    return int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024 >= 1536
def run(jobs, out, mode):
    done = set()
    if os.path.exists(out):
        for l in open(out): done.add(json.loads(l)["img"])
    jobs = [j for j in jobs if j[0] not in done]; print("todo", len(jobs), "done", len(done), flush=True)
    lock = threading.Lock(); f = open(out, "a"); t0 = time.time(); secs = []
    def work(j):
        while not mem_ok(): time.sleep(30)
        path, gts, split = j
        try: gs, dt = inspect(path, mode)
        except Exception as e: print("err", path, type(e).__name__, flush=True); return
        rows = []
        for g in gs:
            if not g.get("embedding") or not g.get("box"): continue
            bx = [g["box"][k] for k in "xywh"]
            best, lab = 0, None
            for gb, gl in gts:
                v = iou(bx, gb)
                if v > best: best, lab = v, gl
            if mode == "laid":
                big = max(gs, key=lambda q: (q.get("box") or {}).get("w", 0) * (q.get("box") or {}).get("h", 0))
                label = gts[0][1] if g is big else None
            else:
                label = lab if best >= 0.3 else ("none" if best < 0.1 else None)
            if label is None or label == "skip": continue
            rows.append({"img": path, "box": bx, "label": label, "iou": round(best, 3), "split": split, "src": "fp-server" if mode == "worn" else "cc0-server",
                         "serverCat": g.get("category"), "serverName": g.get("name"), "v": g["embedding"]})
        if not rows: rows = [{"img": path, "empty": True}]
        with lock:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush(); secs.append(dt)
            n = len(secs)
            if n % 50 == 0: print(n, len(jobs), f"{n/(time.time()-t0):.2f}장/초, 서버 중간 {sorted(secs)[n//2]:.2f}s", flush=True)
    with cf.ThreadPoolExecutor(2) as ex: list(ex.map(work, jobs))
    print("끝", out, flush=True)
if __name__ == "__main__":
    k = sys.argv[1]; random.seed(11)
    if k == "fp":
        ann, img_dir, split, n, out = sys.argv[2:7]; d = json.load(open(ann))
        cats = {c["id"]: c["name"] for c in d["categories"]}; imgs = {i["id"]: i for i in d["images"]}; by = {}
        for a in d["annotations"]:
            im = imgs[a["image_id"]]; x, y, w, h = a["bbox"]
            by.setdefault(a["image_id"], []).append(([x/im["width"], y/im["height"], w/im["width"], h/im["height"]], FP.get(cats[a["category_id"]], "skip")))
        ids = [i for i in by if any(l != "skip" for _, l in by[i])]; random.shuffle(ids)
        run([(os.path.join(img_dir, imgs[i]["file_name"]), by[i], split) for i in ids[:int(n)]], out, "worn")
    else:
        root, n, out = sys.argv[2:5]; jobs = []
        for r in csv.DictReader(open(os.path.join(root, "images.csv"))):
            lab = CC0.get(r["label"]); p = os.path.join(root, "images", r["image"] + ".jpg")
            if lab and os.path.exists(p): jobs.append((p, [([0, 0, 1, 1], lab)], "test" if random.random() < 0.2 else "train"))
        random.shuffle(jobs); run(jobs[:int(n)], out, "laid")
