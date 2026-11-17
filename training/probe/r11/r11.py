# r11: bigger image models (eval only). Same server boxes; zero-shot text vectors + v4-style probe (text init, lambda 0.3, partial labels).
import json, sys, os, time, random, collections, numpy as np, torch, open_clip
from PIL import Image
import embed_crops as ec
from train_probe import OK
dev = "cuda"
K = {"신발":"shoes","원피스":"dress","바지":"pants","티셔츠":"tshirt","자켓":"jacket","치마":"skirt","셔츠":"shirt","반바지":"shorts","코트":"coat","조끼":"vest","니트":"sweater","가디건":"cardigan","겉옷":"jacket","맨투맨":"sweater","후드":"sweater","블라우스":"shirt","청바지":"pants","패딩":"coat","점퍼":"jacket"}
BIG = {"dress":"dress","shoes":"shoes","tshirt":"top","jacket":"outer","shorts":"bottom","pants":"bottom","skirt":"bottom","shirt":"top","coat":"outer","sweater":"top","cardigan":"outer","vest":"outer"}
FINE = {"1":"short sleeve top","2":"long sleeve top","3":"short sleeve outwear","4":"long sleeve outwear","5":"vest","6":"sling","7":"shorts","8":"trousers","9":"skirt","10":"short sleeve dress","11":"long sleeve dress","12":"vest dress","13":"sling dress"}
G = ["shirt","tshirt","sweater","cardigan","jacket","coat","vest","pants","shorts","skirt","dress","shoes"]
TJ = json.load(open(os.path.expanduser("~/acc/mcp-clothes-state/src/main/resources/models/garment-text-fsiglip.json")))
PN = [p["name"] for p in TJ["prompts"]]; PT = [p["text"] for p in TJ["prompts"]]
def labelset(l):
    if ":" in l: return OK[FINE[l.split(":")[1]]]
    return set(l.split("|")) if "|" in l else OK.get(l, {l})
def rows_of(path, filt, root=""):
    out = []
    for l in open(path):
        r = json.loads(l)
        if r.get("empty") or not r.get("box") or not filt(r): continue
        out.append({"img": (root + r["img"]) if not r["img"].startswith("/") else r["img"], "box": r["box"], "label": r["label"], "conf": r.get("conf") or 0, "det": K.get(r.get("serverName"), r.get("serverName"))})
    return out
R = "/home/fashion-agent/closet-probe/"
EV = {"df2val": (rows_of("df2val2000.jsonl", lambda r: ":" in r["label"]), 0.9), "fpval": (rows_of("nt_fp_val.jsonl", lambda r: r["label"] != "none", R), 0.9)}
okg = lambda r: r["label"] != "none" and labelset(r["label"]) <= set(G)
rnd = random.Random(0)
df2 = rows_of("srv_df2_train.jsonl", okg) + rows_of("srv_df2_train2.jsonl", okg); fp = rows_of("srv_fp_train.jsonl", okg, R) + rows_of("srv_fp_train_rare.jsonl", okg, R)
TR = rnd.sample(df2, 5000) + rnd.sample(fp, 5000)
print("eval", {k: len(v[0]) for k, v in EV.items()}, "train", len(TR), flush=True)
MODELS = [("fashionSigLIP(지금)", "hf-hub:Marqo/marqo-fashionSigLIP", None), ("fashionCLIP", "hf-hub:Marqo/marqo-fashionCLIP", None),
          ("SigLIP2 B/16-256", "ViT-B-16-SigLIP2-256", "webli"), ("SigLIP2 L/16-256", "ViT-L-16-SigLIP2-256", "webli"),
          ("SigLIP2 So400m/14-224", "ViT-SO400M-14-SigLIP2", "webli")]
if len(sys.argv) > 1: MODELS = [m for m in MODELS if m[0] in sys.argv[1:]]
def memavail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
def embed_all(model, pre, rows, size, mean, std):
    out = []; bs = 16
    for i in range(0, len(rows), bs):
        while memavail() < 1200: print("메모리 부족 대기", memavail(), flush=True); time.sleep(30)
        a = []
        for r in rows[i:i+bs]:
            try: im = ec.crop(Image.open(r["img"]).convert("RGB"), r["box"]).resize((size, size), Image.BICUBIC)
            except Exception: im = Image.new("RGB", (size, size), ec.PAD)
            x = (np.asarray(im, np.float32) / 255 - mean) / std; x = x.transpose(2, 0, 1); a += [x, x[:, :, ::-1].copy()]
        with torch.no_grad(): v = model.encode_image(torch.from_numpy(np.stack(a).astype(np.float32)).to(dev).half()).float().cpu().numpy()
        v /= np.linalg.norm(v, axis=1, keepdims=True); v = v[0::2] + v[1::2]; out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out)
def gscores(V, M, names):
    s = V @ M.T; return np.stack([s[:, [i for i, n in enumerate(names) if n == g]].max(1) if g in names else np.full(len(V), -9.0) for g in G], 1)
def evaluate(S, rows, keep):
    S = S.copy(); S[:, G.index("shoes")] = -9
    P = [r["det"] if (r["conf"] >= keep or r["det"] == "shoes") else G[int(s.argmax())] for r, s in zip(rows, S)]
    fine = np.mean([p in labelset(r["label"]) for p, r in zip(P, rows)])
    big = np.mean([BIG.get(p) in {BIG.get(x) for x in labelset(r["label"])} | ({"top", "outer"} if r["label"].endswith((":5", ":6")) else set()) for p, r in zip(P, rows)])
    lo = [i for i, r in enumerate(rows) if not (r["conf"] >= keep or r["det"] == "shoes")]
    low = np.mean([P[i] in labelset(rows[i]["label"]) for i in lo])
    return big, fine, low
def probe(Vtr, Ltr, M0, names, lam=0.3):
    sets = [labelset(l) for l in Ltr]
    keepn = [i for i, n in enumerate(names) if n != "none"]; names = [names[i] for i in keepn]; M0 = M0[keepn]
    X = torch.tensor(Vtr, device=dev); mask = torch.tensor([[g in s for g in G] for s in sets], device=dev)
    cls = torch.tensor([G.index(n) for n in names], device=dev)
    W0 = torch.tensor(M0, device=dev); W = W0.clone().requires_grad_(True); logs = torch.tensor(3.4, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, logs], lr=1e-3); torch.manual_seed(0)
    for ep in range(40):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 256):
            ix = perm[i:i+256]; s = X[ix] @ torch.nn.functional.normalize(W, dim=1).T
            sc = torch.full((len(ix), len(G)), -9.0, device=dev).scatter_reduce(1, cls.expand(len(ix), -1), s, "amax") * logs.exp()
            loss = (torch.logsumexp(sc, 1) - torch.logsumexp(sc.masked_fill(~mask[ix], -1e4), 1)).mean() + lam * ((W - W0) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return torch.nn.functional.normalize(W, dim=1).detach().cpu().numpy(), names
res = {}
for nm, arch, pre in MODELS:
    t0 = time.time(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, _, prep = open_clip.create_model_and_transforms(arch, pretrained=pre) if pre else open_clip.create_model_and_transforms(arch)
    tok = open_clip.get_tokenizer(arch); model = model.to(dev).eval().half()
    size = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
    pc = model.visual.preprocess_cfg; mean, std = np.array(pc["mean"], np.float32), np.array(pc["std"], np.float32); print(nm, "size", size, "mean", mean, flush=True)
    with torch.no_grad(): M = model.encode_text(tok(PT).to(dev)).float().cpu().numpy()
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    # latency: one box, batch 2 (flip), 20 runs
    im = ec.crop(Image.open(EV["df2val"][0][0]["img"]).convert("RGB"), EV["df2val"][0][0]["box"]).resize((size, size))
    x = (np.asarray(im, np.float32) / 255 - mean) / std; x = torch.from_numpy(np.stack([x.transpose(2, 0, 1)] * 2).astype(np.float32)).to(dev).half()
    with torch.no_grad():
        for _ in range(3): model.encode_image(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): model.encode_image(x)
        torch.cuda.synchronize(); lat = (time.time() - t) / 20
    params = sum(p.numel() for p in model.visual.parameters()) / 1e6
    E = {k: embed_all(model, prep, rows, size, mean, std) for k, (rows, keep) in EV.items()}
    Etr = embed_all(model, prep, TR, size, mean, std)
    peak = torch.cuda.max_memory_allocated() / 2**20
    tag = arch.split(chr(58))[-1].replace(chr(47), chr(95)); np.savez("r11_" + tag + ".npz", tr=Etr, **E)
    Mp, pn = probe(Etr, [r["label"] for r in TR], M, PN)
    line = {"model": nm, "img": size, "vision_M_params": round(params), "gpu_ms_per_box(2 flips, fp16)": round(lat * 1000), "gpu_peak_MB": round(peak), "minutes": round((time.time() - t0) / 60, 1)}
    for k, (rows, keep) in EV.items():
        line[k + " zero-shot (big/fine/low)"] = [round(100 * x, 1) for x in evaluate(gscores(E[k], M, PN), rows, keep)]
        line[k + " probe (big/fine/low)"] = [round(100 * x, 1) for x in evaluate(gscores(E[k], Mp, pn), rows, keep)]
    print(json.dumps(line, ensure_ascii=False), flush=True); res[nm] = line
    json.dump(res, open("r11_result.json", "w"), ensure_ascii=False, indent=1)
    del model; torch.cuda.empty_cache()
print("R11 DONE", flush=True)
