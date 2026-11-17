# r12b <visual|text>: build only one So400m tower in fp16 and load its weights from the safetensors (lazy), under MemoryMax 2G.
import os, sys, time, json, gc, numpy as np, torch, open_clip
from safetensors import safe_open
from open_clip.model import _build_vision_tower, _build_text_tower
CK = os.path.expanduser("~/.cache/huggingface/hub/models--timm--ViT-SO400M-14-SigLIP2/snapshots/f9e2bfdc4807cd0fbadf67301e45193d6f330c20/open_clip_model.safetensors")
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
def rss(): return int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) // 1024
def guard(tag):
    a = avail(); print(f"[{tag}] MemAvailable {a}MB, RSS {rss()}MB, GPU {torch.cuda.memory_allocated()//2**20}MB", flush=True)
    if a < 1000: print("멈춤: 남은 메모리 1GB 미만", flush=True); sys.exit(2)
cfg = open_clip.get_model_config("ViT-SO400M-14-SigLIP2"); part = sys.argv[1]; out = {}
guard("시작")
t = time.time()
with torch.device("meta"):
  if part == "visual":
    m = _build_vision_tower(cfg["embed_dim"], cfg["vision_cfg"]); pre = "visual."
  else:
    m = _build_text_tower(cfg["embed_dim"], cfg["text_cfg"]); pre = "text."
guard("빈 탑 만듦(meta)")
sd = {}
if part == "text":
    TJ = json.load(open(os.path.expanduser("~/acc/mcp-clothes-state/src/main/resources/models/garment-text-fsiglip.json")))
    tok = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2"); TOK = tok([p["text"] for p in TJ["prompts"]]); U = sorted(set(TOK.flatten().tolist()))
    remap = {u: i for i, u in enumerate(U)}; TOK2 = TOK.clone().apply_(lambda v: remap[v]); print("쓰는 토큰", len(U), flush=True)
    with torch.device("meta"): m.token_embedding = torch.nn.Embedding(len(U), cfg["text_cfg"]["width"])
with safe_open(CK, "pt") as f:
    for k in f.keys():
        if not k.startswith(pre): continue
        if part == "text" and k == "text.token_embedding.weight":
            sd[k[len(pre):]] = f.get_slice(k)[torch.tensor(U)].half().cuda() if False else torch.stack([f.get_slice(k)[int(u):int(u)+1][0] for u in U]).half().cuda()
        else: sd[k[len(pre):]] = f.get_tensor(k).half().cuda()
miss = m.load_state_dict(sd, strict=False, assign=True); del sd; gc.collect(); print("missing", len(miss.missing_keys), "unexpected", len(miss.unexpected_keys), miss.missing_keys[:3], flush=True)
m = m.eval().cuda(); gc.collect(); out["load_sec"] = round(time.time() - t, 1); out[part + "_fp16_gpu_MB"] = torch.cuda.memory_allocated() // 2**20; guard("GPU에 올림")
if part == "text":
    TJ = json.load(open(os.path.expanduser("~/acc/mcp-clothes-state/src/main/resources/models/garment-text-fsiglip.json")))
    tok = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2")
    with torch.no_grad(): M = torch.cat([m(TOK2[i:i+8].cuda()) for i in range(0, 35, 8)]).float().cpu().numpy()
    np.save("r12_so400m_text.npy", M / np.linalg.norm(M, axis=1, keepdims=True)); print("text saved", M.shape, flush=True)
else:
    # sanity: same embedding as r11 for first df2val box
    for bs in [2, 8]:
        x = torch.randn(bs, 3, 224, 224, device="cuda", dtype=torch.float16); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(3): m(x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10): m(x)
            torch.cuda.synchronize()
        out[f"gpu_fp16_batch{bs}_ms"] = round((time.time() - t) / 10 * 1000); out[f"gpu_fp16_batch{bs}_peak_MB"] = torch.cuda.max_memory_allocated() // 2**20; guard(f"배치 {bs}")
    out["process_rss_MB"] = rss()
print(json.dumps(out, ensure_ascii=False), flush=True); json.dump(out, open(f"r12b_{part}.json", "w"), ensure_ascii=False, indent=1)
