# r12a: So400m memory/latency on Nano (fp16 GPU, visual-only; CPU fp32 single box). Saves text vectors for the probe.
import os, sys, time, json, gc, numpy as np, torch, open_clip
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
def rss(): return int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) // 1024
def guard(tag):
    a = avail(); print(f"[{tag}] MemAvailable {a}MB, 이 프로세스 RSS {rss()}MB, GPU 할당 {torch.cuda.memory_allocated()//2**20}MB", flush=True)
    if a < 1000: print("멈춤: 남은 메모리 1GB 미만", flush=True); sys.exit(2)
TJ = json.load(open(os.path.expanduser("~/acc/mcp-clothes-state/src/main/resources/models/garment-text-fsiglip.json")))
PT = [p["text"] for p in TJ["prompts"]]; out = {}
guard("시작")
t = time.time(); model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP2", pretrained="webli", precision="fp16", device="cuda"); model.eval()
out["load_sec"] = round(time.time() - t, 1); guard("fp16 전체 GPU에 올림"); out["full_fp16_gpu_MB"] = torch.cuda.memory_allocated() // 2**20
tok = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2")
with torch.no_grad(): M = torch.cat([model.encode_text(tok(PT[i:i+8]).cuda()) for i in range(0, len(PT), 8)]).float().cpu().numpy()
np.save("r12_so400m_text.npy", M / np.linalg.norm(M, axis=1, keepdims=True))
vis = model.visual; del model; gc.collect(); torch.cuda.empty_cache(); guard("글 부분 지움(비전만)"); out["visual_fp16_gpu_MB"] = torch.cuda.memory_allocated() // 2**20
for bs in [2, 8]:
    x = torch.randn(bs, 3, 224, 224, device="cuda", dtype=torch.float16); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3): vis(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10): vis(x)
        torch.cuda.synchronize()
    out[f"gpu_fp16_batch{bs}_ms"] = round((time.time() - t) / 10 * 1000); out[f"gpu_fp16_batch{bs}_peak_MB"] = torch.cuda.max_memory_allocated() // 2**20
    guard(f"GPU 배치 {bs}")
# CPU fp32 (서버처럼 CPU면): 비전만, 1장
vis = vis.float().cpu(); torch.cuda.empty_cache(); gc.collect(); guard("비전 fp32 CPU")
torch.set_num_threads(4); x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    vis(x); t = time.time()
    for _ in range(3): vis(x)
out["cpu_fp32_1img_ms(4 threads)"] = round((time.time() - t) / 3 * 1000); out["cpu_fp32_rss_MB"] = rss(); guard("CPU 측정 끝")
# int8 dynamic quant (CPU, Linear만)
try:
    q = torch.ao.quantization.quantize_dynamic(vis, {torch.nn.Linear}, dtype=torch.qint8); del vis; gc.collect()
    with torch.no_grad():
        q(x); t = time.time()
        for _ in range(3): q(x)
    out["cpu_int8_1img_ms(4 threads)"] = round((time.time() - t) / 3 * 1000); out["cpu_int8_rss_MB"] = rss(); guard("int8")
except Exception as e: out["int8_error"] = repr(e)[:200]
print(json.dumps(out, ensure_ascii=False), flush=True); json.dump(out, open("r12a_result.json", "w"), ensure_ascii=False, indent=1)
