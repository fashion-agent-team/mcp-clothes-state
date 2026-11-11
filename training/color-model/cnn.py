# K-Fashion 정답 상자 옷 조각(64x64)으로 색 20가지 분류 CNN 학습. CPU.
import csv, hashlib, numpy as np, time, sys, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
CLASSES = ['화이트','블랙','베이지','핑크','그레이','스카이블루','네이비','브라운','블루','카키','레드','그린','라벤더','실버','옐로우','와인','민트','퍼플','오렌지','골드']
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rows = list(csv.reader(open('crops/meta.csv')))
X = np.load('img64.npy', mmap_mode='r')
def split(img):
    h = int(hashlib.md5(img.encode()).hexdigest()[:8], 16) % 100
    return 'test' if h < 15 else 'val' if h < 25 else 'train'
sp = np.array([split(r[1]) for r in rows]); y = np.array([CLASSES.index(r[4]) if r[4] in CLASSES else -1 for r in rows])
ok = y >= 0
tr = np.where(ok & (sp=='train'))[0]; va = np.where(ok & (sp=='val'))[0]; te = np.where(ok & (sp=='test'))[0]
Xt = torch.from_numpy(np.ascontiguousarray(X[tr])); yt = torch.from_numpy(y[tr])
class Net(nn.Module):
    def __init__(s, n=20, w=W):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        c = [24*w, 48*w, 96*w, 160*w]
        s.f = nn.Sequential(nn.Conv2d(3, c[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
                            blk(c[0], c[1]), nn.MaxPool2d(2), blk(c[1], c[2]), nn.MaxPool2d(2), blk(c[2], c[3]),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c[3], n))
        s.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
    def forward(s, x):  # x: N,3,64,64 RGB 0..1
        return s.f((x - s.mean) / 0.25)
def batch(idx, aug):
    xb = torch.from_numpy(np.ascontiguousarray(X[idx])) if not isinstance(idx, torch.Tensor) else Xt[idx]
    xb = xb.permute(0, 3, 1, 2).float() / 255
    if aug:
        n = xb.shape[0]
        flip = torch.rand(n) < 0.5; xb[flip] = xb[flip].flip(3)
        xb = F.pad(xb, (4, 4, 4, 4), mode='reflect')
        i, j = np.random.randint(0, 9, 2); xb = xb[:, :, i:i+64, j:j+64]
        gain = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.2 + (torch.rand(n, 3, 1, 1) - 0.5) * 0.06
        xb = (xb * gain).clamp(0, 1)
    return xb
net = Net(); print('params', sum(p.numel() for p in net.parameters()))
BS = 256; steps = EPOCHS * (len(tr) // BS)
opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=5e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
def evaluate(ids):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(ids), 1024): out.append(F.softmax(net(batch(ids[s:s+1024], False)), 1))
    net.train(); return torch.cat(out).numpy()
best = 0; start = 0
import os
CK = f'ckpt_w{W}_e{EPOCHS}.pt'
if os.path.exists(CK):  # 컨테이너가 다시 시작돼도 이어서 학습
    ck = torch.load(CK); net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch'])
    start = ck['ep']; best = ck['best']; torch.set_rng_state(ck['rng']); print('이어서', start, best, flush=True)
for ep in range(start, EPOCHS):
    t = time.time(); perm = torch.randperm(len(tr)); tl = 0
    for k in range(len(tr) // BS):
        b = perm[k*BS:(k+1)*BS]
        loss = F.cross_entropy(net(batch(b, True)), yt[b], label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item()
    pv = evaluate(va); acc = (pv.argmax(1) == y[va]).mean()
    print(f'epoch {ep+1} loss {tl/(len(tr)//BS):.3f} val {acc:.3f} {time.time()-t:.0f}s', flush=True)
    if acc > best: best = acc; torch.save(net.state_dict(), f'cnn_w{W}.pt')
    torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'ep': ep + 1, 'best': best,
                'rng': torch.get_rng_state()}, CK + '.tmp'); os.replace(CK + '.tmp', CK)
net.load_state_dict(torch.load(f'cnn_w{W}.pt'))
pt = evaluate(te); print('test 20색', (pt.argmax(1) == y[te]).mean().round(3))
np.save(f'pred_cnn_w{W}.npy', np.stack([te, pt.argmax(1), pt.max(1)*1000], 1)); np.save(f'prob_cnn_w{W}.npy', pt)
