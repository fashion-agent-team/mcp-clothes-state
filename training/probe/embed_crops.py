#!/usr/bin/env python3
"""옷 박스 사진 → FashionSigLIP 벡터(서버 EmbedInput과 같은 자르기: 사방 3% 넓힘, 회색 정사각형, 224 BICUBIC, 0.5/0.5,
좌우 뒤집기 평균 후 길이 1). 입력: jsonl {"img": 경로, "box": [x,y,w,h] 0~1, "label": 이름, "split": ..., "src": ...}
사용: python3 embed_crops.py list.jsonl out.npz [--onnx model.onnx]   (기본은 open_clip + GPU)
중간에 끊겨도 out.npz.part 에서 이어 함.
"""
import json, math, os, sys, time
import numpy as np
from PIL import Image

PAD = (124, 116, 104)


def crop(img, box, margin=0.03):
    iw, ih = img.size
    x, y, w, h = box
    bw, bh = w * iw, h * ih
    x0 = min(max(int(math.floor(x * iw - bw * margin)), 0), iw - 1)
    y0 = min(max(int(math.floor(y * ih - bh * margin)), 0), ih - 1)
    x1 = min(max(int(math.ceil((x + w) * iw + bw * margin)), x0 + 1), iw)
    y1 = min(max(int(math.ceil((y + h) * ih + bh * margin)), y0 + 1), ih)
    c = img.crop((x0, y0, x1, y1))
    side = max(c.size)
    sq = Image.new("RGB", (side, side), PAD)
    sq.paste(c, ((side - c.size[0]) // 2, (side - c.size[1]) // 2))
    return sq.resize((224, 224), Image.BICUBIC)


def to_arr(im):
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return a.transpose(2, 0, 1)


def main():
    lst, out = sys.argv[1], sys.argv[2]
    onnx = sys.argv[sys.argv.index("--onnx") + 1] if "--onnx" in sys.argv else None
    rows = [json.loads(l) for l in open(lst)]
    part = out + ".part.npz"
    done = 0
    vecs = []
    if os.path.exists(part):
        z = np.load(part)
        vecs = list(z["v"])
        done = len(vecs)
        print("이어 함:", done)
    if onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx, providers=ort.get_available_providers())
        name = sess.get_inputs()[0].name
        dyn = not isinstance(sess.get_inputs()[0].shape[0], int)

        def run(batch):
            if dyn:
                return sess.run(None, {name: batch})[0]
            return np.concatenate([sess.run(None, {name: b[None]})[0] for b in batch])
    else:
        import torch, open_clip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
        model = model.to(dev).eval().half() if dev == "cuda" else model.eval()

        def run(batch):
            with torch.no_grad():
                t = torch.from_numpy(batch).to(dev)
                t = t.half() if dev == "cuda" else t
                return model.encode_image(t).float().cpu().numpy()
    bs = 32
    t0 = time.time()
    for i in range(done, len(rows), bs):
        chunk = rows[i:i + bs]
        arrs = []
        for r in chunk:
            try:
                im = crop(Image.open(r["img"]).convert("RGB"), r["box"])
            except Exception as e:
                print("못 읽음", r["img"], e)
                im = Image.new("RGB", (224, 224), PAD)
            a = to_arr(im)
            arrs += [a, a[:, :, ::-1].copy()]
        v = run(np.stack(arrs).astype(np.float32))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        v = v[0::2] + v[1::2]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        vecs += list(v.astype(np.float32))
        if (i // bs) % 50 == 0:
            np.savez(part, v=np.stack(vecs))
            n = len(vecs)
            print(f"{n}/{len(rows)}  {(n - done) / (time.time() - t0):.1f}장/초", flush=True)
    np.savez(out, v=np.stack(vecs), label=np.array([r["label"] for r in rows]),
             split=np.array([r.get("split", "") for r in rows]), src=np.array([r.get("src", "") for r in rows]))
    if os.path.exists(part):
        os.remove(part)
    print("끝", out, len(vecs))


if __name__ == "__main__":
    main()
