"""Sweep a 'say a pattern only when sure' rule for the pattern classifier.
Rule: if top1 is a pattern class with prob >= t -> that pattern; else if top1 is a pattern but prob < t -> "unsure"
(server says nothing about pattern / asks), top1 plain -> plain.
For each t prints, on test_fp and test_sh: plain wrongly called a pattern (%), pattern recall (said the right pattern),
pattern said-wrong rate, unsure share. Also writes probs to JSON so the sweep can be redone without the GPU.
usage: python -I pattern_threshold.py <weights .pt> <dataset dir> <out json>"""
import json, sys
from pathlib import Path
from ultralytics import YOLO

W, D, OUTJ = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m = YOLO(W); names = m.names
rows = {}
for split in ("test_fp", "test_sh"):
    paths = [(p, p.parent.name) for p in sorted((D / split).glob("*/*.jpg"))]
    out = []
    for i in range(0, len(paths), 64):
        chunk = paths[i:i + 64]
        for (p, gt), r in zip(chunk, m.predict([str(p) for p, _ in chunk], imgsz=224, device=0, verbose=False)):
            pr = r.probs.data.tolist(); k = max(range(len(pr)), key=pr.__getitem__)
            out.append({"gt": gt, "top": names[k], "p": round(pr[k], 4), "p_plain": round(pr[[i for i in names if names[i] == "plain"][0]], 4)})
    rows[split] = out
OUTJ.write_text(json.dumps(rows))
print("t | split | plain->pattern% | pattern right% | pattern wrong-kind% | pattern->plain% | unsure%")
for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    for split, out in rows.items():
        pl = [o for o in out if o["gt"] == "plain"]; pa = [o for o in out if o["gt"] != "plain"]
        said = lambda o: "plain" if o["top"] == "plain" else (o["top"] if o["p"] >= t else "unsure")
        fp = sum(said(o) not in ("plain", "unsure") for o in pl) / len(pl)
        right = sum(said(o) == o["gt"] for o in pa) / len(pa)
        wrong = sum(said(o) not in ("plain", "unsure", o["gt"]) for o in pa) / len(pa)
        miss = sum(said(o) == "plain" for o in pa) / len(pa)
        uns = sum(said(o) == "unsure" for o in out) / len(out)
        print(f"{t:.1f} | {split} | {fp*100:.1f} | {right*100:.1f} | {wrong*100:.1f} | {miss*100:.1f} | {uns*100:.1f}")
