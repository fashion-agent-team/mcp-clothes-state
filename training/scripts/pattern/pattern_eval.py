"""Evaluate a pattern classifier on test_fp (worn, Fashionpedia) and test_sh (laid, Second-Hand).
Prints per-class recall, macro recall, overall accuracy, plain-vs-patterned accuracy and a confusion matrix; writes JSON.
usage: python -I pattern_eval.py <weights .pt> <dataset dir> <out json>"""
import json, os, sys, time
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

W, D, OUTJ = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m = YOLO(W); names = m.names; inv = {v: k for k, v in names.items()}
res = {}
for split in ("test_fp", "test_sh"):
    paths = [(p, p.parent.name) for p in sorted((D / split).glob("*/*.jpg"))]
    conf = Counter(); t0 = time.time()
    for i in range(0, len(paths), 64):
        chunk = paths[i:i + 64]
        for (p, gt), r in zip(chunk, m.predict([str(p) for p, _ in chunk], imgsz=224, device=os.environ.get("DEVICE", "0"), verbose=False)):
            conf[(gt, names[int(r.probs.top1)])] += 1
    n = sum(conf.values()); ms = (time.time() - t0) * 1000 / max(n, 1)
    per = {}
    for c in names.values():
        tot = sum(v for (g, _), v in conf.items() if g == c)
        if tot: per[c] = {"n": tot, "recall": round(conf[(c, c)] / tot, 3),
                          "precision": round(conf[(c, c)] / max(1, sum(v for (_, p), v in conf.items() if p == c)), 3)}
    acc = sum(v for (g, p), v in conf.items() if g == p) / n
    binacc = sum(v for (g, p), v in conf.items() if (g == "plain") == (p == "plain")) / n
    pat = [c for c in per if c != "plain"]
    res[split] = {"n": n, "acc": round(acc, 3), "macro_recall": round(sum(per[c]["recall"] for c in per) / len(per), 3),
                  "pattern_macro_recall": round(sum(per[c]["recall"] for c in pat) / max(1, len(pat)), 3),
                  "plain_vs_pattern_acc": round(binacc, 3), "ms_per_img": round(ms, 1), "per_class": per,
                  "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())}}
    print(split, json.dumps({k: v for k, v in res[split].items() if k not in ("confusion",)}, ensure_ascii=False))
OUTJ.write_text(json.dumps(res, ensure_ascii=False, indent=1))
