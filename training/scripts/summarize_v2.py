"""Write a markdown table of v2 test results (and old v1 models on the v2 test split) from the val logs."""
import csv, re, sys
from pathlib import Path

R = Path(sys.argv[1])
LOGS, RUNS = R / "logs", R / "runs"
rows = [("yolo26n v1 (old, 30ep)", "test_yolo26n_v1model_on_v2test.log", None),
        ("yolo26s v1 (old, 30ep)", "test_yolo26s_v1model_on_v2test.log", None)]
rows += [(f"{m}_v2", f"test_{m}_v2.log", f"{m}_v2") for m in ("yolo26n", "yolo26s", "yolo11n", "yolo11s")]

def parse(log):
    t = log.read_text(errors="ignore").replace("\r", "\n") if log.exists() else ""
    res = {}
    for c in ("all", "stain", "torn"):
        m = re.findall(rf"^\s+{c}\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t, re.M)
        if m: res[c] = m[-1]
    sp = re.findall(r"Speed: ([\d.]+)ms preprocess, ([\d.]+)ms inference, [\d.]+ms loss, ([\d.]+)ms postprocess", t)
    ms = sum(map(float, sp[-1])) if sp else None
    return res, ms, (sp[-1][1] if sp else None)

def train_info(run):
    f = RUNS / run / "results.csv" if run else None
    if not f or not f.exists(): return "-", "-"
    r = [{k.strip(): v for k, v in x.items()} for x in csv.DictReader(f.open())]
    best = max(r, key=lambda x: float(x["metrics/mAP50-95(B)"]))
    t = float(r[-1]["time"])
    return f"{len(r)} ep, {t/60:.0f} min ({t/len(r):.0f} s/ep)", best["epoch"]

print("# clothes-state v2 results (test split of dataset_v2)\n")
print("| model | class | P | R | mAP50 | mAP50-95 | ms/img total (inference) | training | best epoch |")
print("|---|---|---|---|---|---|---|---|---|")
for name, log, run in rows:
    res, ms, inf = parse(LOGS / log)
    tr, be = train_info(run)
    for c in ("all", "stain", "torn"):
        v = res.get(c, ("-",) * 4)
        first = c == "all"
        spd = (f"{ms:.1f} ({inf})" if ms else "-") if first else ""
        print(f"| {name if first else ''} | {c} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {spd} | {tr if first else ''} | {be if first else ''} |")
