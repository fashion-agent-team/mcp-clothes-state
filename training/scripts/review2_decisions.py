# Second review (2026-11-07) of low-confidence autolabel train photos, pre-labelled with yolo26n_v6_tuned at conf 0.15.
# keep: indices of predicted boxes on a real stain; add: boxes drawn by eye (cx, cy, w, h). Every other photo was dropped
# because its stains could not all be boxed reliably (many faint spots) or the boxes were wrong.
KEEP = {12: [0, 2], 25: [0], 42: [0], 45: [0, 1, 2], 52: [0], 60: [], 64: [1], 69: [0], 72: [], 73: [0], 80: [0], 103: [0]}
ADD = {60: [(0.479, 0.751, 0.035, 0.035)], 72: [(0.46, 0.435, 0.32, 0.26)]}
