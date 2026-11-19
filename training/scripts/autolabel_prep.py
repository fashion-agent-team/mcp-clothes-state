"""Collect Clothes-stain images whose label file is empty, one per original (augmented copies share a stem)."""
import re, sys
from pathlib import Path
R = Path(sys.argv[1]); src = R / "raw/stain"; out = R / "autolabel/images"
out.mkdir(parents=True, exist_ok=True)
seen, n_empty = set(), 0
for split in ("train", "valid", "test"):
    for img in sorted((src / split / "images").iterdir()):
        lab = src / split / "labels" / (img.stem + ".txt")
        if lab.exists() and lab.read_text().strip():
            continue
        n_empty += 1
        stem = re.sub(r"\.rf\..*$", "", img.name).lower()
        if stem in seen:
            continue
        seen.add(stem)
        link = out / f"{split}__{img.name}"
        if not link.exists():
            link.symlink_to(img)
print(f"empty-label images: {n_empty}, unique originals linked: {len(seen)}")
