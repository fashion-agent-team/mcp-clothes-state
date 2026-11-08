"""Merge stain + More clothes into one YOLO dataset with classes [stain, torn]."""
import shutil, sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1]).expanduser()
raw, out = root / "raw", root / "dataset"
if out.exists():
    shutil.rmtree(out)
stats = Counter()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def label_for(img):
    return img.parent.parent / "labels" / (img.stem + ".txt")

for split in ("train", "valid", "test"):
    (out / split / "images").mkdir(parents=True)
    (out / split / "labels").mkdir(parents=True)
    # stain: keep only images with non-empty labels; class 0 stays 0
    for img in sorted((raw / "stain" / split / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lab = label_for(img)
        lines = [l for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        if not lines:
            stats[f"{split}/stain_skipped_empty"] += 1
            continue
        assert all(l.split()[0] == "0" for l in lines), lab
        shutil.copy2(img, out / split / "images" / f"stain_{img.name}")
        (out / split / "labels" / f"stain_{img.stem}.txt").write_text("\n".join(lines) + "\n")
        stats[f"{split}/stain_imgs"] += 1
        stats[f"{split}/stain_boxes"] += len(lines)
    # more clothes: 1 torn -> 1, 0 good -> dropped (image kept as negative)
    for img in sorted((raw / "more" / split / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lab = label_for(img)
        lines = [l for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        kept = []
        for l in lines:
            c = l.split()[0]
            if c == "1":
                kept.append(l)
            elif c == "0":
                stats[f"{split}/good_boxes_dropped"] += 1
            else:
                raise ValueError(f"unexpected class {c} in {lab}")
        shutil.copy2(img, out / split / "images" / f"torn_{img.name}")
        (out / split / "labels" / f"torn_{img.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
        stats[f"{split}/torn_imgs_with_torn" if kept else f"{split}/torn_imgs_negative"] += 1
        stats[f"{split}/torn_boxes"] += len(kept)

(out / "data.yaml").write_text(
    f"path: {out}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ['stain', 'torn']\n"
)
for k in sorted(stats):
    print(k, stats[k])
