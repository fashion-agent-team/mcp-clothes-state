"""1차(v1) 데이터 만들기: Clothes-stain + More clothes 를 클래스 [stain, torn] 하나의 YOLO 데이터로 합친다.

사용: python merge_dataset.py <프로젝트 루트>
  <루트>/raw/stain  = Clothes-stain.v1-version1 (YOLOv11 zip을 푼 것)
  <루트>/raw/more   = More clothes.v1i (YOLOv11 zip을 푼 것)
결과: <루트>/dataset
"""
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
    # stain: 라벨이 비어 있지 않은 사진만 쓴다(빈 라벨 사진은 대부분 얼룩이 있는데 박스가 빠진 것). 0번 그대로.
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
    # More clothes: 1 torn -> 1, 0 good -> 박스는 버리고 사진은 "얼룩 없음"으로 남긴다.
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
