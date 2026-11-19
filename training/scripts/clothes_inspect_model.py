"""Model side of clothes_inspect: find stains/tears and say where they are.

inspect(image_bgr, mode="laid"|"worn", view=None|"front"|"side"|"back", mirrored=False) -> dict

- One detector, yolo26n_v6_tuned (trained from scratch with clean-clothing negatives, tears weighted 2x and the
  hyperparameters from the overnight tuner): finds far more tears than v5, no confident false alarms on clean
  clothes. A separate tear model can still be passed as tear_weights.

- Defects scoring >= TELL are reported directly (status "confirmed"); TELL > score >= ASK are uncertain and are
  sent to a friend/family member to check (status "ask", numbered on one combined close-up image with faces blurred);
  below ASK they are dropped.
- mode=laid: `where` is photo-based (photo left = user's left, the garment is laid down and shot from above).
- mode=worn: `where` is body-based (the wearer's own left/right), from a pose model's keypoints. Pose keypoints are
  labelled by the person's anatomical side, so front/back/side views need no special casing, but a mirrored (selfie)
  photo must be flipped back first or left/right come out swapped.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TELL, ASK = 0.5, 0.3  # chosen 2026-11-08 from the v6 confidence sweep (fewer friend questions than 0.6/0.3)
MODEL_VERSION = "yolo26n-v6"
KIND = {0: "stain", 1: "tear"}
KP_CONF = 0.5
# COCO keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
SIDE = {"L": "왼쪽", "R": "오른쪽"}


class ClothesInspector:
    def __init__(self, defect_weights, pose_weights, tear_weights=None, imgsz=512, device=0):
        det = YOLO(str(defect_weights))
        self.det = {0: det, 1: YOLO(str(tear_weights)) if tear_weights else det}  # class id -> model used for that class
        self.pose = YOLO(str(pose_weights))
        self.imgsz, self.device = imgsz, device

    def inspect(self, img, mode="laid", view=None, mirrored=False, dismissed=None):
        """One photo. dismissed: spots a friend already said were not defects on this garment (see apply_garment_memory)."""
        result, img, person = self._inspect_raw(img, mode, view, mirrored)
        if dismissed:
            apply_garment_memory(result, dismissed)
        return self._finish(result, img, person)

    def inspect_views(self, imgs, mode="worn", views=None, mirrored=False, dismissed=None):
        """Several photos of one check (worn front/side/back, or a few laid shots); uncertain spots seen in only one
        photo are dropped (see combine_views), so the friend is asked less often."""
        views = views or [None] * len(imgs)
        raw = [self._inspect_raw(im, mode, v, mirrored) for im, v in zip(imgs, views)]
        results = [r for r, _, _ in raw]
        if dismissed:
            for r in results:
                apply_garment_memory(r, dismissed)
        combine_views(results)
        return [self._finish(r, im, person) for r, im, person in zip(results, [x[1] for x in raw], [x[2] for x in raw])]

    def _finish(self, result, img, person):
        _renumber(result)
        asks = [d for d in result["defects"] if d["status"] == "ask"]
        result["askImage"] = self._ask_image(img, asks, person) if asks else None
        return result

    def _inspect_raw(self, img, mode="laid", view=None, mirrored=False):
        mode = mode or "laid"
        if mirrored:
            img = cv2.flip(img, 1)  # undo the selfie flip so the wearer's left/right are real
        h, w = img.shape[:2]
        dets, results = [], {}
        for cls, model in self.det.items():
            if id(model) not in results:  # run each distinct model once
                results[id(model)] = model.predict(img, imgsz=self.imgsz, conf=ASK, device=self.device, verbose=False)[0]
            dets += [(cls, float(b.conf), b.xyxy[0].tolist()) for b in results[id(model)].boxes if int(b.cls) == cls]

        person, framing = None, None
        if mode == "worn":
            person = self._person(img)
            framing = self._framing(person, view, w, h)

        defects = []
        for cls, conf, (x0, y0, x1, y1) in sorted(dets, key=lambda d: -d[1]):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            where = self._where_worn(person, cx, cy, view) if person is not None else None
            if where is None:
                where = self._where_laid(cx / w, cy / h)
            defects.append({
                "kind": KIND.get(cls, "other"), "where": where, "confidence": round(conf, 3),
                "status": "confirmed" if conf >= TELL else "ask",
                "box": {"x": round(x0 / w, 4), "y": round(y0 / h, 4), "w": round((x1 - x0) / w, 4), "h": round((y1 - y0) / h, 4)},
            })
        result = {
            "defects": defects,
            "askImage": None,
            "framing": framing,
            "lighting": "dark" if cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 60 else "ok",
            "source": "model",
            "modelVersion": MODEL_VERSION,
        }
        return result, img, person

    # ---- worn: pose ----
    def _person(self, img):
        r = self.pose.predict(img, imgsz=640, conf=0.4, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.tolist()]))  # biggest person
        return {"kp": r.keypoints.xy[i].cpu().numpy(), "kc": r.keypoints.conf[i].cpu().numpy(), "box": r.boxes.xyxy[i].tolist()}

    @staticmethod
    def _framing(person, view, w, h):
        if person is None:
            return "사람이 보이지 않아요. 폰 앞에 서 주세요."
        ok = person["kc"] >= KP_CONF
        x0, y0, x1, y1 = person["box"]
        if not (ok[L_AN] or ok[R_AN]):
            return "발이 안 보여요. 한 걸음 더 물러나 주세요."
        if not (ok[NOSE] or ok[L_EAR] or ok[R_EAR] or ok[L_SH] or ok[R_SH]):
            return "머리가 잘렸어요. 한 걸음 더 물러나 주세요."
        if x0 < 0.01 * w or x1 > 0.99 * w:
            return "몸이 화면 옆으로 잘렸어요. 폰 정면 가운데로 와 주세요."
        return None

    def _where_worn(self, p, x, y, view):
        kp, ok = p["kp"], p["kc"] >= KP_CONF
        back = view == "back"
        pt = np.array([x, y])
        # torso: shoulders needed; when the hips are out of frame, put them ~1.6 shoulder widths below the shoulders
        if ok[L_SH] and ok[R_SH]:
            sh_l, sh_r = kp[L_SH], kp[R_SH]
            if ok[L_HIP] and ok[R_HIP]:
                hp_l, hp_r = kp[L_HIP], kp[R_HIP]
            else:
                down = 1.6 * np.linalg.norm(sh_r - sh_l)
                hp_l, hp_r = sh_l + np.array([0, down]), sh_r + np.array([0, down])
            if cv2.pointPolygonTest(np.float32([sh_l, sh_r, hp_r, hp_l]), (float(x), float(y)), False) >= 0:
                top, bot = (sh_l + sh_r) / 2, (hp_l + hp_r) / 2
                t = np.dot(pt - top, bot - top) / max(1e-6, np.dot(bot - top, bot - top))
                # signed position across the body: 0 = person's left side, 1 = person's right side
                left_edge, right_edge = sh_l + (hp_l - sh_l) * t, sh_r + (hp_r - sh_r) * t
                s = np.dot(pt - left_edge, right_edge - left_edge) / max(1e-6, np.dot(right_edge - left_edge, right_edge - left_edge))
                side = "왼쪽 " if s < 0.38 else "오른쪽 " if s > 0.62 else ""
                if back:
                    part = "등 위쪽" if t < 0.4 else "등 가운데" if t < 0.75 else "허리 뒤쪽"
                else:
                    part = "가슴" if t < 0.4 else "배" if t < 0.75 else "허리"
                return side + part
        # limbs: nearest visible segment
        segs = [("L", "arm", L_SH, L_EL), ("L", "fore", L_EL, L_WR), ("R", "arm", R_SH, R_EL), ("R", "fore", R_EL, R_WR),
                ("L", "thigh", L_HIP, L_KN), ("L", "shin", L_KN, L_AN), ("R", "thigh", R_HIP, R_KN), ("R", "shin", R_KN, R_AN)]
        scale = np.linalg.norm(kp[L_SH] - kp[R_SH]) if ok[L_SH] and ok[R_SH] else 0.25 * (p["box"][2] - p["box"][0])
        best = None
        for side, part, a, b in segs:
            if not (ok[a] and ok[b]):
                continue
            pa, pb = kp[a], kp[b]
            t = float(np.clip(np.dot(pt - pa, pb - pa) / max(1e-6, np.dot(pb - pa, pb - pa)), 0, 1))
            d = np.linalg.norm(pt - (pa + t * (pb - pa)))
            if best is None or d < best[0]:
                best = (d, side, part, t)
        if best is None or best[0] > 0.7 * max(scale, 1):
            # only the hips visible (shoulders/knees out of frame): just above the hip line -> waist, below -> thigh
            hips = [(s, kp[i]) for s, i in (("L", L_HIP), ("R", R_HIP)) if ok[i]]
            if hips:
                hip_y = min(hp[1] for _, hp in hips)
                hip_w = np.linalg.norm(kp[L_HIP] - kp[R_HIP]) if len(hips) == 2 else 0.5 * scale
                side = min(hips, key=lambda sh: abs(sh[1][0] - x))[0]
                if y > hip_y:
                    return f"{SIDE[side]} 허벅지"
                if y > hip_y - 0.5 * hip_w:
                    return f"{SIDE[side]} 허리 근처"
            return None
        _, side, part, t = best
        name = {"arm": "팔 위쪽 소매", "fore": "소매 끝" if t > 0.7 else "팔꿈치 아래",
                "thigh": "허벅지", "shin": "바지 밑단" if t > 0.75 else "무릎 아래"}[part]
        return f"{SIDE[side]} {name}"

    # ---- laid: photo grid (photo left = user's left) ----
    @staticmethod
    def _where_laid(fx, fy):
        col = "왼쪽" if fx < 0.36 else "오른쪽" if fx > 0.64 else "가운데"
        row = "위쪽" if fy < 0.36 else "아래쪽" if fy > 0.64 else "가운데"
        if col == "가운데" and row == "가운데":
            return "옷 한가운데"
        if col == "가운데":
            return f"옷 {row} 가운데"
        if row == "가운데":
            return f"옷 {col}"
        return f"옷 {col} {row}"

    # ---- one image for the friend: numbered close-ups, faces blurred ----
    @staticmethod
    def _ask_image(img, asks, person, tile=320):
        img = img.copy()
        h, w = img.shape[:2]
        if person is not None:
            kp, ok = person["kp"], person["kc"] >= 0.3
            face = [kp[i] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if ok[i]]
            if face:
                f = np.array(face)
                r = max(25.0, 1.6 * float(np.ptp(f[:, 0])) if len(f) > 1 else 40.0)
                fx0, fy0 = int(max(0, f[:, 0].mean() - r)), int(max(0, f[:, 1].mean() - 1.3 * r))
                fx1, fy1 = int(min(w, f[:, 0].mean() + r)), int(min(h, f[:, 1].mean() + 1.3 * r))
                if fx1 > fx0 and fy1 > fy0:
                    img[fy0:fy1, fx0:fx1] = cv2.GaussianBlur(img[fy0:fy1, fx0:fx1], (0, 0), 25)
        tiles = []
        for d in asks:
            b = d["box"]
            cx, cy = (b["x"] + b["w"] / 2) * w, (b["y"] + b["h"] / 2) * h
            half = max(b["w"] * w, b["h"] * h) * 1.5 + 40  # some context around the spot
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
            crop = img[y0:y1, x0:x1].copy()
            bx0, by0 = int(b["x"] * w - x0), int(b["y"] * h - y0)
            cv2.rectangle(crop, (bx0, by0), (bx0 + int(b["w"] * w), by0 + int(b["h"] * h)), (0, 0, 255), 3)
            s = tile / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
            t = np.full((tile + 36, tile, 3), 255, np.uint8)
            t[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(t, f"{d['askNumber']}", (8, tile + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            tiles.append(t)
        sheet = np.hstack(tiles) if len(tiles) <= 3 else np.vstack(
            [np.hstack(tiles[i:i + 3] + [np.full_like(tiles[0], 255)] * (3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)])
        return base64.b64encode(cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()).decode()


def save_feedback(img, result, answers, out_dir):
    """Turn a friend's answers into training data: answers = {askNumber: True (it is a defect) | False (it is not)}.
    The caller should also keep the "no" spots ({kind, where, box}) under the garment id and pass them back as
    `dismissed` next time, so the same print or logo is not asked about again.
    Writes the photo plus a YOLO label holding confirmed defects (status "confirmed" or answered True). If every
    asked spot was answered False and nothing was confirmed, the label is empty, so the photo becomes a clean negative.
    Spots nobody answered are left out of the label, and the photo is skipped, so unknowns never become negatives."""
    import time
    if any(d["status"] == "ask" and d["askNumber"] not in answers for d in result["defects"]):
        return None
    rows = []
    for d in result["defects"]:
        if d["status"] == "confirmed" or answers.get(d.get("askNumber")) is True:
            b = d["box"]
            cls = {v: k for k, v in KIND.items()}.get(d["kind"])
            if cls is not None:
                rows.append(f"{cls} {b['x'] + b['w'] / 2:.6f} {b['y'] + b['h'] / 2:.6f} {b['w']:.6f} {b['h']:.6f}")
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    name = f"fb_{int(time.time() * 1000)}"
    cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    return name


# ---- fewer friend questions ----
def _box_iou(a, b):
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    return inter / max(1e-9, a["w"] * a["h"] + b["w"] * b["h"] - inter)


def apply_garment_memory(result, dismissed, min_iou=0.3):
    """Drop uncertain spots a friend already said were not defects on this same garment (e.g. a print or logo).
    dismissed: [{"kind", "where", "box"}] saved per garment id from earlier "no" answers. A spot matches when it is the
    same kind and either has the same `where` or overlaps the saved box. Confirmed spots are never dropped."""
    keep = []
    for d in result["defects"]:
        known = d["status"] == "ask" and any(
            m["kind"] == d["kind"] and (m.get("where") == d["where"] or _box_iou(m["box"], d["box"]) >= min_iou)
            for m in dismissed)
        if not known:
            keep.append(d)
    result["defects"] = keep
    return _renumber(result)


def combine_views(results):
    """For several photos of one check (worn front/side/back, or a few laid shots): an uncertain spot is only asked
    about when the same kind shows up at the same `where` in at least two photos; one-off spots are most often
    shadows or creases. Confirmed spots are always kept. With a single photo nothing is filtered."""
    if len(results) < 2:
        return results
    seen = {}
    for i, r in enumerate(results):
        for d in r["defects"]:
            seen.setdefault((d["kind"], d["where"]), set()).add(i)
    for r in results:
        r["defects"] = [d for d in r["defects"] if d["status"] == "confirmed" or len(seen[(d["kind"], d["where"])]) >= 2]
        _renumber(r)
    return results


def _renumber(result):
    """Keep askNumber 1..n for the remaining uncertain spots (the ask image is built afterwards from these)."""
    for d in result["defects"]:
        d.pop("askNumber", None)
    for i, d in enumerate([d for d in result["defects"] if d["status"] == "ask"], 1):
        d["askNumber"] = i
    return result

if __name__ == "__main__":
    import json, sys
    # usage: python clothes_inspect_model.py <defect.pt> <pose.pt> <image> [laid|worn] [front|side|back] [out_ask.jpg]
    ins = ClothesInspector(sys.argv[1], sys.argv[2])
    res = ins.inspect(cv2.imread(sys.argv[3]), *(sys.argv[4:6] if len(sys.argv) > 4 else []))
    if res["askImage"] and len(sys.argv) > 6:
        Path(sys.argv[6]).write_bytes(base64.b64decode(res["askImage"]))
    res["askImage"] = "<jpeg base64>" if res["askImage"] else None
    print(json.dumps(res, ensure_ascii=False, indent=1))
