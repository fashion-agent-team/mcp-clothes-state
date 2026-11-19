"""서버가 모은 옷 실수 학습 후보(GPT 답 + 옷 조각)로 이어 학습용 데이터셋을 만듭니다(2026-11-16, docs/gpt-check.md).

후보: candidate-dir/날짜/wear-<종류>-<아이디>.jpg + .json
  {"task":"wear","kind":"buttons_misaligned","gptMistake":true,"gptConfidence":0.9,"modelProbability":0.02,"reviewed":false,...}
사람이 확인해 reviewed를 true로 바꾼 것만 씁니다(--all이면 확인 안 한 것도, GPT 답을 그대로 정답으로).
정답을 고쳤으면 json의 gptMistake를 고친 값으로 바꿔 두세요.

출력: OUT/train|val/<클래스>/ (단추: ok·shifted, 뒤집힘: ok·inside_out, 택: none·tag). 사진 10장 중 1장은 val.
조각 위쪽(얼굴 쪽)이 회색으로 가려져 있으니, 합성 데이터셋(build_*_dataset.py)과 섞어 쓰세요:
  python3 build_from_candidates.py CANDIDATE_DIR buttons_misaligned ds_btn_real
  # ds_btn3/train 과 ds_btn_real/train 을 한 폴더로 합친 뒤(심볼릭 링크) train_cls2.py로 btn3에서 이어 학습
그리고 실제 사진 점수(eval_cls.py ds_btn_real)가 오를 때만 서버 모델을 바꿉니다.
택·앞뒤 후보는 옷 박스 전체 조각이라, 택 모델에 쓰려면 목 부분(가운데 60%, 위 -10%~+30%)을 다시 잘라야 합니다.

python3 build_from_candidates.py CANDIDATE_DIR KIND OUT_DIR [--all]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

CLASSES = {
    'buttons_misaligned': ('ok', 'shifted'),
    'inside_out': ('ok', 'inside_out'),
    'tag_out': ('none', 'tag'),
    'backwards': ('none', 'tag'),  # 앞모습 앞 목 택 = 앞뒤 바뀜(택 모델이 봄)
}


def main():
    src, kind, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    use_all = '--all' in sys.argv
    neg, pos = CLASSES[kind]
    n = {'train': {neg: 0, pos: 0}, 'val': {neg: 0, pos: 0}}
    skipped = 0
    for meta_path in sorted(src.glob(f'*/wear-{kind}-*.json')):
        meta = json.loads(meta_path.read_text())
        if not use_all and not meta.get('reviewed'):
            skipped += 1
            continue
        jpg = meta_path.with_suffix('.jpg')
        if not jpg.exists():
            continue
        cls = pos if meta.get('gptMistake') else neg
        split = 'val' if int(hashlib.md5(jpg.name.encode()).hexdigest(), 16) % 10 == 0 else 'train'
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jpg, d / jpg.name)
        n[split][cls] += 1
    print(json.dumps({'kind': kind, 'counts': n, 'notReviewed': skipped}, ensure_ascii=False))


if __name__ == '__main__':
    main()
