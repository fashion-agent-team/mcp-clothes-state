#!/usr/bin/env bash
# Pattern classifier (8 classes) on the Jetson. Waits for any running YOLO training to finish first.
# A = Fashionpedia crops only, B = Fashionpedia + Second-Hand laid photos. Same val, same tests.
# usage: R=<clothes-state root> train_queue_pattern.sh <fashionpedia raw dir> <second-hand parquet dir>
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"
P=$R/pattern; PY=$R/venv/bin/python; YOLO=$R/venv/bin/yolo; LOGS=$R/logs
export YOLO_OFFLINE=1
mkdir -p $LOGS
while pgrep -f "yolo detect train" > /dev/null; do echo "$(date '+%F %T') waiting for running training"; sleep 300; done
echo "=== $(date '+%F %T') build dataset"
[ -d $P/ds/train ] || $PY -I $P/scripts/pattern_build.py $P/pattern_crops.json "$1" "$2" $P/ds > $LOGS/pattern_build.log 2>&1
cat $LOGS/pattern_build.log
# dataset A: Fashionpedia-only train, shared val
rm -rf $P/dsA; mkdir -p $P/dsA
for c in $(ls $P/ds/train); do mkdir -p $P/dsA/train/$c; find $P/ds/train/$c -name 'fp*.jpg' -exec ln -s {} $P/dsA/train/$c/ \; ; done
ln -s $P/ds/val $P/dsA/val
cd $P
for N in A B; do
  D=$P/ds; [ $N = A ] && D=$P/dsA
  for b in 64 32 16; do
    echo "=== $(date '+%F %T') train pattern_$N batch=$b"
    $YOLO classify train model=$P/yolo26n-cls.pt data=$D imgsz=224 epochs=25 patience=8 batch=$b device=0 workers=2 \
         project=$P/runs name=pattern_$N exist_ok=True > $LOGS/train_pattern_$N.log 2>&1 && break
    grep -qiE "out of memory|CUDA error" $LOGS/train_pattern_$N.log || break
  done
  $PY -I $P/scripts/pattern_eval.py $P/runs/pattern_$N/weights/best.pt $P/ds $P/eval_pattern_$N.json > $LOGS/eval_pattern_$N.log 2>&1
  cat $LOGS/eval_pattern_$N.log
done
echo "=== $(date '+%F %T') ALL DONE"
