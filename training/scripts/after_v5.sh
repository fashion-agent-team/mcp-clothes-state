#!/usr/bin/env bash
# Runs on the Jetson after the v5 queue: confidence sweep for the two-model setup, then the hyperparameter search.
R="${R:-$(cd "$(dirname "$0")/.." && pwd)}"  # project root (default: parent of scripts/)
LOGS=$R/logs
export YOLO_OFFLINE=1
until grep -q "ALL DONE" $LOGS/queue_v5.log 2>/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') conf sweep"
cd $HOME && $R/venv/bin/python -I $R/scripts/conf_sweep.py $R $R/conf_sweep.json > $LOGS/conf_sweep.log 2>&1
echo "=== $(date '+%F %T') tune start"
cd $R/weights
for b in 16 8; do
  $R/venv/bin/python -I $R/scripts/tune_v5.py $R $b > $LOGS/tune.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && break
  [ $rc -eq 137 ] || grep -qiE "out of memory" $LOGS/tune.log || break
  echo "=== tune OOM at batch=$b, retrying smaller"
done
echo "=== $(date '+%F %T') ALL DONE (tune rc=$rc)"
