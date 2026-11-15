#!/usr/bin/env bash
# after B: score B, then C1 (LoRA, then probe), then C2 (50-photo timing, then 3-epoch training at 768). Times -> exp3/times.log
cd ~/clothes-state-train; E=exp3; PY=venv/bin/python; export YOLO_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while pgrep -x -f "bash ./run_B.sh" > /dev/null; do sleep 60; done
stamp() { echo "$1 $(date +%F_%T) $2" >> $E/times.log; }
stamp "Beval start"
$PY $E/eval_B.py $E/B_val.json yolo26n_v6_tuned yolo26n_v6ft > $E/B_eval.log 2>&1
$PY scripts/conf_sweep.py . $E/conf_sweep_v6.json yolo26n_v6_tuned >> $E/B_eval.log 2>&1
$PY scripts/conf_sweep.py . $E/conf_sweep_v6ft.json yolo26n_v6ft >> $E/B_eval.log 2>&1
$PY $E/realtest_eval.py . yolo26n_v6_tuned $E/realtest_v6.json 0 >> $E/B_eval.log 2>&1
$PY $E/realtest_eval.py . yolo26n_v6ft $E/realtest_v6ft.json 0 >> $E/B_eval.log 2>&1
stamp "Beval end"
for b in 16 8; do
  rm -f $E/ALERT; stamp "C1lora start" "batch $b"
  $PY $E/c1_siglip_lora.py lora $E/c1_lora $b > $E/c1_lora.log 2>&1
  [ -f $E/c1_lora/result.json ] && break
  grep -qiE "out of memory|CUDA error" $E/c1_lora.log || [ -f $E/ALERT ] || break
done
stamp "C1lora end" "$([ -f $E/c1_lora/result.json ] && echo ok || echo FAILED)"
stamp "C1probe start"; $PY $E/c1_siglip_lora.py probe $E/c1_probe 16 > $E/c1_probe.log 2>&1
stamp "C1probe end" "$([ -f $E/c1_probe/result.json ] && echo ok || echo FAILED)"
stamp "C2time start"; $PY $E/c2_owlv2_lora.py time $E/c2 768 50 > $E/c2_time.log 2>&1; stamp "C2time end" "$(grep -o '{"photos.*' $E/c2_time.log)"
stamp "C2train start"; $PY $E/c2_owlv2_lora.py train $E/c2 768 3 > $E/c2_train.log 2>&1
stamp "C2train end" "$(grep -c '"map50"' $E/c2/log.txt)"
