#!/usr/bin/env bash
# step 2: LoRA r16/a32, 10 epochs patience 3, warmup 1 epoch, weak classes x2, crop 0.7-1, weak color; then step-1 combos on it
cd ~/clothes-state-train; E=exp3; PY=venv/bin/python; export HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
stamp() { echo "$1 $(date +%F_%T) $2" >> $E/times.log; }
stamp "C1v2 start" "batch 8 r16"
LORA_R=16 LORA_A=32 EPOCHS=10 PATIENCE=3 WARMUP_EP=1 CLASS_W="dot=2,other=2,graphic=2,stripe=2" CROP_MIN=0.7 JITTER="0.1,0.1,0.1" \
  $PY $E/c1_siglip_lora.py lora $E/c1_lora_v2 8 > $E/c1_lora_v2.log 2>&1
stamp "C1v2 end" "$([ -f $E/c1_lora_v2/result.json ] && echo ok || echo FAILED)"
[ -f $E/c1_lora_v2/result.json ] && $PY $E/c1_ensemble.py $E/c1_lora_v2 $E/c1_probe $E/c1_ens_v2 > $E/c1_ens_v2.log 2>&1
stamp "C1v2ens end"
