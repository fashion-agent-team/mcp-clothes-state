#!/usr/bin/env bash
# 384 input: continue the K-Fashion LoRA, old + K-Fashion data, 2 epochs, lr 5e-5 cosine, batch 4 x grad acc 2; then combos
cd ~/clothes-state-train; E=exp3; PY=venv/bin/python; export HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
stamp() { echo "$1 $(date +%F_%T) $2" >> $E/times.log; }
stamp "C1_384 start" "batch 8 acc 1 (restart)"
IMG=384 GRAD_ACC=1 EXTRA_DS=$E/kfashion_pattern INIT_LORA=$E/c1_lora_kf/lora_best INIT_HEAD=$E/c1_lora_kf/head_best.pt EPOCHS=2 LR_LORA=5e-5 LR_HEAD=5e-5 \
  $PY $E/c1_siglip_lora.py lora $E/c1_lora_384 8 > $E/c1_lora_384.log 2>&1
stamp "C1_384 end" "$([ -f $E/c1_lora_384/result.json ] && echo ok || echo FAILED)"
[ -f $E/c1_lora_384/result.json ] && IMG=384 $PY $E/c1_ensemble.py $E/c1_lora_384 $E/c1_probe_kf $E/c1_ens_384 > $E/c1_ens_384.log 2>&1
stamp "C1_384 ens end"
