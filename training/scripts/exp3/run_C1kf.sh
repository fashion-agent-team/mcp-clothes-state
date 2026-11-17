#!/usr/bin/env bash
# K-Fashion added: (a) linear probe on old + K-Fashion, (b) continue first LoRA 2 epochs lr 5e-5, then combos
cd ~/clothes-state-train; E=exp3; PY=venv/bin/python; export HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
stamp() { echo "$1 $(date +%F_%T) $2" >> $E/times.log; }
stamp "C1kf probe start"
EXTRA_DS=$E/kfashion_pattern $PY $E/c1_siglip_lora.py probe $E/c1_probe_kf 16 > $E/c1_probe_kf.log 2>&1
stamp "C1kf probe end" "$([ -f $E/c1_probe_kf/result.json ] && echo ok || echo FAILED)"
stamp "C1kf lora start"
EXTRA_DS=$E/kfashion_pattern INIT_LORA=$E/c1_lora/lora_best INIT_HEAD=$E/c1_lora/head_best.pt EPOCHS=2 LR_LORA=5e-5 LR_HEAD=5e-5 \
  $PY $E/c1_siglip_lora.py lora $E/c1_lora_kf 8 > $E/c1_lora_kf.log 2>&1
stamp "C1kf lora end" "$([ -f $E/c1_lora_kf/result.json ] && echo ok || echo FAILED)"
[ -f $E/c1_lora_kf/result.json ] && $PY $E/c1_ensemble.py $E/c1_lora_kf $E/c1_probe_kf $E/c1_ens_kf > $E/c1_ens_kf.log 2>&1
stamp "C1kf ens end"
