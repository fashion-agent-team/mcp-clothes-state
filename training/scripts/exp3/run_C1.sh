#!/usr/bin/env bash
# rerun C1 (LoRA then probe) after the device fix. Times -> exp3/times.log
cd ~/clothes-state-train; E=exp3; PY=venv/bin/python; export HF_HUB_OFFLINE=1
stamp() { echo "$1 $(date +%F_%T) $2" >> $E/times.log; }
T=$(date +%H%M); mv $E/c1_lora $E/c1_lora_stop$T 2>/dev/null; mv $E/c1_probe $E/c1_probe_stop$T 2>/dev/null
for b in 8; do
  rm -f $E/ALERT; stamp "C1lora start" "batch $b (rerun)"
  $PY $E/c1_siglip_lora.py lora $E/c1_lora $b > $E/c1_lora.log 2>&1
  [ -f $E/c1_lora/result.json ] && break
  grep -qiE "out of memory|CUDA error" $E/c1_lora.log || [ -f $E/ALERT ] || break
done
stamp "C1lora end" "$([ -f $E/c1_lora/result.json ] && echo ok || echo FAILED)"
stamp "C1probe start"; $PY $E/c1_siglip_lora.py probe $E/c1_probe 16 > $E/c1_probe.log 2>&1
stamp "C1probe end" "$([ -f $E/c1_probe/result.json ] && echo ok || echo FAILED)"
