#!/usr/bin/env bash
# B: v6ft with batch 8; if memwatch stopped it (exp3/ALERT), retry once with batch 4
cd ~/clothes-state-train; E=exp3; export YOLO_OFFLINE=1
echo "B start $(date +%F_%T) batch 8" >> $E/times.log
venv/bin/python $E/train_v6ft.py 8 yolo26n_v6ft > $E/v6ft.log 2>&1
if [ -f $E/ALERT ] && [ ! -f runs/yolo26n_v6ft/weights/best.pt -o -n "$(grep -c . $E/ALERT)" ] && ! grep -q "epochs completed" $E/v6ft.log; then
  echo "B retry $(date +%F_%T) batch 4 after: $(tail -1 $E/ALERT)" >> $E/times.log; mv $E/ALERT $E/ALERT.b8
  venv/bin/python $E/train_v6ft.py 4 yolo26n_v6ft > $E/v6ft_b4.log 2>&1
fi
echo "B end $(date +%F_%T) $(grep -m1 'epochs completed' $E/v6ft*.log)" >> $E/times.log
