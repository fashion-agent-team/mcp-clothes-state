#!/usr/bin/env bash
# every minute: available memory + container health -> exp3/mem.log; external health every 10 min.
# Stops the training PID in exp3/train.pid (writes exp3/ALERT) only when:
#   available memory < 500MB for 2 checks in a row, or
#   a container is unhealthy/restarting for more than 5 checks (minutes) in a row, or
#   a container (by name) starts 3+ times within 10 minutes (crash loop).
# A teammate's redeploy (containers recreated once) is NOT a reason to stop.
# usage: HEALTH_URL=https://<app>/actuator/health ./memwatch.sh   (HEALTH_URL optional)
E=~/clothes-state-train/exp3; low=0; i=0
declare -A sick last_start; starts=""   # starts: lines "epoch name"
while true; do
  now=$(date +%s)
  av=$(free -m | awk '/^Mem|^메모리/{print $7}')
  ps_out=$(docker ps --format '{{.Names}} {{.Status}}'); bad=$(echo "$ps_out" | grep -E 'unhealthy|Restarting')
  echo "$(date +%F_%T) avail=${av}MB swapused=$(free -m | awk 'NR==3{print $3}')MB n=$(docker ps -q | wc -l) bad=[$bad]" >> $E/mem.log
  [ "$av" -lt 500 ] && low=$((low+1)) || low=0
  reason=""; [ $low -ge 2 ] && reason="low_mem ${av}MB"
  for n in $(docker ps -a --format '{{.Names}}'); do
    st=$(docker inspect -f '{{.State.StartedAt}}' $n 2>/dev/null)
    if [ -n "${last_start[$n]}" ] && [ "$st" != "${last_start[$n]}" ]; then starts+="$now $n"$'\n'; echo "$(date +%F_%T) started $n" >> $E/mem.log; fi
    last_start[$n]=$st
    if echo "$ps_out" | grep -qE "^$n .*(unhealthy|Restarting)"; then sick[$n]=$(( ${sick[$n]:-0} + 1 )); else sick[$n]=0; fi
    [ ${sick[$n]} -gt 5 ] && reason="container $n unhealthy ${sick[$n]} min"
  done
  starts=$(echo -n "$starts" | awk -v t=$((now-600)) 'NF && $1>=t')$'\n'
  loop=$(echo -n "$starts" | awk 'NF{c[$2]++} END{for(k in c) if(c[k]>=3) print k}' | head -1)
  [ -n "$loop" ] && reason="container $loop started 3+ times in 10 min"
  if [ -n "$reason" ] && [ -f $E/train.pid ]; then
    echo "$(date +%F_%T) $reason" >> $E/ALERT; kill -INT $(cat $E/train.pid) 2>/dev/null; rm -f $E/train.pid; low=0
  fi
  i=$((i+1)); if [ -n "$HEALTH_URL" ] && [ $((i % 10)) -eq 1 ]; then
    echo "$(date +%F_%T) health $(timeout 20 curl -s -o /dev/null -w '%{http_code} %{time_total}s' $HEALTH_URL)" >> $E/mem.log; fi
  sleep 60
done
