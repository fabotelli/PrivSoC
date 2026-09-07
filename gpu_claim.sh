#!/usr/bin/env bash
# Claim the A10 from the xbdream idle-harvester for the duration of "$@".
# Cooperates with ~/xbrain/gpu_yield_watchdog.sh: holds ITS flock so it can't
# (re)start vLLM while we train, kills the current vLLM, waits for the memory
# to free, logs a non-silent notice, then runs the command while holding the
# lock.  On exit the lock releases and the watchdog resumes opportunism.
set -u

exec 8>/tmp/gpu_yield_watchdog.lock
flock 8   # wait out any in-flight watchdog cycle, then hold

VP=$(pgrep -u xbdream -f "vllm serve|VLLM::" | tr '\n' ' ')
if [ -n "$VP" ]; then
  echo "gpu_claim: killing xbdream vLLM ($VP) for pipeline training"
  sudo -u xbdream kill $VP 2>/dev/null
fi
for i in $(seq 1 36); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break
  [ "$i" = 18 ] && sudo -u xbdream pkill -9 -f "vllm serve|VLLM::" 2>/dev/null
  sleep 5
done
echo "gpu_claim: GPU mem used=${USED:-?}MiB -> running: $*"
echo "{\"line\": \"gpu_claim: mycobot_stack training took the A10 ($(date -u +%FT%TZ)); vLLM killed, watchdog lock held for train duration\"}" \
    >> /home/ubuntu/xbrain/data/daemon_notices.jsonl 2>/dev/null

"$@"
RC=$?
echo "gpu_claim: command rc=$RC; releasing watchdog lock"
exit $RC
