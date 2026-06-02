#!/bin/bash
# Profile cc_queue overhead: vary m (column length = compute per task) at fixed
# n,nt. If the cc_queue/async_prio gap SHRINKS as m grows -> per-task sync is the
# cost (chunk it). If the gap is FLAT -> main is the serialization bottleneck
# (chunking won't help). No code change; just measurement.
#
# Uncapped runs polled for completion (NOT fixed timeouts — those mis-read slow
# as hung on this stdout-buffered ISS). Each run waits until the sim exits.
set -u
cd /home/vishnu/QR_factorisation
OUT=ccq_profile.txt
: > "$OUT"
echo "# cc_queue(4) vs async_prio(0), n=64 nt=4, vary m. label | cycles | R-err" >> "$OUT"

run() {  # label m n sched
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done; sleep 1
  printf '%d %d 4 4 %d 8\n' "$2" "$3" "$4" > build/inparams.txt
  # launch uncapped, poll until the sim process exits (max ~25 min safety)
  : > /tmp/ccq_prof_run.log
  ( make sim-v7_dag_hvx_sched > /tmp/ccq_prof_run.log 2>&1 ) &
  local mkpid=$!
  local waited=0
  while kill -0 "$mkpid" 2>/dev/null; do sleep 5; waited=$((waited+5)); [ $waited -ge 1500 ] && { kill -9 "$mkpid" 2>/dev/null; break; }; done
  local line
  line=$(grep -E "^\[alg\]" /tmp/ccq_prof_run.log | head -1)
  [ -z "$line" ] && line="(no result; waited ${waited}s)"
  printf '%-22s | %s\n' "$1" "$line" >> "$OUT"
}

for m in 128 512 1024; do
  run "async_prio m=${m}" $m 64 0
  run "cc_queue   m=${m}" $m 64 4
done
echo "ALL DONE" >> "$OUT"
