#!/bin/bash
# v8 feasible ablation at 256x256 B=64 (HVX panel). ISS runs ~6-7k sim-cyc/sec,
# so ~2M-cycle runs take ~5-6 min wall -> use 700s caps and kill strays first.
set -u
cd /home/vishnu/QR_factorisation
OUT=v8_abl.txt
: > "$OUT"
echo "# v8 256x256 B64 ablation (HVX panel). label | cycles | err" >> "$OUT"
run() {  # label nt sched
  local label="$1" nt="$2" sched="$3"
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done
  sleep 1
  printf '256 256 %d 4 %d 64\n' "$nt" "$sched" > build/inparams.txt
  local line
  line=$(timeout 700 make sim-v8_tiled_dag_hvx 2>/dev/null | grep -E "^\[alg\]")
  [ -z "$line" ] && line="(NO RESULT)"
  printf '%-26s | %s\n' "$label" "$line" >> "$OUT"
}
# priority ablation at nt=4
run "barrier    nt4" 4 2
run "async      nt4" 4 1
run "async_prio nt4" 4 0
# nt scaling (async_prio)
run "async_prio nt2" 2 0
run "async_prio nt6" 6 0
run "async_prio nt8" 8 0
echo "ALL DONE" >> "$OUT"
