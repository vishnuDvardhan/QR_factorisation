#!/bin/bash
# Verify all THREE v7 scheduler modes are correct (esp. the new async, sched=2),
# and capture cycles for the priority ablation (async vs async_prio, square+tall).
# sched: 0=async_prio, 1=barrier, 2=async
set -u
cd /home/vishnu/QR_factorisation
OUT=v7_async_verify.txt
: > "$OUT"
echo "# v7 3-mode verify+timing (mode m n nt | cycles | R-err)" >> "$OUT"
run() {  # label m n nt sched
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done; sleep 1
  printf '%d %d %d 4 %d 8\n' "$2" "$3" "$4" "$5" > build/inparams.txt
  local line
  line=$(timeout 600 make sim-v7_dag_hvx_sched 2>/dev/null | grep -E "^\[alg\]")
  [ -z "$line" ] && line="(NO RESULT)"
  printf '%-22s | %s\n' "$1" "$line" >> "$OUT"
}
# correctness on a square config, all 3 modes
run "barrier    256x256" 256 256 4 1
run "async      256x256" 256 256 4 2
run "async_prio 256x256" 256 256 4 0
# priority ablation on tall-skinny (where async_prio beat barrier): async vs async_prio
run "async      1024x64" 1024 64 4 2
run "async_prio 1024x64" 1024 64 4 0
run "barrier    1024x64" 1024 64 4 1
echo "ALL DONE" >> "$OUT"
