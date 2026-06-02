#!/bin/bash
# Does holding an HVX context on the critical path (sched=3, async_prio_cc) help?
# Compare cc vs async_prio vs barrier, square + tall-skinny. GENEROUS 600s caps
# (the earlier 'deadlocks' were just too-short timeouts on a stdout-buffered ISS).
set -u
cd /home/vishnu/QR_factorisation
OUT=cc_compare.txt
: > "$OUT"
echo "# CC (sched=3) vs async_prio(0) vs barrier(1). nt=4. label | cycles | R-err" >> "$OUT"
run() {  # label m n sched
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done; sleep 1
  printf '%d %d 4 4 %d 8\n' "$2" "$3" "$4" > build/inparams.txt
  local line
  line=$(timeout 600 make sim-v7_dag_hvx_sched 2>/dev/null | grep -E "^\[alg\]")
  [ -z "$line" ] && line="(NO RESULT in 600s)"
  printf '%-26s | %s\n' "$1" "$line" >> "$OUT"
}
for shape in "256 256" "1024 64" "2048 32"; do
  set -- $shape; m=$1; n=$2
  run "barrier    ${m}x${n}" $m $n 1
  run "async_prio ${m}x${n}" $m $n 0
  run "cc         ${m}x${n}" $m $n 3
done
echo "ALL DONE" >> "$OUT"
