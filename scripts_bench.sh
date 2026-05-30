#!/bin/bash
# Clean sequential benchmark runner. Each run sets build/inparams.txt, runs one
# sim, and appends a single labeled result line to results_bench.txt.
set -u
cd /home/vishnu/QR_factorisation
OUT=results_bench.txt
: > "$OUT"

run_v7() {  # label sched nt
  local label="$1" sched="$2" nt="$3"
  printf '640 64 %d 4 %d 8\n' "$nt" "$sched" > build/inparams.txt
  local line
  line=$(make sim-v7_dag_hvx_sched 2>/dev/null | grep -E "^\[alg\]")
  printf '%-26s | %s\n' "$label" "$line" >> "$OUT"
}
run_pool() {  # label nt
  local label="$1" nt="$2"
  printf '640 64 %d 4 0 8\n' "$nt" > build/inparams.txt
  local line
  line=$(make sim-pool 2>/dev/null | grep -E "^\[alg\]")
  printf '%-26s | %s\n' "$label" "$line" >> "$OUT"
}
run_colmajor() {  # label nt
  local label="$1" nt="$2"
  printf '640 64 %d 4 0 8\n' "$nt" > build/inparams.txt
  local line
  line=$(make sim-baseline 2>/dev/null | grep -E "QR cycles")
  printf '%-26s | %s\n' "$label" "$line" >> "$OUT"
}

echo "# E1 head-to-head @ m=640 n=64 chunk=4" >> "$OUT"
run_colmajor "colmajor_mt nt=4"        4
run_pool     "v0_pool nt=4"            4
run_v7       "v7_barrier nt=4"     1   4
run_v7       "v7_async_prio nt=4"  0   4
run_v7       "v7_async_prio nt=3"  0   3

echo "# E3 nt sweep (v7 async_prio vs v0_pool)" >> "$OUT"
for nt in 1 2 4 6 8; do
  run_pool "v0_pool nt=$nt"          "$nt"
  run_v7   "v7_async_prio nt=$nt" 0  "$nt"
done

echo "ALL DONE" >> "$OUT"
