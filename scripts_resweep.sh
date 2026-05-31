#!/bin/bash
# Re-sweep v7_async_prio (context-handoff fix) across nt, append to resweep.txt
set -u
cd /home/vishnu/QR_factorisation
OUT=resweep.txt
: > "$OUT"
echo "# v7_async_prio (context-handoff fix) @ m=640 n=64 chunk=4" >> "$OUT"
for nt in 1 2 3 4 6 8; do
  printf '640 64 %d 4 0 8\n' "$nt" > build/inparams.txt
  line=$(make sim-v7_dag_hvx_sched 2>/dev/null | grep -E "^\[alg\]")
  printf 'async_prio nt=%d | %s\n' "$nt" "$line" >> "$OUT"
done
echo "ALL DONE" >> "$OUT"
