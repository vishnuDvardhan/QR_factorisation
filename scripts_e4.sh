#!/bin/bash
# E4 size sweep — v0_pool vs v7_barrier vs v7_async_prio across matrix sizes.
# Fixed nt=4, chunk=4. Confirms the Phase-1 win holds (and stays correct) as m,n
# grow beyond the single 640x64 point tested so far.
set -u
cd /home/vishnu/QR_factorisation
OUT=e4_results.txt
: > "$OUT"
echo "# E4 size sweep @ nt=4 chunk=4  (cycles via c15:14)" >> "$OUT"
echo "# cols: variant m n cycles | correctness" >> "$OUT"

run() {  # variant m n
  local var="$1" m="$2" n="$3" line cyc
  case "$var" in
    v0_pool)       printf '%d %d 4 4 0 8\n' "$m" "$n" > build/inparams.txt
                   line=$(make sim-pool 2>/dev/null | grep -E "^\[alg\]");;
    v7_barrier)    printf '%d %d 4 4 1 8\n' "$m" "$n" > build/inparams.txt
                   line=$(make sim-v7_dag_hvx_sched 2>/dev/null | grep -E "^\[alg\]");;
    v7_async_prio) printf '%d %d 4 4 0 8\n' "$m" "$n" > build/inparams.txt
                   line=$(make sim-v7_dag_hvx_sched 2>/dev/null | grep -E "^\[alg\]");;
  esac
  printf '%-14s m=%-4d n=%-3d | %s\n' "$var" "$m" "$n" "$line" >> "$OUT"
}

for n in 32 64; do
  for m in 128 256 512 1024; do
    [ "$n" -gt "$m" ] && continue
    run v0_pool       "$m" "$n"
    run v7_barrier    "$m" "$n"
    run v7_async_prio "$m" "$n"
  done
done
echo "ALL DONE" >> "$OUT"
