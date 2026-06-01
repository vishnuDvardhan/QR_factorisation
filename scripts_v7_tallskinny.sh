#!/bin/bash
# Does v7 async_prio (critical-path pivot overlap) beat barrier on TALL-SKINNY
# matrices? Hypothesis: long trailing-column updates (O(m)) give the reflector-
# k+1 overlap more to hide behind, so async_prio should separate from barrier
# here in a way it did NOT on square matrices.
#
# v7 inparams: m n nt chunk sched nb  (sched 0=async_prio, 1=barrier). v7 is
# unblocked so nb is ignored. Both modes use identical kernels -> isolates the
# scheduling policy.
set -u
cd /home/vishnu/QR_factorisation
OUT=v7_tallskinny.txt
: > "$OUT"
echo "# v7 async_prio vs barrier on tall-skinny (nt=4). m n | async_prio | barrier | speedup" >> "$OUT"
make v7_dag_hvx_sched >/dev/null 2>&1
run() {  # m n sched
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done; sleep 1
  printf '%d %d 4 4 %d 8\n' "$1" "$2" "$3" > build/inparams.txt
  timeout 1200 make sim-v7_dag_hvx_sched 2>/dev/null | awk '/\[alg\]/{for(i=1;i<=NF;i++) if($i=="cycles") print $(i-1)}'
}
for shape in "1024 16" "1024 32" "1024 64" "2048 16" "2048 32" "4096 16" "640 64"; do
  set -- $shape; m=$1; n=$2
  ap=$(run $m $n 0); [ -z "$ap" ] && ap="NA"
  ba=$(run $m $n 1); [ -z "$ba" ] && ba="NA"
  if [ "$ap" != "NA" ] && [ "$ba" != "NA" ]; then
    sp=$(awk "BEGIN{printf \"%.3f\", $ba/$ap}")
  else sp="NA"; fi
  printf 'm=%-5s n=%-4s | async_prio=%-10s | barrier=%-10s | barrier/ap=%s\n' "$m" "$n" "$ap" "$ba" "$sp" | tee -a "$OUT"
done
echo "ALL DONE" >> "$OUT"
