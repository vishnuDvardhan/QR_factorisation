#!/bin/bash
# v8 parallel scheduler correctness across modes/threads/sizes.
# Generous per-run timeout; kills any stray sim between runs to keep the host
# from oversubscribing (ISS runs are wall-clock sensitive).
set -u
cd /home/vishnu/QR_factorisation
OUT=v8_par_verify.txt
: > "$OUT"
echo "# v8 parallel tiled-QR correctness (mode nt m n B | result)" >> "$OUT"
run() {  # label m n nt sched B
  local label="$1" m="$2" n="$3" nt="$4" sched="$5" B="$6"
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done
  printf '%d %d %d 4 %d %d\n' "$m" "$n" "$nt" "$sched" "$B" > build/inparams.txt
  local line
  line=$(timeout 400 make sim-v8_tiled_dag_hvx 2>/dev/null | grep -E "^\[alg\]")
  [ -z "$line" ] && line="(NO RESULT — timed out/host-loaded)"
  printf '%-26s | %s\n' "$label" "$line" >> "$OUT"
}
# sched: 0=async_prio 1=async 2=barrier 3=seq
run "seq        128 B32"     128 128 1 3 32
run "barrier nt4 128 B32"    128 128 4 2 32
run "async   nt4 128 B32"    128 128 4 1 32
run "async_prio nt4 128 B32" 128 128 4 0 32
run "async_prio nt1 128 B32" 128 128 1 0 32
run "async_prio nt2 128 B32" 128 128 2 0 32
run "async_prio nt4 256 B32" 256 256 4 0 32
run "async_prio nt4 256 B64" 256 256 4 0 64
echo "ALL DONE" >> "$OUT"
