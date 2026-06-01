#!/bin/bash
# v8 scaling + scheduler-ablation sweep (HVX panel factor).
# Kills stray sims before each run and uses a generous timeout so host load
# never produces a false "no result". Appends one line per config.
set -u
cd /home/vishnu/QR_factorisation
OUT=v8_scale.txt
: > "$OUT"
echo "# v8 tiled scaling/ablation (HVX panel). label | cycles | err" >> "$OUT"
run() {  # label m n nt sched B to
  local label="$1" m="$2" n="$3" nt="$4" sched="$5" B="$6" to="$7"
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done
  sleep 1
  printf '%d %d %d 4 %d %d\n' "$m" "$n" "$nt" "$sched" "$B" > build/inparams.txt
  local line
  line=$(timeout "$to" make sim-v8_tiled_dag_hvx 2>/dev/null | grep -E "^\[alg\]")
  [ -z "$line" ] && line="(NO RESULT)"
  printf '%-30s | %s\n' "$label" "$line" >> "$OUT"
}
# --- nt scaling at 512x512 B=64 (async_prio) ---
for nt in 1 2 4 6 8; do run "512 B64 async_prio nt$nt" 512 512 $nt 0 64 500; done
# --- priority ablation at 512x512 B=64 nt=4 ---
run "512 B64 barrier    nt4" 512 512 4 2 64 500
run "512 B64 async      nt4" 512 512 4 1 64 500
# (async_prio nt4 already above)
# --- tile-size sweep at 512x512 nt=4 async_prio ---
run "512 async_prio nt4 B32" 512 512 4 0 32 500
run "512 async_prio nt4 B128" 512 512 4 0 128 500
echo "ALL DONE" >> "$OUT"
