#!/bin/bash
# v8 large-size sweep at B=32 (user request): square m=n in {512,1024,2048}, nt=4,
# async_prio + the priority ablation. These are big: ~16M / ~130M / ~1B sim-cycles,
# so wall times run ~tens of minutes to hours on the ISS. Smallest first so we
# always get the 512 result even if the giants don't finish. Does NOT kill other
# sims (a 256 ablation may be running concurrently); each ISS run is single-threaded.
set -u
cd /home/vishnu/QR_factorisation
OUT=v8_tall.txt
: > "$OUT"
echo "# v8 large-size B=32 sweep. label | cycles | err" >> "$OUT"
run() {  # label m n nt sched B to
  local label="$1" m="$2" n="$3" nt="$4" sched="$5" B="$6" to="$7"
  printf '%d %d %d 4 %d %d\n' "$m" "$n" "$nt" "$sched" "$B" > build/inparams.txt
  local line
  line=$(timeout "$to" make sim-v8_tiled_dag_hvx 2>/dev/null | grep -E "^\[alg\]")
  [ -z "$line" ] && line="(NO RESULT — exceeded ${to}s wall)"
  printf '%-30s | %s\n' "$label" "$line" >> "$OUT"
}
# 512x512 B32 — priority ablation + a couple of nt points (feasible: ~30-40min total)
run "512 B32 async_prio nt4" 512 512 4 0 32 2400
run "512 B32 barrier    nt4" 512 512 4 2 32 2400
run "512 B32 async      nt4" 512 512 4 1 32 2400
run "512 B32 async_prio nt8" 512 512 8 0 32 2400
# 1024x1024 B32 — async_prio only (may take ~1-2h)
run "1024 B32 async_prio nt4" 1024 1024 4 0 32 7200
run "1024 B32 barrier    nt4" 1024 1024 4 2 32 7200
# 2048x2048 B32 — async_prio only (may take many hours; best-effort)
run "2048 B32 async_prio nt4" 2048 2048 4 0 32 14400
echo "ALL DONE" >> "$OUT"
