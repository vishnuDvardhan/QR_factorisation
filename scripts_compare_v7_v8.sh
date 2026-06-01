#!/bin/bash
# =============================================================================
# Comprehensive comparison harness: all v7 (unblocked DAG) and v8 (tiled DAG)
# variants/modes, plus the v0_pool and colmajor_mt baselines, at matched
# (m, n, nt) configs.  Emits a human-readable table + a CSV for plotting.
#
# Why one harness: every binary reads build/inparams.txt (hardcoded
# ./../inparams.txt), so runs MUST be serialized — this script owns that file
# for its whole run.  Do not launch concurrently with another sweep.
#
# inparams formats (all whitespace: m n nt chunk f5 f6):
#   colmajor_mt        : m n nt chunk use_static            (5 fields; "QR cycles:")
#   v0 pool / v1 / v3  : m n nt chunk use_static nb         ("[alg] <name> ... N cycles")
#   v7_dag_hvx_sched   : m n nt chunk sched nb  (sched 0=async_prio,1=barrier)
#   v8_tiled_dag_hvx   : m n nt chunk sched nb  (sched 0=async_prio,1=async,2=barrier,3=seq; nb=B)
#
# Usage:  bash scripts_compare_v7_v8.sh [B] [SIZES...]
#   B      tile size for v8 (default 32; must divide every m,n and be mult of 32)
#   SIZES  square sizes m=n to test (default "128 256")
#   NT     via env NT="1 2 4" (default "4")
# =============================================================================
set -u
cd /home/vishnu/QR_factorisation

B="${1:-32}"
shift || true
SIZES="${*:-128 256}"
NT="${NT:-4}"
TIMEOUT="${TIMEOUT:-1800}"     # per-run wall cap (s); big sizes need more
OUT=compare_v7_v8.txt
CSV=compare_v7_v8.csv

: > "$OUT"
: > "$CSV"
echo "variant,mode,m,n,nt,B,cycles,Rerr,status" >> "$CSV"
{
  echo "# ============================================================"
  echo "# v7 vs v8 comprehensive comparison   (B=$B, nt={$NT})"
  echo "# cycles via c15:14 ISS timing; lower is better"
  echo "# generated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# ============================================================"
} >> "$OUT"

# --- build everything once ---
echo "building variants..." >> "$OUT"
make pool baseline v1_hvx_reduction v3_persistent_hvx \
     v7_dag_hvx_sched v8_tiled_dag_hvx >/dev/null 2>&1

# parse the cycle count + Rerr from a run log (handles all variant formats):
#   "[alg] <name> ... <N> cycles  |  max |R|-err: <E> ..."
#   "QR cycles:        <N>"   (colmajor_mt; no error reported)
parse_cycles() {  # stdin = run output ; echoes "cycles rerr"
  awk '
    /\[alg\].*cycles/ { for (i=1;i<=NF;i++) if ($i=="cycles") c=$(i-1) }
    /err:/            { for (i=1;i<=NF;i++) if ($i ~ /err:$/) e=$(i+1) }
    /QR cycles:/      { c=$NF }
    END { if (c=="") c="NA"; if (e=="") e="NA"; print c, e }
  '
}

# run one variant: label simtarget m n nt chunk f5 f6 mode
run() {
  local variant="$1" simtgt="$2" m="$3" n="$4" nt="$5" chunk="$6" f5="$7" f6="$8" mode="$9"
  for p in $(pgrep hexagon-sim); do kill -9 "$p" 2>/dev/null; done
  sleep 1
  printf '%d %d %d %d %d %d\n' "$m" "$n" "$nt" "$chunk" "$f5" "$f6" > build/inparams.txt
  local log res cyc err status
  log=$(timeout "$TIMEOUT" make "$simtgt" 2>/dev/null)
  res=$(printf '%s\n' "$log" | parse_cycles)
  cyc=$(echo "$res" | awk '{print $1}')
  err=$(echo "$res" | awk '{print $2}')
  if [ "$cyc" = "NA" ]; then status="TIMEOUT/NORESULT"; else status="OK"; fi
  printf '%-22s %-12s m=%-5d n=%-5d nt=%-2d B=%-4d | %12s cyc | err=%-10s | %s\n' \
    "$variant" "$mode" "$m" "$n" "$nt" "$B" "$cyc" "$err" "$status" | tee -a "$OUT"
  echo "$variant,$mode,$m,$n,$nt,$B,$cyc,$err,$status" >> "$CSV"
}

for m in $SIZES; do
  n=$m
  echo "" >> "$OUT"
  echo "### size ${m}x${n}  (B=$B for v8; v7/v0/colmajor ignore B) ###" >> "$OUT"
  for nt in $NT; do
    # baselines
    run "colmajor_mt"  sim-baseline            "$m" "$n" "$nt" 4 0  "$B" "spawn"
    run "v0_pool"      sim-pool                "$m" "$n" "$nt" 4 0  "$B" "barrier"
    run "v1_hvx_red"   sim-v1_hvx_reduction    "$m" "$n" "$nt" 4 0  "$B" "P1"
    run "v3_persist"   sim-v3_persistent_hvx   "$m" "$n" "$nt" 4 0  "$B" "P4"
    # v7 unblocked DAG (sched: 0=async_prio, 1=barrier)
    run "v7_unblk_dag" sim-v7_dag_hvx_sched    "$m" "$n" "$nt" 4 0  "$B" "async_prio"
    run "v7_unblk_dag" sim-v7_dag_hvx_sched    "$m" "$n" "$nt" 4 1  "$B" "barrier"
    # v8 tiled DAG (sched: 0=async_prio,1=async,2=barrier,3=seq)
    run "v8_tiled_dag" sim-v8_tiled_dag_hvx    "$m" "$n" "$nt" 4 3  "$B" "seq"
    run "v8_tiled_dag" sim-v8_tiled_dag_hvx    "$m" "$n" "$nt" 4 2  "$B" "barrier"
    run "v8_tiled_dag" sim-v8_tiled_dag_hvx    "$m" "$n" "$nt" 4 1  "$B" "async"
    run "v8_tiled_dag" sim-v8_tiled_dag_hvx    "$m" "$n" "$nt" 4 0  "$B" "async_prio"
  done
done
echo "" >> "$OUT"
echo "ALL DONE" >> "$OUT"
