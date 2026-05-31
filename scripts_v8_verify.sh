#!/bin/bash
# v8 scalar tiled-QR correctness sweep across (m,n,B) — gate before HVX.
set -u
cd /home/vishnu/QR_factorisation
OUT=v8_verify.txt
: > "$OUT"
echo "# v8 scalar tiled QR correctness sweep (m n B | result)" >> "$OUT"
# configs: m n B  (require B|m, B|n, m multiple of 32)
for cfg in \
  "128 128 32" "128 128 64" "256 128 32" "256 128 64" \
  "256 256 64" "512 64 32"  "512 256 64" "384 96 32" \
  "640 64 32" "1024 128 64"; do
  set -- $cfg
  printf '%d %d 4 4 0 %d\n' "$1" "$2" "$3" > build/inparams.txt
  line=$(make sim-v8_tiled_dag_hvx 2>/dev/null | grep -E "^\[alg\]")
  printf 'm=%-4s n=%-4s B=%-3s | %s\n' "$1" "$2" "$3" "$line" >> "$OUT"
done
echo "ALL DONE" >> "$OUT"
