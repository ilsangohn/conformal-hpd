#!/bin/sh
# Bike-sharing application (Appendix G.6), one random split per call. Usage: scripts/run_bike.sh SPLIT (0..9).
# Output: results/bike/rank_bike_SPLIT.jsonl and .err. Data: data/hour.csv (UCI Bike Sharing Dataset, hourly).
root=$(cd "$(dirname "$0")/.." && pwd); out=$root/results/bike; mkdir -p "$out"
cd "$root/sim" || exit 1
if [ "$(uname)" = Darwin ]; then TIME="/usr/bin/time -l"; else TIME="/usr/bin/time -v"; fi
env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    REF=gmix RANK_SWEEP=1 TUNE_GRID=ext CELL_GRID=sim \
    $TIME python realdata_rank.py $1 > "$out/rank_bike_$1.jsonl" 2> "$out/rank_bike_$1.err"
echo "# exit=$? lines=$(grep -c '^{' "$out/rank_bike_$1.jsonl")" >> "$out/rank_bike_$1.err"
