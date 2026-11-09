#!/bin/sh
# One simulation job (sim_rank.py), single-threaded. Usage: run_rank.sh SCEN SEED DY
# Sizes come from the environment (see scripts/run_main.sh for the settings of the paper).
# Output: ${OUT_DIR:-../results/rank}/dY$DY/rank_${SCEN}_${N}_${SEED}.jsonl and .err (with the resource use from /usr/bin/time).
# The job writes to WORK_DIR (default ../work) and copies the finished files to OUT_DIR.
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
sc=$1; s=$2; d=$3
N=${N:-9000}
export N_REG=${N_REG:-1500} NVAL=${NVAL:-1500} NCAL=${NCAL:-1000}
dir=${OUT_DIR:-../results/rank}/dY$d; mkdir -p "$dir"
work=${WORK_DIR:-../work}/$(basename "$(cd "$dir/.." && pwd)")/dY$d; mkdir -p "$work"
name=rank_${sc}_${N}_${s}
if [ "$(uname)" = Darwin ]; then TIME="/usr/bin/time -l"; else TIME="/usr/bin/time -v"; fi
$TIME python sim_rank.py "$N" "$s" "$sc" "$d" > "$work/$name.jsonl" 2> "$work/$name.err"
rc=$?
echo "# exit=$rc lines=$(grep -c '^{' "$work/$name.jsonl")" >> "$work/$name.err"
cp "$work/$name.jsonl" "$work/$name.err" "$dir/"
