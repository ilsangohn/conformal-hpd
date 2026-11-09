#!/bin/sh
# Coverage along the segment {(x_1, 0.5, 0.5)} (Figure 5): the fits of the main experiment, evaluated at 40 covariates.
# Usage: scripts/run_curves.sh MODEL DY SEED (after scripts/run_main.sh for the same job). The one-stage thresholds are
# read from the result file of the main experiment (TAU_FILE); FULL_CAL=1 recomputes them instead.
# Output: results/curves_n5k_matched/MODEL.
root=$(cd "$(dirname "$0")/.." && pwd)
export ISO_ONLY=1 REF=gmix MC_HP=0.05,0.15,0.35 N=4000 N_REG=1000 NVAL=750 NCAL=1000 EVAL_LINE=40 ONLY=nn,lp,cell NO_LPUS=1
case $2 in 2) export MC_M_CAL=16000 ;; 5) export MC_M_CAL=40000 ;; *) echo "unknown DY $2"; exit 2 ;; esac
case $1 in
  gaussian)   sc=gauss; main=main_n5k_gaussrej ;;
  multimodal) sc=moe;   main=main_n5k_matched; export MOE_K=4 MOE_S=0.02,0.5 MOE_G=3 ;;
  manifold)   sc=curve; main=main_n5k_matched; export CV_D0=0.02 CV_D1=0.02 CV_PURE=0 ;;
  *) echo "unknown MODEL $1"; exit 2 ;;
esac
[ -n "$FULL_CAL" ] || export TAU_FILE=$root/results/$main/$1/dY$2/rank_${sc}_4000_$3.jsonl
OUT_DIR=$root/results/curves_n5k_matched/$1 "$root/sim/run_rank.sh" $sc $3 $2
