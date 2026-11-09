#!/bin/sh
# Main experiment (Section 8, Tables 3-5 and Figure 4 of the supplement and paper), one job per call.
# Usage: scripts/run_main.sh MODEL DY SEED   with MODEL in {gaussian, multimodal, manifold}, DY in {2, 5}, SEED in 20..39.
# Output: results/main_n5k_gaussrej/gaussian (Gaussian model) or results/main_n5k_matched/MODEL (other models).
# One job takes about 0.3-1 hours (d_Y = 2) and 1-12 hours (d_Y = 5) on one core, with at most about 5 GB of memory.
root=$(cd "$(dirname "$0")/.." && pwd)
export ISO_ONLY=1 REF=gmix MC_HP=0.05,0.15,0.35 N=4000 N_REG=1000 NVAL=750 NCAL=1000 ONLY=nn,lp,cell,ell,oracle NO_LPUS=1
case $2 in 2) export MC_M_CAL=16000 ;; 5) export MC_M_CAL=40000 ;; *) echo "unknown DY $2"; exit 2 ;; esac
case $1 in
  gaussian)   sc=gauss; out=$root/results/main_n5k_gaussrej/gaussian ;;
  multimodal) sc=moe;   out=$root/results/main_n5k_matched/multimodal; export MOE_K=4 MOE_S=0.02,0.5 MOE_G=3 ;;
  manifold)   sc=curve; out=$root/results/main_n5k_matched/manifold;   export CV_D0=0.02 CV_D1=0.02 CV_PURE=0 ;;
  *) echo "unknown MODEL $1"; exit 2 ;;
esac
OUT_DIR=$out "$root/sim/run_rank.sh" $sc $3 $2
