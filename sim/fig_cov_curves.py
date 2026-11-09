"""Coverage along x_1 (Section 8): run from the project root, python sim/fig_cov_curves.py [RESULT_ROOT].

  figures/cov_curves_main.pdf   conditional coverage at the 40 covariates (x_1, 0.5, 0.5), x_1 = 0.03, ..., 0.97, of
                                one-stage calibration (dashed) and Algorithm 1 (solid) for NN, LP and the partition
                                estimate, averaged over replications; models in columns, d_Y in rows; gray: hard region
Data: RESULT_ROOT (default results/curves_n5k_matched)/{gaussian,multimodal,manifold}/dY{2,5}/rank_*.jsonl, seeds 20-39,
written by run_curves_matched.sh (sim_rank.py with EVAL_LINE=40): the row "EVAL" holds the covariates and their regions,
every method row its coverages "cov_x". Jobs count only with exit=0 (main_data.read; SEEDS=a-b keeps those seeds). The fits
and the calibration are those of the main experiment for the same seed; one-stage thresholds are read from its result files
(TAU_FILE), so these runs have no CHCDS and PCP rows, and only the evaluation covariates differ.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main_data import MODELS, DYS, read, cell

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/curves_n5k_matched"
runs = read(ROOT)
EST = (("NN", "HPD-NN", "#ee5a61", "#bd1820"), ("LP", "HPD-LP", "#5c9cf0", "#1d5bc4"),
       ("partition", "CELL", "#58c46f", "#1c8c37"))

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.linewidth": 0.6})
fig, axes = plt.subplots(len(DYS), len(MODELS), figsize=(6.2, 4.4), sharex=True, sharey="row")
reps = {}
for i, d in enumerate(DYS):
    for k, (name, disp) in enumerate(MODELS):
        ax, R = axes[i, k], cell(runs, name, d)
        reps[f"{name}{d}"] = len(R)
        if not R:
            ax.set_axis_off(); continue
        x1 = np.array(R[0]["EVAL"]["x"])[:, 0]; hard = np.array(R[0]["EVAL"]["hard"])
        edges = np.flatnonzero(np.diff(np.r_[0, hard.astype(int), 0]))      # runs of consecutive hard covariates
        half = (x1[1] - x1[0]) / 2
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(max(x1[a] - half, 0), min(x1[b - 1] + half, 1), color="0.92", lw=0, zorder=0)
        ax.axhline(0.9, color="k", lw=0.6, zorder=1)
        for lab, pre, c1, c2 in EST:
            for m, c, ls, lw in ((pre, c1, (0, (4, 1.5)), 1.1), (pre + "-RANKISO", c2, "-", 1.3)):
                if all(m in r for r in R):
                    cal = "one-stage" if m == pre else "two-stage (Alg. 1)"
                    ax.plot(x1, np.mean([r[m]["cov_x"] for r in R], axis=0), color=c, ls=ls, lw=lw, zorder=3,
                            label=f"{lab}, {cal}")
        ax.set_xlim(0, 1); ax.tick_params(length=2.5, pad=1.5)
        if i == 0:
            ax.set_title(disp)
        if i == len(DYS) - 1:
            ax.set_xlabel("$x_1$")
    axes[i, 0].set_ylabel(rf"$d_Y={d}$" + "\nconditional coverage")
h, l = next((a.get_legend_handles_labels() for a in axes.flat if a.get_legend_handles_labels()[0]), ([], []))
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.2,
           handlelength=2.4, columnspacing=1.2, labelspacing=0.3, borderaxespad=0.1)
plt.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.8, w_pad=0.5)
plt.savefig("figures/cov_curves_main.pdf", bbox_inches="tight", pad_inches=0.06)
plt.savefig("figures/cov_curves_main.png", dpi=160, bbox_inches="tight", pad_inches=0.06)
print("replications per panel:", reps)
