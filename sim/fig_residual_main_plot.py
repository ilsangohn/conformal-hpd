"""Figure fig:residual from $FIG1_DIR (default results/fig1_main, written by fig_residual_main.py; 60x60 score grid,
Monte Carlo evaluation). Layout and colors of fig_residual_rank_plot.py. The first seed is the illustrated curve; shading
in (b) is the pointwise min/max over all replications, gray background the hard region of the model. Writes
figures/residual_fig_main.pdf (and .png) unless PDF is set, and $FIG1_DIR/figure_summary.json.
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
D = Path(os.environ.get("FIG1_DIR", str(root / "results/fig1_main")))
GAM = 0.9
seeds = sorted(int(p.stem.replace("rank_seed", "")) for p in D.glob("rank_seed*.npz"))
reps = [np.load(D / f"rank_seed{k}.npz") for k in seeds]
d0 = reps[0]
x1 = d0["x1"]
for r in reps:
    assert np.allclose(r["x1"], x1)
stack = {k: np.stack([r[k] for r in reps]) for k in ("R1", "R2", "cov1", "cov2", "u_x")}
corr2 = GAM - d0["u_x"]
gt1 = np.array([GAM - float(r["tau1"]) for r in reps])
band = lambda k: (stack[k].min(0), stack[k].max(0))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
for a in ax:
    hx1 = x1[d0["hard"]]
    if len(hx1):
        a.axvspan(hx1.min(), hx1.max(), color="0.92", lw=0, zorder=0)
    a.set_xlim(0, 1); a.set_xlabel("$x_1$")
n1, n2 = int(d0["n1"]), int(d0["n2"])
# (a) one-stage only: the residual varies with x, the single threshold corrects it only on average
ax[0].axhline(0, c="0.6", lw=0.8)
ax[0].fill_between(x1, *band("R1"), color="C3", alpha=0.15, lw=0)
ax[0].plot(x1, d0["R1"], c="C3", lw=1.8, label=r"residual $R(\gamma_\star,x)$")
ax[0].axhline(gt1[0], c="C3", ls="--", lw=1.6, label=r"correction $\gamma_\star-\hat\tau$")
ax[0].set_ylabel("coverage residual")
ax[0].set_title("(a) one-stage calibration")
lo, hi = min(stack["R1"].min(), gt1[0], 0), max(stack["R1"].max(), gt1[0])
ax[0].set_ylim(lo - 0.01, hi + 0.45 * (hi - lo))
ax[0].legend(fontsize=8.2, loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)
ax[1].axhline(GAM, c="0.4", ls=":", lw=1.2, label=r"target $\gamma_\star=0.9$")
ax[1].fill_between(x1, *band("cov1"), color="C3", alpha=0.15, lw=0)
ax[1].fill_between(x1, *band("cov2"), color="C0", alpha=0.15, lw=0)
ax[1].plot(x1, d0["cov1"], c="C3", lw=1.6, ls="--", label="one-stage")
ax[1].plot(x1, d0["cov2"], c="C0", lw=1.6, label="two-stage (Alg. 1)")
ax[1].set_ylabel("conditional coverage")
ax[1].set_title("(b) conditional coverage")
ylo, yhi = ax[1].get_ylim(); ax[1].set_ylim(ylo - 0.22 * (yhi - ylo), yhi)
ax[1].legend(fontsize=8.2, loc="lower left", frameon=False, handlelength=2.0, labelspacing=0.25, borderaxespad=0.2)
plt.tight_layout(w_pad=1.0)
out = os.environ.get("PDF", str(root / "figures/residual_fig_main.pdf"))
plt.savefig(out, bbox_inches="tight", pad_inches=0.08)
plt.savefig(os.path.splitext(out)[0] + ".png", dpi=180, bbox_inches="tight", pad_inches=0.08)

inR = d0["hard"]
e1, e2 = np.abs(stack["cov1"] - GAM), np.abs(stack["cov2"] - GAM)
js = [json.loads((D / f"rank_seed{k}.json").read_text()) for k in seeds]
summ = {"reps": len(reps), "seeds": seeds,
        "cerr1_all_mean": float(e1.mean()), "cerr2_all_mean": float(e2.mean()),
        "cerr1_per_rep": e1.mean(1).round(5).tolist(), "cerr2_per_rep": e2.mean(1).round(5).tolist(),
        "two_better_all": int((e2.mean(1) < e1.mean(1)).sum()),
        "cerr1_hard_mean": float(e1[:, inR].mean()), "cerr2_hard_mean": float(e2[:, inR].mean()),
        "two_better_hard": int((e2[:, inR].mean(1) < e1[:, inR].mean(1)).sum()),
        "cov1_min_all": float(stack["cov1"].min()), "cov2_min_seed0": float(d0["cov2"].min()),
        "cov2_min_all": float(stack["cov2"].min()),
        "two_undercover_x1_gt_0.9": int((stack["cov2"][:, x1 > 0.9] < GAM).any(1).sum()),
        "gam_minus_tau1": gt1.round(4).tolist(),
        "tau2": [round(float(r["tau2"]), 4) for r in reps],
        "regression_bandwidth": [j["regression_bandwidth"] for j in js],
        "lp_bandwidth_one_x_y": [[round(j["lp_bandwidth_one"][0], 4), round(j["lp_bandwidth_one"][3], 4)] for j in js],
        "lp_bandwidth_two_x_y": [[round(j["lp_bandwidth_two"][0], 4), round(j["lp_bandwidth_two"][3], 4)] for j in js],
        "kx_mean_line": [round(j["kx_mean_line"], 1) for j in js],
        "max_abs_cov2_minus_F2u": float(max(j["max_abs_cov2_minus_F2u"] for j in js)),
        "cerr2_fixed_all_mean": float(np.mean([j["cerr_two_fixed"] for j in js])) if "cerr_two_fixed" in js[0] else None,
        "regression_bandwidth_oldcv": [j.get("regression_bandwidth_oldcv") for j in js],
        "regression_bandwidth_fixed": js[0].get("regression_bandwidth_fixed"),
        "seconds_one_stage_calibration": [round(j["seconds_one_stage_calibration"], 1) for j in js],
        "seconds_two_stage_calibration": [round(j["seconds_two_stage_calibration"], 1) for j in js]}
(D / "figure_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
print(json.dumps(summ, indent=1))
