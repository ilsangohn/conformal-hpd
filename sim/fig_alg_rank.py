"""Schematic of rank calibration (Algorithm 1) on a toy model with d_X = d_Y = 1 (figure fig:alg).

X ~ U[0,1]; Y | x is a symmetric two-component Gaussian mixture whose modes separate and widen with x.
The ranking function a_hat(. | x) is the Gaussian density with the conditional mean and variance, so that its
super-level sets are intervals with the wrong shape; the selected level depends only on their order at x.
Panel (a): regression points; at the target x0 every neighbour is scored by a_hat(. | x0), and the
fixed candidate set U_{x0}(t_hat) (shaded) is applied to all points of the window.
Panel (b): a_hat(. | x0) and the true density p(. | x0) on the response axis, with the level t_hat, the oracle
set C*(x0) (bars) and the true coverages of U_{x0}(t_hat) and C*(x0).
Panel (c): estimated coverage Q_hat (weighted counts), its envelope Q_bar, the true coverage Q of the candidate
sets and the mass that a_hat assigns to them (used by one-stage calibration); Q_bar falls to tau at t_hat.

usage: python fig_alg_rank.py [out.pdf]   (default ../figures/alg_rank.pdf)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(3)
GAMMA, X0, H = 0.9, 0.6, 0.22


def mix(y, x):
    m, s = 0.9 + 0.7 * x, 0.3 + 0.15 * x
    return 0.5 * norm.pdf(y, -m, s) + 0.5 * norm.pdf(y, m, s)


def sample(n):
    x = rng.uniform(0, 1, n); m, s = 0.9 + 0.7 * x, 0.3 + 0.15 * x
    sign = np.where(rng.uniform(size=n) < 0.5, -1.0, 1.0)
    return x, sign * m + s * rng.standard_normal(n)


p = lambda y, x: mix(y, x)                                                                  # true conditional density
ahat = lambda y, x: norm.pdf(y, 0.0, np.sqrt((0.9 + 0.7 * x) ** 2 + (0.3 + 0.15 * x) ** 2))   # Gaussian ranking function

X, Y = sample(400)
# local linear equivalent-kernel weights at X0 (triweight kernel, as in the simulations)
D = (X - X0) / H; k = np.clip(1 - D ** 2, 0, None) ** 3
F = np.stack([np.ones_like(D), D], 1); Gm = (F * k[:, None]).T @ F
w = k * (F @ np.linalg.solve(Gm, np.array([1.0, 0.0])))
a = ahat(Y, X0)
yy = np.linspace(-3.2, 3.2, 4001); dy = yy[1] - yy[0]
ay, py = ahat(yy, X0), p(yy, X0)
tgrid = np.linspace(0, ay.max() * 1.02, 600)
Qhat = np.array([(w * (a >= t)).sum() for t in tgrid])
Qbar = np.clip(np.maximum.accumulate(Qhat[::-1])[::-1], 0, 1)
Qtrue = np.array([(py * (ay >= t)).sum() * dy for t in tgrid])
Mest = np.array([(ay * (ay >= t)).sum() for t in tgrid]) / ay.sum()      # mass that a_hat assigns to U_x(t)
that = tgrid[np.argmax(Qbar <= GAMMA)]
inB = ay >= that
tone = tgrid[np.argmax(Mest <= GAMMA)]
srt = np.sort(py)[::-1]; tstar = srt[np.argmax(np.cumsum(srt) * dy >= GAMMA)]
inC = py >= tstar
covB, covC = (py * inB).sum() * dy, (py * inC).sum() * dy
print(f"t_hat={that:.4f}  coverage U_x(t_hat)={covB:.3f}  coverage C*={covC:.3f}  "
      f"one-stage level coverage={(py * (ay >= tone)).sum() * dy:.3f}  volumes {inB.sum() * dy:.2f} vs {inC.sum() * dy:.2f}  "
      f"nonzero weights={(w != 0).sum()}")


def intervals(mask):
    e = np.flatnonzero(np.diff(mask.astype(int)))
    return list(zip(yy[e[::2] + 1], yy[e[1::2]]))


ints, intsC = intervals(inB), intervals(inC)
YL = (-3.2, 4.5)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.35, 0.75, 1.1]})
# (a)
win = np.abs(X - X0) <= H
inside = ahat(Y, X0) >= that
ax[0].scatter(X[~win], Y[~win], s=4, c="0.75", lw=0)
ax[0].scatter(X[win & inside], Y[win & inside], s=6, c="C0", lw=0, label=r"$\hat a(Y_i\mid x)\geq\hat t_x$")
ax[0].scatter(X[win & ~inside], Y[win & ~inside], s=6, c="C1", lw=0, label=r"$\hat a(Y_i\mid x)<\hat t_x$")
for lo, hi in ints:
    ax[0].fill_between([X0 - H, X0 + H], lo, hi, color="C0", alpha=0.12, lw=0)
ax[0].axvline(X0, color="k", lw=0.8, ls="--"); ax[0].axvspan(X0 - H, X0 + H, color="0.9", alpha=0.5, lw=0, zorder=0)
ax[0].set_xlim(0, 1); ax[0].set_ylim(*YL); ax[0].set_yticks(range(-3, 4)); ax[0].set_xlabel("covariate"); ax[0].set_ylabel("response $y$")
ax[0].set_xticks([0, X0, 1]); ax[0].set_xticklabels(["0", "$x$", "1"])
ax[0].set_title("(a) neighbours scored at $x$", loc="left")
ax[0].legend(loc="upper left", fontsize=7, frameon=False, handletextpad=0.2, markerscale=2, ncol=2, columnspacing=0.6)
# (b)
xr = max(py.max(), ay.max()) * 1.12
ax[1].plot(py, yy, color="k", lw=1.0, label=r"$\mathsf{p}(\cdot\mid x)$")
ax[1].plot(ay, yy, color="C0", lw=1.2, label=r"$\hat a(\cdot\mid x)$")
ax[1].axvline(that, color="C0", lw=0.8, ls=":")
for lo, hi in ints:
    ax[1].axhspan(lo, hi, color="C0", alpha=0.12, lw=0)
for j, (lo, hi) in enumerate(intsC):
    ax[1].plot([xr, xr], [lo, hi], color="k", lw=2.2, solid_capstyle="butt", label=r"$\mathsf{C}^\star(x)$" if j == 0 else None)
ax[1].text(0.62 * xr, 0.0, f"{covB:.2f}", color="C0", fontsize=8, ha="center", va="center")
ax[1].text(xr * 1.05, intsC[0][0] - 0.12, f"{covC:.2f}", color="k", fontsize=8, ha="right", va="top")
ax[1].set_xlim(0, xr * 1.08); ax[1].set_ylim(*YL); ax[1].set_yticks(range(-3, 4)); ax[1].set_yticklabels([])
ax[1].set_xlabel("value at $x$"); ax[1].set_xticks([0, that]); ax[1].set_xticklabels(["0", r"$\hat t_x$"])
ax[1].set_title("(b) ranking at $x$", loc="left")
ax[1].legend(loc="upper right", bbox_to_anchor=(1.03, 1.03), fontsize=7, frameon=False, handlelength=1.2, labelspacing=0.25)
# (c)
hb, = ax[2].step(tgrid, Qbar, where="post", color="C0", lw=2.8, alpha=0.45, zorder=1, label=r"$\overline{Q}_x$")
hh, = ax[2].step(tgrid, Qhat, where="post", color="C4", lw=0.8, zorder=3, label=r"$\widehat{Q}_x$")
hq, = ax[2].plot(tgrid, Qtrue, color="k", lw=1.0, label=r"$Q_x$")
hm, = ax[2].plot(tgrid, Mest, color="C0", lw=1.0, ls="--", label=r"mass under $\hat a$")
ax[2].axhline(GAMMA, color="k", lw=0.6, ls="--"); ax[2].axvline(that, color="C0", lw=0.8, ls=":")
ax[2].text(tgrid[-1], GAMMA + 0.03, r"$\hat\tau$", ha="right", va="bottom", fontsize=8)
ax[2].set_xticks([0, that]); ax[2].set_xticklabels(["0", r"$\hat t_x$"])
ax[2].set_xlabel("level $t$"); ax[2].set_ylabel(r"coverage of $\mathcal{U}_x(t)$"); ax[2].set_ylim(-0.02, 1.05)
ax[2].set_title("(c) estimated coverage", loc="left")
ax[2].legend(handles=[hh, hb, hq, hm], loc="lower left", fontsize=7, frameon=False, handlelength=1.4)   # caption order
plt.tight_layout(w_pad=0.6)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "figures", "alg_rank.pdf")
plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
plt.savefig(os.path.splitext(out)[0] + ".png", dpi=180, bbox_inches="tight", pad_inches=0.05)
print("saved", out)
