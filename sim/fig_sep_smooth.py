"""Illustration of the smooth counterexample of Theorem 7.2 (thm:sep-smooth).
(a) level sets at the level t_star(x): the perturbation of height delta on a shell of half-width w around the level set raises the
    density on the upper half and lowers it on the lower half;
(b) profile across the level set on the raised half for two shells of the same height delta and different widths: the
    super-level set moves by the same amount, so the misplaced layer and the shape loss are the same, while the error
    (bottom) has mass proportional to the width.
Usage (from sim/): python fig_sep_smooth.py [output.pdf]; default ../figures/sep_smooth.pdf. A PNG is written next to the PDF."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "../figures/sep_smooth.pdf"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
ORANGE, NAVY, TEAL = "#f2a65a", "#0b2a6f", "#2a9d8f"


def step(u):   # smooth step: 0 for u <= 0, 1 for u >= 1
    e = lambda v: np.where(v > 0, np.exp(-1 / np.maximum(v, 1e-12)), 0.0)
    return e(u) / (e(u) + e(1 - u))


chi = lambda u: step(2 * (1 - np.abs(u)))                  # chi = 1 on [-1/2, 1/2], 0 outside (-1, 1)
a = lambda u: 2 * step(u + 0.5) - 1                        # odd, a = 1 on [1/2, inf), a = -1 on (-inf, -1/2]

# ---------------- (a) level sets of a bivariate radial density and of p_delta
f0 = lambda r: np.exp(-r ** 2 / 2) / (2 * np.pi)
rstar = 1.4; t = f0(rstar); delta, w = 0.2 * t, 0.3
phat = lambda x, y: f0(np.hypot(x, y)) + delta * chi((np.hypot(x, y) - rstar) / w) * a(y / rstar)
g = np.linspace(-2.0, 2.0, 900); X, Y = np.meshgrid(g, g)
P, Ph = f0(np.hypot(X, Y)), phat(X, Y)

fig = plt.figure(figsize=(5.8, 2.75))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[2.1, 1], hspace=0.08, wspace=0.28)
ax1 = fig.add_subplot(gs[:, 0])
ax1.add_patch(matplotlib.patches.Annulus((0, 0), rstar + w, 2 * w, color="0.88", lw=0, zorder=0))
ax1.contourf(X, Y, ((Ph >= t) != (P >= t)).astype(float), levels=[0.5, 1.5], colors=[ORANGE], zorder=1)
ax1.contour(X, Y, P, levels=[t], colors="k", linewidths=1.0)
ax1.contour(X, Y, Ph, levels=[t], colors=NAVY, linewidths=1.0, linestyles="--")
for ang in np.deg2rad([40, 90, 140]):
    ax1.text(1.85 * np.cos(ang), 1.85 * np.sin(ang), "+", ha="center", va="center", fontsize=9)
    ax1.text(1.85 * np.cos(ang), -1.85 * np.sin(ang), "−", ha="center", va="center", fontsize=9)
ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2); ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_xlabel("$y_1$"); ax1.set_ylabel("$y_2$", rotation=0, labelpad=6)
ax1.set_title(r"(a) level sets at the level $t_\star(x)$")

# ---------------- (b) profile across the level set on the raised half (schematic, slope 1 at the level)
s = np.linspace(-1.0, 1.0, 801); T, d = 1.0, 0.18
WIDE, THIN = 0.85, 0.42
p = T - s
ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
ax2.axvspan(0, d, color=ORANGE, alpha=0.9, lw=0, zorder=0)
ax2.axhline(T, color="0.55", lw=0.7, zorder=1)
ax2.plot(s, p, "k", lw=1.1, label=r"$\mathsf{p}(\cdot\mid x)$", zorder=3)
ax2.plot(s, p + d * chi(s / WIDE), color=NAVY, ls="--", lw=1.1, label=r"$\mathsf{p}_\delta(\cdot\mid x)$ ($w$)", zorder=4)
ax2.plot(s, p + d * chi(s / THIN), color=TEAL, ls=(0, (1, 1.2)), lw=1.4, label=r"$\mathsf{p}_\delta(\cdot\mid x)$ ($w'$)", zorder=5)
ax2.text(-0.97, T + 0.03, r"level $t_\star(x)$", ha="left", va="bottom", fontsize=7.5, color="0.35")
ax2.annotate("", xy=(0, T - 0.42), xytext=(d, T - 0.42), arrowprops=dict(arrowstyle="<->", lw=0.7, shrinkA=0, shrinkB=0))
ax2.text(d / 2, T - 0.47, r"layer $\asymp\delta$", ha="center", va="top", fontsize=7.5)
ax2.set_ylim(T - 0.62, T + 0.62); ax2.set_yticks([]); ax2.set_ylabel("density")
ax2.tick_params(labelbottom=False, length=0)
ax2.set_title("(b) across the level set on the raised half")
ax2.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.8, borderaxespad=0.4, labelspacing=0.25)
# error profiles: same height delta, mass proportional to the width
ax3.fill_between(s, 0, d * chi(s / WIDE), color=NAVY, alpha=0.18, lw=0)
ax3.plot(s, d * chi(s / WIDE), color=NAVY, ls="--", lw=1.0)
ax3.fill_between(s, 0, d * chi(s / THIN), color=TEAL, alpha=0.45, lw=0)
ax3.plot(s, d * chi(s / THIN), color=TEAL, ls=(0, (1, 1.2)), lw=1.4)
ax3.annotate("", xy=(0.93, 0), xytext=(0.93, d), arrowprops=dict(arrowstyle="<->", lw=0.7, shrinkA=0, shrinkB=0))
ax3.text(0.9, d / 2, r"$\delta$", ha="right", va="center", fontsize=8)
ax3.set_ylim(-0.012, 0.23); ax3.set_yticks([]); ax3.set_ylabel(r"$(\mathsf{p}_\delta-\mathsf{p})(\cdot\mid x)$", fontsize=7)
ax3.set_xticks([-WIDE, -THIN, 0, THIN, WIDE]); ax3.set_xticklabels(["$-w$", "$-w'$", "0", "$w'$", "$w$"], fontsize=7.5)
ax3.set_xlabel("signed distance from the level set (outward)", fontsize=8)
ax3.set_xlim(-1.0, 1.0)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.9)
plt.savefig(OUT)
plt.savefig(OUT.rsplit(".", 1)[0] + ".png", dpi=150)
