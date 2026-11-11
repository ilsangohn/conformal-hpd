"""Main experiment figures (Section 8): run from the project root, python sim/fig_main.py [RESULT_ROOT].

  figures/main_plane.pdf   conditional coverage error against shape loss, models in columns and d_Y in rows
Data: main_data.load (RESULT_ROOT/{gaussian,multimodal,manifold}/dY{2,5}/rank_*.jsonl, default
results/main_n5k_matched and, for the Gaussian model, results/main_n5k_gaussrej; jobs with exit=0, SEEDS=a-b). Two-stage = rank calibration with the isotropic frozen-target CV
bandwidth (-RANKISO). The dotted vertical line is the conditional coverage error of one-stage calibration (HPD-split) with the true density
(ORACLE-COND), whose HPD values are exactly uniform given x: the error left by the calibration and evaluation samples.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.textpath import TextPath
from main_data import MODELS, DYS, load, cerr, cell as cell_of

runs = load(sys.argv[1] if len(sys.argv) > 1 else None)
cell = lambda name, d: cell_of(runs, name, d)


# ---------------- styles (as the earlier coverage--shape plane): hue = density estimate, marker = calibration
FAMC = {"NN": {"one": "#ee5a61", "two": "#bd1820", "q": "#f7a3a7", "loc": "#f7a3a7", "edge": "#7a0b12"},
        "LP": {"one": "#5c9cf0", "two": "#1d5bc4", "q": "#a9cbf7", "loc": "#a9cbf7", "edge": "#0d2f75"},
        "cell": {"one": "#58c46f", "two": "#1c8c37", "q": "#a6dcb2", "loc": "#a6dcb2", "edge": "#0b4d1f"}}
MK = {"one": "o", "two": "D", "q": "^", "loc": "s"}
ZO = {"q": 3, "loc": 3, "one": 5, "two": 6}
STYLE = {}
for fam, pre, lab in (("NN", "HPD-NN", "NN"), ("LP", "HPD-LP", "LP"), ("cell", "CELL", "partition")):
    for key, suf, cal in (("one", "", "one-stage"), ("two", "-RANKISO", "two-stage (Alg. 1)"), ("q", "-RCP", "two-stage (quantile)"),
                          ("loc", "-LOC", "two-stage (own covariate)")):
        if key in FAMC[fam]:
            STYLE[pre + suf] = (FAMC[fam][key], MK[key], f"{lab}, {cal}", FAMC[fam]["edge"], ZO[key])
for m, c, lab in (("CHCDS", "#ffcc99", "CHCDS"),
                  ("ELLIPSOID", "#e3d6a3", "local ellipsoid"), ("RCP-ELLIPSOID", "#f5bdd6", "RCP (ellipsoid score)"),
                  ("PCP", "#d6b8a0", "PCP"), ("RCP-PCP", "#a6dbd7", "RCP (PCP score)"), ("OT-CP", "#c9c9d6", "OT-CP")):
    STYLE[m] = (c, "o", lab, "w", 2)
ARROWS = {"NN": ("HPD-NN", "HPD-NN-RANKISO"), "LP": ("HPD-LP", "HPD-LP-RANKISO"), "cell": ("CELL", "CELL-RANKISO")}

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.linewidth": 0.6})

# ---------------- coverage--shape plane
FIGH, LH = 5.2, 0.95                                  # figure and legend heights (inches)
fig, axes = plt.subplots(2, 3, figsize=(6.2, FIGH))
for i, d in enumerate(DYS):
    for k, (name, disp) in enumerate(MODELS):
        ax, R = axes[i, k], cell(name, d)
        if not R:
            ax.set_axis_off(); continue
        P = {m: (np.mean([cerr(r[m]) for r in R]), np.mean([r[m]["shape"] for r in R])) for m in STYLE
             if all(m in r for r in R)}
        xy = lambda m: P[m]
        ax.axvline(np.mean([cerr(r["ORACLE-COND"]) for r in R]), color="0.45", ls=(0, (1, 1.5)), lw=0.9, zorder=1,
                   label="one-stage, true density")
        for m in STYLE:
            if m not in P:
                continue
            c, mk, lab, ec, zo = STYLE[m]
            size = 4.6 if mk in ("D", "s") else 5.2
            ax.plot(*xy(m), ls="none", marker=mk, color=c, ms=size, mec=ec, mew=(0.6 if ec != "w" else 0.4), label=lab, zorder=zo)
        for fam, (a, b) in ARROWS.items():
            ax.annotate("", xy=xy(b), xytext=xy(a), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=FAMC[fam]["edge"], lw=0.7, mutation_scale=7, shrinkA=3.5, shrinkB=4))
        ax.set_xscale("log"); ax.set_yscale("log")
        for a_ in (ax.xaxis,):
            a_.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
            a_.set_major_formatter(mticker.FormatStrFormatter("%g")); a_.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(length=2.5, pad=1.5)
        if i == 0:
            ax.set_title(disp)
    axes[i, 0].set_ylabel(rf"$d_Y={d}$" + "\n" + r"shape loss $\mathcal{S}^\circ$ / $\lambda(\mathsf{C}^\star)$")
axes[1, 1].set_xlabel("conditional coverage error (region-averaged)")
plt.tight_layout(rect=(0, (LH + 0.06) / FIGH, 1, 1), h_pad=0.8, w_pad=0.4)
# legend within the width of the panels, in two framed groups: the HPD sets, a table with the calibrations in rows and
# the density estimates in columns (CHCDS: network only; dotted line: one-stage with the true density), and the other
# methods (coordinates in inches from the lower left corner below the panels; text widths from TextPath)
p0, p1 = axes[1, 0].get_position(), axes[1, 2].get_position()
W = (p1.x1 - p0.x0) * fig.get_figwidth()
lax = fig.add_axes([p0.x0, 0.0, p1.x1 - p0.x0, LH / FIGH]); lax.set_axis_off()
lax.set_xlim(0, W); lax.set_ylim(0, LH)
FS, PAD, DY = 7.0, 0.08, 0.135
TOP = LH - 0.05; ROWY = [TOP - 0.12 - k * DY for k in range(6)]          # header row, then five rows
wd = lambda t: TextPath((0, 0), t, size=FS).get_extents().width / 72
txt = lambda x, y, t, **kw: lax.text(x, y, t, va="center", fontsize=FS, **kw)


def mark(x, y, m):
    c, mk, _, ec, _ = STYLE[m]
    lax.plot([x], [y], ls="none", marker=mk, color=c, ms=(4.6 if mk in ("D", "s") else 5.2) * 0.85, mec=ec,
             mew=(0.6 if ec != "w" else 0.4))


def frame(x0, x1, title):
    lax.add_patch(plt.Rectangle((x0, 0.02), x1 - x0, TOP - 0.02, fill=False, lw=0.6, ec="0.6"))
    txt(x0 + PAD, TOP, title, ha="left", bbox=dict(fc="w", ec="none", pad=0.8))


ROWS = (("one-stage", ""), ("two-stage (Alg. 1)", "-RANKISO"), ("two-stage (own covariate)", "-LOC"),
        ("two-stage (quantile)", "-RCP"), ("CHCDS", None))                # the row order of the tables
OTH = ("ELLIPSOID", "RCP-ELLIPSOID", "PCP", "RCP-PCP", "OT-CP")
LW, OW, GAP = max(wd(r) for r, _ in ROWS), max(wd(STYLE[m][2]) for m in OTH), 0.2
W1 = PAD + LW + 0.18 + 0.68 + wd("partition") / 2 + 0.1 + wd("true density") + PAD     # widths of the two frames
W2 = PAD + 0.15 + OW + PAD
X0 = (W - W1 - GAP - W2) / 2                                  # the two frames centered below the panels
xc = X0 + PAD + LW + 0.18
for y, (r, _) in zip(ROWY[1:], ROWS):
    txt(X0 + PAD, y, r, ha="left")
for pre, lab, step in (("HPD-NN", "NN", 0.3), ("HPD-LP", "LP", 0.38), ("CELL", "partition", 0.0)):
    txt(xc, ROWY[0], lab, ha="center")
    for y, (_, suf) in zip(ROWY[1:], ROWS):
        if suf is not None or pre == "HPD-NN":
            mark(xc, y, pre + suf if suf is not None else "CHCDS")
    xc += step
xt = xc + wd("partition") / 2 + 0.1 + wd("true density") / 2
txt(xt, ROWY[0], "true density", ha="center")
lax.plot([xt, xt], [ROWY[1] - 0.05, ROWY[1] + 0.05], color="0.45", ls=(0, (1, 1.5)), lw=0.9)
frame(X0, X0 + W1, "HPD sets")
x2 = X0 + W1 + GAP + PAD
for y, m in zip(ROWY[1:], OTH):
    mark(x2 + 0.05, y, m); txt(x2 + 0.15, y, STYLE[m][2], ha="left")
frame(x2 - PAD, x2 - PAD + W2, "other methods")
plt.savefig("figures/main_plane.pdf", bbox_inches="tight", pad_inches=0.06)
plt.savefig("figures/main_plane.png", dpi=160, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
