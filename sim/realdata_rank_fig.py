"""Figure fig:bike: estimated sets at 8 a.m. and 2 p.m. (temperature 0.5, humidity 0.6), first split of
results/bike_1031. Solid: one-stage calibration (HPD-split) with the network density fitted on the 11000 fitting hours
with the gmix contrastive reference; dashed: local ellipsoid (mean network and 150-NN residual covariance on the same
11000 hours); thin lines: contours of the network density; gray: all hours at the same hour of the day with
temperature within 0.08 and humidity within 0.12. Both sets are calibrated on the 1500 calibration hours as in
realdata_rank.py (HPD values of the calibration points on the 30 x 30 grid); the HPD set is drawn on an 80 x 80 grid,
the ellipse in full. The fits and thresholds reproduce the rows HPD-NN and ELLIPSOID of realdata_rank.py for split 0
(same code, seeds and reference); the thresholds are checked against RESULTS_JSONL.

usage (from the project root): python sim/realdata_rank_fig.py [OUT_PDF] [RESULTS_JSONL]
  OUT_PDF        default figures/bike_sets.pdf (a PNG is written next to it)
  RESULTS_JSONL  default results/bike_1031/rank_bike_0.jsonl
Environment: REF=gmix (the default here, as in results/bike_1031).
"""
import json, os, sys
os.environ.setdefault("REF", "gmix")
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import realdata_rank as RR
import sim_rank as SR
from sim_rank import BOX
from sim_rank import fit_mean_net

assert SR.REF == "gmix", SR.REF
seed = 0
out_pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures", "bike_sets.pdf")
res_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "bike_1031", "rank_bike_0.jsonl")
D = RR.split(seed)
XF, YF, YA, YB, Xc, Yc, VI = (D[k] for k in ("XF", "YF", "YA", "YB", "Xc", "Yc", "VI"))
x, y, scale = D["x"], D["y"], D["scale"]
SR.REF_OBJ = SR.Ref(YA)                                     # as in realdata_rank.run
ctx = RR.make_ctx(YB)

# one-stage network (11000 hours) and its HPD-split threshold (30 x 30 grid, as in the experiments)
nn_f, _, inf_f = RR.fit_density(ctx, XF, YF, VI, seed, "full", True, False)
sc, _ = RR.hpd_grid(RR.grid_dens(nn_f, Xc), nn_f.dens(np.hstack([Xc, Yc])))
tau_hpd = RR.tau_of(sc)
# local ellipsoid on the same hours (realdata_rank.run, ellipsoid)
mnet = fit_mean_net(XF, YF, seed)
resid = YF - mnet(XF); tree = cKDTree(XF)


def cov_at(Xq, k=150):
    _, ii = tree.query(Xq, k=k); R = resid[ii]; R = R - R.mean(1, keepdims=True)
    return np.einsum("nki,nkj->nij", R, R) / (k - 1) + 1e-4 * np.eye(2)


d_c = Yc - mnet(Xc)
tau_ell = RR.tau_of(np.sqrt(np.einsum("ni,nij,nj->n", d_c, np.linalg.inv(cov_at(Xc)), d_c)))
res = {r["method"]: r for r in map(json.loads, open(res_path)) if r.get("seed") == seed}
check = {"tau_hpd": tau_hpd, "tau_hpd_run": res["HPD-NN"]["tau"], "tau_ell": tau_ell, "tau_ell_run": res["ELLIPSOID"]["tau"],
         "nn_epochs": inf_f["nn_epochs"]}
assert abs(tau_hpd - check["tau_hpd_run"]) < 1e-9 and abs(tau_ell - check["tau_ell_run"]) < 1e-9, check

G = 80; g = np.linspace(-BOX, BOX, G + 1); g = (g[:-1] + g[1:]) / 2; cell = (g[1] - g[0]) ** 2
Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
lo = lambda v, j: v * scale[j]                              # back to log-count units (centered)
area_unit = cell * scale[0] * scale[1]


def closed_contour(ax, inside, **kw):
    """contour of a grid set, padded with one empty cell on each side so that a set reaching the box edge is closed"""
    d = g[1] - g[0]; gp = np.concatenate([[g[0] - d], g, [g[-1] + d]])
    ip = np.zeros((G + 2, G + 2)); ip[1:-1, 1:-1] = inside
    GXp, GYp = np.meshgrid(lo(gp, 0), lo(gp, 1), indexing="ij")
    ax.contour(GXp, GYp, ip, levels=[0.5], **kw)


plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
ext, areas = [], {}
th = np.linspace(0, 2 * np.pi, 400)
for ax, (hr, title) in zip(axes, ((8, "8 a.m."), (14, "2 p.m."))):
    x0 = np.array([[hr / 23, 0.5, 0.6]])
    E = np.hstack([np.repeat(x0, len(Yg), 0), Yg])
    pd = nn_f.dens(E); pn = pd / pd.sum()
    o = np.argsort(-pn); cs = np.cumsum(pn[o]); s = np.empty(len(pn)); s[o] = cs - pn[o]
    in_hpd = (s <= tau_hpd).reshape(G, G)
    S = cov_at(x0)[0]; el = mnet(x0)[0] + tau_ell * (np.linalg.cholesky(S) @ np.vstack([np.cos(th), np.sin(th)])).T
    areas[title] = {"one_stage_nn": float(in_hpd.sum() * area_unit),
                    "ellipsoid": float(np.pi * tau_ell ** 2 * np.sqrt(np.linalg.det(S)) * scale[0] * scale[1])}
    near = (np.abs(x[:, 0] - hr / 23) < 0.5 / 23) & (np.abs(x[:, 1] - 0.5) < 0.08) & (np.abs(x[:, 2] - 0.6) < 0.12)
    ax.scatter(lo(y[near, 0], 0), lo(y[near, 1], 1), s=6, c="0.55", alpha=0.6)
    GX, GY = np.meshgrid(lo(g, 0), lo(g, 1), indexing="ij")
    ax.contour(GX, GY, pd.reshape(G, G), levels=8, colors="C0", linewidths=0.4, alpha=0.5)
    closed_contour(ax, in_hpd, colors="C0", linewidths=1.8)
    ax.plot(lo(el[:, 0], 0), lo(el[:, 1], 1), c="C3", lw=1.5, ls="--")
    d = (g[1] - g[0]) / 2
    i0, i1 = np.flatnonzero(in_hpd.any(1)), np.flatnonzero(in_hpd.any(0))
    ext.append((lo(g[i0.min()] - d, 0), lo(g[i0.max()] + d, 0), lo(g[i1.min()] - d, 1), lo(g[i1.max()] + d, 1)))
    ext.append((lo(el[:, 0].min(), 0), lo(el[:, 0].max(), 0), lo(el[:, 1].min(), 1), lo(el[:, 1].max(), 1)))
    ax.set_title(f"{title}, temperature 0.5, humidity 0.6", fontsize=9); ax.set_xlabel("log casual rentals (centered)")
axes[0].set_ylabel("log registered rentals (centered)")
ext = np.array(ext)
axes[0].set_xlim(min(-1.6, ext[:, 0].min() - 0.15), max(3.2, ext[:, 1].max() + 0.15))
axes[0].set_ylim(min(-0.8, ext[:, 2].min() - 0.15), max(3.6, ext[:, 3].max() + 0.15))
axes[0].scatter([], [], s=6, c="0.55", label="observations near $x$")
axes[0].plot([], [], c="C0", lw=1.8, label="one-stage (network)")
axes[0].plot([], [], c="C3", lw=1.5, ls="--", label="local ellipsoid")
axes[0].legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(out_pdf); plt.savefig(os.path.splitext(out_pdf)[0] + ".png", dpi=110)
print(json.dumps({**check, "areas": areas, "out": out_pdf}))
