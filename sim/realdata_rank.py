"""Bike-sharing application (tab:bike, fig:bike) with rank calibration (Algorithm 1 = alg:local) and the tuning
conventions of sim_rank.py (whose functions are imported; md5 recorded in the output).

usage: python realdata_rank.py SEED        one random split; JSON lines on stdout (one per method, then one "INFO" line)
       python realdata_rank.py selftest    unit tests

Data and preprocessing: realdata.load(seed), unchanged: Y = (log(1 + casual + U1), log(1 + registered + U2)) with
U ~ U(0,1) drawn afresh per split, each coordinate mapped affinely into [-2.5, 2.5] (0.1% and 99.9% quantiles over all
hours to -2.25 and 2.25, clipped); X = (hour/23, temp, hum); 15 evaluation groups (5 hour blocks x 3 temperature
terciles over all hours). The permutation is that of realdata.py (np.random.default_rng(seed + 1).permutation).

Sizes (environment; defaults): the first N_A = 8000 permuted hours fit the base model of the two-stage and other
auxiliary-step methods, the next N_REG = 3000 are the regression (auxiliary) points, and one-stage methods fit on these
N_A + N_REG = 11000 hours (A first). NCAL = 1500 calibration hours follow; all remaining hours (4879; or the first
N_TEST) are the test set. With the defaults, A = old train (6500) + old validation (1500), D_reg = old D2,
calibration = old calibration, test = old test (all of it, not only the first 3000).
Tuning: the last NVAL = 1500 of the first N_A hours (= the old validation hours) are held out from every density fit,
the same hours for every estimate and both fitting samples. For each candidate value of its tuning parameters an
estimate is fitted on its fitting sample without these hours; the value with the smallest held-out loss is selected,
and the estimate is refitted on its whole fitting sample (11000 or 8000 hours) with it (as fit_density / fit_cell /
the KDE block of sim_rank.run): NN number of epochs (sim_rank.fit_net_tuned); LP bandwidths on LP_HX x LP_HY, no rate
factor; KDE (h_x, h_y); partition (m, h_Y). NN and LP: contrastive log-loss on 3000 random contrastive rows of the
tuning hours; KDE: conditional log-likelihood on the first 800 tuning hours; partition: on all 1500.

HPD values (one-stage, -A, -RCP, -LOC): on the G x G = 30 x 30 grid of cell midpoints of the response box, as in
realdata.py (HPD value of y at x = normalized grid mass of the cells where the estimate exceeds its value at y).
Every set is evaluated on the same grid; areas are numbers of grid cells times the cell area in log-count units.
Rank calibration needs no HPD values: its score at (x, y) is Qbar_x(a(y | x)) with a = the density estimate of the
8000-hour fit (sim_rank.rank_table / rank_eval, weights sim_rank.ll_weights_batch over the 3000 regression hours).

Contrastive reference (NN and LP): REF of sim_rank (environment, read there at import), unif or gmix; with REF=gmix
the reference is sim_rank.Ref of the N_A responses and the NN cap is 0.9999, as in sim_rank.run.

Methods (rows):
  one-stage, fit on 11000: HPD-NN, HPD-LP, HPD-KDE, CELL (partition), CHCDS (NN, grid level), PCP (50 draws from the
    NN grid density, jittered in the cell), ELLIPSOID (mean network + 150-NN residual covariance).
  auxiliary step, fit on 8000, D_reg = 3000:
    HPD-NN-RANKISO, HPD-LP-RANKISO, CELL-RANKISO  (Algorithm 1 with the bandwidth rule of the simulations: isotropic
      frozen-target CV, sim_rank.frozen_cv, on H_BIKE)
    HPD-NN-RANK, HPD-LP-RANK, CELL-RANK  (reference: Algorithm 1; regression bandwidth by sim_rank.ind_cv on H_GRID)
    HPD-NN-RANKFIX, HPD-LP-RANKFIX, CELL-RANKFIX  (reference: bandwidth sim_rank.h_fixed(3000))
    HPD-NN-LOC, HPD-LP-LOC, CELL-LOC  (two-stage (own covariate): the same regression of the HPD values of the
      regression hours at their own covariates; bandwidth by ind_cv on -s)
    HPD-NN-RCP, HPD-LP-RCP, CELL-RCP  (two-stage (quantile): HPD score / k-NN 0.9-quantile on D_reg, k by pinball CV)
    HPD-NN-A, HPD-LP-A, CELL-A  (diagnostic: one-stage calibration of the 8000-hour fit)
    RCP-ELLIPSOID, RCP-PCP (PCP scores of all 3000 regression hours), OT-CP (all 3000 regression residuals matched to
      3000 uniform draws on the unit disk; score = norm of the mean match of the 5 nearest residuals).
Metrics on the test set: marginal coverage; coverage in the 15 groups, mean and max |cov_g - 0.9|; mean set area.

Tuning grids. On the bike data the LP and KDE selections hit the lower edges of the simulation grids (split 0: LP
(h_x, h_y) = (0.6, 0.2) in both fits, KDE h_x = 0.2), so the LP and KDE grids are extended by the same ratios
(environment TUNE_GRID, default "ext"): LP h_x in {0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.35, 2.0}, h_y in {0.14, 0.2,
0.3, 0.45, 0.65, 1.0}; KDE h_x in {0.07, 0.1, 0.14, 0.2, 0.35, 0.5, 0.7, 1.0} (h_y grid unchanged). TUNE_GRID=sim gives
the grids of sim_rank.py (LP_HX, LP_HY; KDE grid of sim_rank.run). All other grids are those of sim_rank.py; the
partition grid m in {2, 3, 4, 5} can be extended to {2, 3, 4, 5, 6, 7, 8, 10} by CELL_GRID=ext (diagnostic only).
ONLY (optional, comma-separated subset of nn, lp, kde, cell, ell) restricts the method groups (tests).
H_BIKE (environment, comma-separated; default 0.1, 0.15, 0.22, 0.3, 0.4, 0.55, 0.75, 1.0, 1.5): H_GRID extended
downward, since on H_GRID the frozen-target CV selected its lower edge 0.22 in 28 of 30 split x estimate cases
(results/bike_1030/final, rows -RANKFCV).
RANK_SWEEP=1 adds diagnostic rows -RANKH<h>: Algorithm 1 with the fixed bandwidth h, every h of H_BIKE.
"""
import hashlib, itertools, json, os, sys, time
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import realdata as RD                                   # load(seed): data, preprocessing, groups (unchanged)
import sim_rank as SR
from sim_rank import (crows, fit_net_tuned, eta2dens, rank_table, rank_eval, ll_weights_batch, ind_cv, h_fixed,
                      LP_HX, LP_HY, H_GRID, grid_edge, CellFit, CKDE, EtaFit)
from llfast_batch import local_linear_fast
from sim_contrastive import BOX, bce

K, ALPHA, GAMMA, DX, DY = SR.K, SR.ALPHA, SR.GAMMA, 3, 2
assert K == RD.K == 5 and ALPHA == RD.ALPHA
KDE_HX, KDE_HY = (0.2, 0.35, 0.5, 0.7, 1.0), (0.05, 0.07, 0.1, 0.14, 0.2, 0.3)      # as the KDE block of sim_rank.run
CELL_M, CELL_HY = (2, 3, 4, 5), (0.1, 0.15, 0.2, 0.3, 0.45)                         # as sim_rank fit_cell
KNN_GRID = (25, 50, 100, 200, 400, 600, 1000)                                       # as sim_rank knn_quantile_cv
TUNE_GRID = os.environ.get("TUNE_GRID", "ext")
if TUNE_GRID == "ext":                                  # LP and KDE grids extended to lower bandwidths (see the docstring)
    LP_HX, LP_HY = (0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.35, 2.0), (0.14, 0.2, 0.3, 0.45, 0.65, 1.0)
    KDE_HX = (0.07, 0.1, 0.14, 0.2, 0.35, 0.5, 0.7, 1.0)
else:
    assert TUNE_GRID == "sim", TUNE_GRID
CELL_GRID = os.environ.get("CELL_GRID", "sim")
if CELL_GRID == "ext":
    CELL_M = (2, 3, 4, 5, 6, 7, 8, 10)
else:
    assert CELL_GRID == "sim", CELL_GRID
ONLY = os.environ.get("ONLY")
SWEEP = os.environ.get("RANK_SWEEP") == "1"             # diagnostics: Algorithm 1 at every h of H_BIKE
H_BIKE = tuple(float(v) for v in os.environ.get("H_BIKE", "0.1,0.15,0.22,0.3,0.4,0.55,0.75,1.0,1.5").split(","))
assert SR.H_PILOT in H_BIKE
want = lambda grp: not ONLY or grp in ONLY.split(",")
G = int(os.environ.get("GRID", "30"))
_g = np.linspace(-BOX, BOX, G + 1); GMID = (_g[:-1] + _g[1:]) / 2; CELL_AREA = (GMID[1] - GMID[0]) ** 2
YG = np.stack(np.meshgrid(GMID, GMID, indexing="ij"), -1).reshape(-1, 2); NG = len(YG)
MD5_SR = hashlib.md5(open(SR.__file__, "rb").read()).hexdigest()
STAGE = defaultdict(float)
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:7.1f}s]", *a, file=sys.stderr, flush=True)


class Timer:
    def __init__(self, stage=None):
        self.stage = stage
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        if self.stage:
            STAGE[self.stage] += self.dt


class Ctx:
    pass


def make_ctx(YB):
    """the attributes of sim_rank.Ctx used by EtaFit, CellFit and CKDE: dY, YB (regression responses), model.Q"""
    ctx = Ctx(); ctx.dY = DY; ctx.YB = YB
    ctx.model = Ctx(); ctx.model.Q = 1.0 / (2 * BOX) ** DY
    return ctx


# ====================================================================== data
def sizes():
    return {"N_A": int(os.environ.get("N_A", "8000")), "N_REG": int(os.environ.get("N_REG", "3000")),
            "NVAL": int(os.environ.get("NVAL", "1500")), "NCAL": int(os.environ.get("NCAL", "1500")),
            "N_TEST": int(os.environ.get("N_TEST", "0"))}


def split(seed, sz=None):
    sz = sz or sizes()
    x, y, grp, scale = RD.load(seed)
    idx = np.random.default_rng(seed + 1).permutation(len(x))
    nA, nB, nC = sz["N_A"], sz["N_REG"], sz["NCAL"]
    iA, iB, iC, iT = idx[:nA], idx[nA:nA + nB], idx[nA + nB:nA + nB + nC], idx[nA + nB + nC:]
    if sz["N_TEST"] > 0:
        iT = iT[:sz["N_TEST"]]
    assert 0 < sz["NVAL"] < nA
    VI = np.arange(nA - sz["NVAL"], nA)                 # tuning hours: positions in the fitting samples (A first)
    return {"x": x, "y": y, "grp": grp, "scale": scale, "idx": idx, "iA": iA, "iB": iB, "iC": iC, "iT": iT,
            "VI": VI, "XA": x[iA], "YA": y[iA], "XB": x[iB], "YB": y[iB], "Xc": x[iC], "Yc": y[iC],
            "Xt": x[iT], "Yt": y[iT], "gt": grp[iT], "XF": x[np.r_[iA, iB]], "YF": y[np.r_[iA, iB]],
            "area_unit": CELL_AREA * scale[0] * scale[1], "sz": sz}


# ====================================================================== density fits (tune without VI, refit on all)
def fit_density(ctx, Xf, Yf, VI, seed, tag, do_nn=True, do_lp=True):
    """NN and LP of sim_rank.run.fit_density: contrastive rows with K references (sim_rank.crows); tuning rows = the rows of the
    fitting sample without the tuning hours VI (points and their references); held-out rows = 3000 random rows of VI;
    NN epochs by early stopping, then retrained on all rows; LP (hx, hy) on LP_HX x LP_HY by held-out log-loss, then
    evaluated with all rows."""
    n = len(Xf); tr = np.setdiff1d(np.arange(n), VI)
    Z, L = crows(Xf, Yf, np.random.default_rng(seed + (0 if tag == "full" else 77)))
    rows = lambda I: np.concatenate([I, n + (I[:, None] * K + np.arange(K)).ravel()])   # point i and its K references
    rtr = rows(tr)
    rv = np.random.default_rng(seed + 5).choice(rows(VI), size=min(3000, len(VI) * (K + 1)), replace=False)
    Ztr, Ltr, Zvs, Lvs = Z[rtr], L[rtr], Z[rv], L[rv]
    with Timer("fit_nn") as t1:
        prn, n_ep = fit_net_tuned(Ztr, Ltr, Zvs, Lvs, Z, L, seed) if do_nn else (None, 0)
    poly = list(range(DX + DY))
    best, lp_cv, lp = (np.nan, None, None), {}, None
    if do_lp:
        with Timer("fit_lp") as t2:
            best = None
            for hx, hy in itertools.product(LP_HX, LP_HY):
                sc = np.array([hx] * DX + [hy] * DY)
                loss = bce(local_linear_fast(Ztr, Ltr, Zvs, sc, poly, ydim=DY)[0], Lvs); lp_cv[f"{hx},{hy}"] = round(float(loss), 6)
                if best is None or loss < best[0]:
                    best = (loss, hx, hy)
            sc_f = np.array([best[1]] * DX + [best[2]] * DY)
        lp = EtaFit(ctx, lambda A: local_linear_fast(Z, L, A, sc_f, poly, ydim=DY)[0], "LP")
    else:
        t2 = Timer(); t2.dt = 0.0
    nn = EtaFit(ctx, prn, "NN", cap=0.995 if SR.REF_OBJ is None else 0.9999) if do_nn else None
    info = {"nn_epochs": int(n_ep), "nn_edge": "high" if n_ep >= 200 else ("low" if n_ep <= 1 else "interior"),
            "lp_hx": best[1], "lp_hy": best[2], "lp_cv": lp_cv,
            "lp_edge": [grid_edge([lp_cv[f"{h},{best[2]}"] for h in LP_HX], LP_HX),
                        grid_edge([lp_cv[f"{best[1]},{h}"] for h in LP_HY], LP_HY)] if do_lp else None,
            "lp_grid": [list(LP_HX), list(LP_HY)],
            "n_fit": n, "n_tune_fit": len(tr), "n_tune_rows": len(rv), "nn_fit_s": round(t1.dt, 1), "lp_fit_s": round(t2.dt, 1)}
    log(f"fit {tag}: nn {t1.dt:.1f}s, lp {t2.dt:.1f}s, epochs {n_ep}, lp ({best[1]}, {best[2]}) edge {info['lp_edge']}")
    return nn, lp, info


def fit_cell(ctx, Xf, Yf, VI, XV, YV):
    tr = np.setdiff1d(np.arange(len(Xf)), VI)
    with Timer("fit_cell") as t:
        best, cv = None, {}
        for m_, hy in itertools.product(CELL_M, CELL_HY):
            f = CellFit(ctx, Xf[tr], Yf[tr], m_, hy)
            ll = np.mean(np.log(np.maximum(f.dens(np.hstack([XV, YV])), 1e-12))); cv[f"{m_},{hy}"] = round(float(ll), 6)
            if best is None or ll > best[0]:
                best = (ll, m_, hy)
        f = CellFit(ctx, Xf, Yf, best[1], best[2])
    f.fit_s = t.dt
    return f, {"m": best[1], "hy": best[2], "cell_cv": cv,
               "cell_edge": [grid_edge([-cv[f"{m_},{best[2]}"] for m_ in CELL_M], CELL_M),
                             grid_edge([-cv[f"{best[1]},{h}"] for h in CELL_HY], CELL_HY)]}


def fit_kde(ctx, Xf, Yf, VI, XV, YV):
    tr = np.setdiff1d(np.arange(len(Xf)), VI)
    with Timer("fit_kde") as t:
        best, cv = None, {}
        for hx, hy in itertools.product(KDE_HX, KDE_HY):
            f = CKDE(ctx, Xf[tr], Yf[tr], hx, hy)
            ll = np.mean(np.log(np.maximum(f.dens(np.hstack([XV[:800], YV[:800]])), 1e-8))); cv[f"{hx},{hy}"] = round(float(ll), 6)
            if best is None or ll > best[0]:
                best = (ll, hx, hy)
        f = CKDE(ctx, Xf, Yf, best[1], best[2])
    f.fit_s = t.dt
    return f, {"kde_hx": best[1], "kde_hy": best[2], "kde_cv": cv,
               "kde_edge": [grid_edge([-cv[f"{h},{best[2]}"] for h in KDE_HX], KDE_HX),
                            grid_edge([-cv[f"{best[1]},{h}"] for h in KDE_HY], KDE_HY)]}


# ====================================================================== grid HPD values
def grid_dens(fit, T, ch=400):
    """(len(T), NG) densities on the response grid at the covariates T (partition: one evaluation per cube)"""
    out = np.empty((len(T), NG))
    if isinstance(fit, CellFit):
        c = fit.cid(T)
        for cc in np.unique(c):
            out[c == cc] = np.maximum(fit.kde(cc, YG), 1e-12)
        return out
    for s0 in range(0, len(T), ch):
        Tc = T[s0:s0 + ch]
        out[s0:s0 + ch] = fit.dens(np.hstack([np.repeat(Tc, NG, 0), np.tile(YG, (len(Tc), 1))])).reshape(len(Tc), NG)
    return out


def hpd_grid(pg, po):
    """HPD values on the grid (realdata.py, vectorized): own point s = normalized mass of the cells with pg > po;
    grid cell = normalized mass of the cells ranked above it (decreasing order)."""
    pn = pg / pg.sum(1, keepdims=True)
    s_own = (pn * (pg > po[:, None])).sum(1)
    o = np.argsort(-pn, axis=1)
    ps = np.take_along_axis(pn, o, 1)
    s_grid = np.empty_like(pn); np.put_along_axis(s_grid, o, np.cumsum(ps, 1) - ps, 1)
    return s_own, s_grid


# ====================================================================== rank calibration
def rank_scores(fit, XB, hs, T, own, grid=None, ch=50):
    """Algorithm 1 at the covariates T for each bandwidth in hs, sharing the evaluations a(Y_i | x) (as sim_rank.rank):
    local linear weights over D_reg, a_i = a(Y_i | x) at x for the regression points in the union of the windows,
    (u, R) = rank_table(a, w), score = rank_eval(u, R, a(y | x)). own: (nT,) values a(y | x) of the points to score;
    grid: (nT, NG) values a(y_g | x) or None. Returns ([s_own per h], [s_grid per h or None], kx (len(hs), nT))."""
    nT = len(T)
    s_own = [np.empty(nT) for _ in hs]
    s_grid = [np.empty((nT, NG)) if grid is not None else None for _ in hs]
    kx = np.zeros((len(hs), nT), int)
    for s0 in range(0, nT, ch):
        Tc = T[s0:s0 + ch]
        Ws = [ll_weights_batch(XB, Tc, h) for h in hs]
        idxs = [np.flatnonzero(np.any(np.stack([W[i] != 0 for W in Ws]), axis=0)) for i in range(len(Tc))]
        av = fit.aux_vals(Tc, idxs)
        for i in range(len(Tc)):
            I = idxs[i]
            for v, W in enumerate(Ws):
                w = W[i][I]; kx[v, s0 + i] = int((w != 0).sum())
                u, R = rank_table(av[i], w)
                s_own[v][s0 + i] = rank_eval(u, R, np.array([own[s0 + i]]))[0]
                if grid is not None:
                    s_grid[v][s0 + i] = rank_eval(u, R, grid[s0 + i])
    return s_own, s_grid, kx


def loc_scores(sB, XB, h, T, s_own, s_grid=None, ch=200):
    """two-stage (own covariate), as sim_rank.own_cov: the regression of Algorithm 1 applied to the HPD values sB of
    the regression points at their own covariates (ranking value -s), score = Qbar_x(-s(x, y))."""
    out_o = np.empty(len(T)); out_g = np.empty((len(T), NG)) if s_grid is not None else None
    for s0 in range(0, len(T), ch):
        for i, w in enumerate(ll_weights_batch(XB, T[s0:s0 + ch], h)):
            I = np.flatnonzero(w != 0); u, R = rank_table(-sB[I], w[I])
            out_o[s0 + i] = rank_eval(u, R, np.array([-s_own[s0 + i]]))[0]
            if s_grid is not None:
                out_g[s0 + i] = rank_eval(u, R, -s_grid[s0 + i])
    return out_o, out_g


def knn_quantile_cv(Xtr, Vtr, seed, grid=KNN_GRID):
    """as sim_rank.run.knn_quantile_cv: k by five-fold CV of the pinball loss of the k-NN 0.9-quantile"""
    idx = np.random.default_rng(seed + 3000).permutation(len(Xtr)); folds = np.array_split(idx, 5); bestk = None; cv = {}
    for k in grid:
        if k >= 0.8 * len(Xtr):
            continue
        loss = 0.0
        for f in folds:
            tr = np.setdiff1d(idx, f)
            _, ii = cKDTree(Xtr[tr]).query(Xtr[f], k=k); q = np.quantile(Vtr[tr][ii], 1 - ALPHA, axis=1)
            u = Vtr[f] - q; loss += np.mean(np.maximum((1 - ALPHA) * u, -ALPHA * u))
        cv[k] = round(float(loss), 7)
        if bestk is None or loss < bestk[0]:
            bestk = (loss, k)
    k = bestk[1]; tree = cKDTree(Xtr)
    return k, (lambda Xq: np.quantile(Vtr[tree.query(Xq, k=k)[1]], 1 - ALPHA, axis=1)), \
        {"k": int(k), "k_cv": {str(a): b for a, b in cv.items()}, "k_edge": grid_edge(list(cv.values()), list(cv))}


# ====================================================================== evaluation
def tau_of(s_cal):
    n = len(s_cal); k = int(np.ceil((1 - ALPHA) * (n + 1)))
    return np.inf if k > n else float(np.sort(s_cal)[k - 1])


def sampling_floor(ngrp):
    """mean over groups of E|Bin(n_g, 0.9)/n_g - 0.9| (exact)"""
    from scipy.stats import binom
    out = []
    for n in ngrp:
        k = np.arange(n + 1); out.append(float(np.sum(binom.pmf(k, n, GAMMA) * np.abs(k / n - GAMMA))))
    return float(np.mean(out))


class Reporter:
    def __init__(self, D, base):
        self.D, self.base = D, base
        self.groups = np.unique(D["gt"]); self.results = {}

    def __call__(self, name, s_cal, s_test, s_grid, extra=None):
        tau = tau_of(s_cal)
        cov = s_test <= tau
        cg = np.array([cov[self.D["gt"] == j].mean() for j in self.groups])
        area = (s_grid <= tau).sum(1) * self.D["area_unit"]
        res = {**self.base, "method": name, "marg": float(cov.mean()), "cerr_mean": float(np.mean(np.abs(cg - GAMMA))),
               "cerr_max": float(np.max(np.abs(cg - GAMMA))), "area": float(area.mean()), "tau": tau,
               "cg": [round(float(c), 5) for c in cg], "area_full_frac": float(np.mean((s_grid <= tau).all(1)))}
        res.update(extra or {})
        print(json.dumps(res), flush=True)
        self.results[name] = res
        log(f"{name}: marg {res['marg']:.4f} cerr {res['cerr_mean']:.4f}/{res['cerr_max']:.4f} area {res['area']:.3f}")


# ====================================================================== one split
def run(seed):
    D = split(seed); sz = D["sz"]
    XA, YA, XB, YB, Xc, Yc, Xt, Yt, XF, YF, VI = (D[k] for k in ("XA", "YA", "XB", "YB", "Xc", "Yc", "Xt", "Yt", "XF", "YF", "VI"))
    XV, YV = XA[VI], YA[VI]
    nC, nT = len(Xc), len(Xt)
    SR.REF_OBJ = SR.Ref(YA) if SR.REF == "gmix" else None   # contrastive reference from the N_A responses (sim_rank.run)
    base = {"seed": seed, "n_A": len(XA), "n_reg": len(XB), "n_val": len(VI), "n_cal": nC, "n_test": nT, "grid": G,
            "tune_grid": TUNE_GRID, "cell_grid": CELL_GRID, "sweep": SWEEP, "ref": SR.REF}
    rep = Reporter(D, base)
    ctx = make_ctx(YB)
    info = {"seed": seed, "method": "INFO", **base, "md5_sim_rank": MD5_SR, "hpd_values": f"grid {G}x{G} (realdata.py)",
            "only": ONLY, "grids": {"LP": [list(LP_HX), list(LP_HY)], "KDE": [list(KDE_HX), list(KDE_HY)],
                                    "CELL": [list(CELL_M), list(CELL_HY)], "H_GRID": list(H_GRID), "H_BIKE": list(H_BIKE),
                                    "KNN": list(KNN_GRID)}}
    rr = np.random.default_rng(seed + 7)                # PCP draws: full cal, full test, A cal, A reg, A test

    def pcp_draws(prow):
        ii = rr.choice(NG, size=50, p=prow / prow.sum())
        return YG[ii] + rr.uniform(-0.5, 0.5, size=(50, 2)) * (GMID[1] - GMID[0])

    def pcp_scores(PG, Yq, grid=False):
        so = np.empty(len(PG)); sg = np.empty((len(PG), NG)) if grid else None
        for i in range(len(PG)):
            tr_ = cKDTree(pcp_draws(PG[i]))
            so[i] = tr_.query(Yq[i])[0]
            if grid:
                sg[i] = tr_.query(YG)[0]
        return so, sg

    def one_stage(fit, name, extra=None):
        with Timer("onestage_" + name) as t:
            pgc, poc = grid_dens(fit, Xc), fit.dens(np.hstack([Xc, Yc]))
            pgt, pot = grid_dens(fit, Xt), fit.dens(np.hstack([Xt, Yt]))
            sc, _ = hpd_grid(pgc, poc); st, sg = hpd_grid(pgt, pot)
        rep(name, sc, st, sg, {"time_s": round(t.dt, 1), **(extra or {})})
        return {"pgc": pgc, "poc": poc, "pgt": pgt, "pot": pot, "sc": sc, "st": st, "sg": sg, "t": t.dt}

    def ellipsoid(Xf, Yf):
        from sim_final import fit_mean_net
        with Timer("fit_mean"):
            mnet = fit_mean_net(Xf, Yf, seed)
        res = Yf - mnet(Xf); tree = cKDTree(Xf)
        def prec_at(Xq, k=150):
            _, ii = tree.query(Xq, k=k); R = res[ii]; R = R - R.mean(1, keepdims=True)
            return np.linalg.inv(np.einsum("nki,nkj->nij", R, R) / (k - 1) + 1e-4 * np.eye(DY))
        def score(Xq, Yq):
            d = Yq - mnet(Xq); return np.sqrt(np.einsum("ni,nij,nj->n", d, prec_at(Xq), d))
        def score_grid(Xq):
            d = YG[None] - mnet(Xq)[:, None]; return np.sqrt(np.einsum("ngi,nij,ngj->ng", d, prec_at(Xq), d))
        return mnet, score, score_grid

    # ================= one-stage methods: fit on N_A + N_REG hours
    nn_f, lp_f, info["fit_full"] = fit_density(ctx, XF, YF, VI, seed, "full", want("nn"), want("lp"))
    if want("nn"):
        r = one_stage(nn_f, "HPD-NN", {"nn_epochs": info["fit_full"]["nn_epochs"]})
        with Timer("chcds") as t:
            def level(PG):
                o = -np.sort(-PG, axis=1); cs = np.cumsum(o, 1) / o.sum(1, keepdims=True)
                k = np.minimum((cs < 1 - ALPHA).sum(1), NG - 1)
                return o[np.arange(len(o)), k]
            lc, lt = level(r["pgc"]), level(r["pgt"]); zc, zt = r["pgc"].sum(1) * CELL_AREA, r["pgt"].sum(1) * CELL_AREA
        rep("CHCDS", (lc - r["poc"]) / zc, (lt - r["pot"]) / zt, (lt[:, None] - r["pgt"]) / zt[:, None],
            {"time_s": round(r["t"] + t.dt, 1)})
        with Timer("pcp") as t:
            sc_p, _ = pcp_scores(r["pgc"], Yc); st_p, sg_p = pcp_scores(r["pgt"], Yt, grid=True)
        rep("PCP", sc_p, st_p, sg_p, {"time_s": round(r["t"] + t.dt, 1)})
        del r, sg_p
    if want("lp"):
        r = one_stage(lp_f, "HPD-LP", {"lp_hx": info["fit_full"]["lp_hx"], "lp_hy": info["fit_full"]["lp_hy"]}); del r
    del nn_f, lp_f
    if want("kde"):
        kde, info["kde"] = fit_kde(ctx, XF, YF, VI, XV, YV)
        log(f"kde: {info['kde']['kde_hx']}, {info['kde']['kde_hy']} edge {info['kde']['kde_edge']}")
        r = one_stage(kde, "HPD-KDE", {"kde_hx": info["kde"]["kde_hx"], "kde_hy": info["kde"]["kde_hy"]}); del r, kde
    if want("cell"):
        cell_f, info["cell_full"] = fit_cell(ctx, XF, YF, VI, XV, YV)
        log(f"cell full: m {info['cell_full']['m']} hy {info['cell_full']['hy']} edge {info['cell_full']['cell_edge']}")
        r = one_stage(cell_f, "CELL", {"m": info["cell_full"]["m"], "hy": info["cell_full"]["hy"]}); del r, cell_f
    if want("ell"):
        with Timer("ellipsoid") as t:
            _, ell_f, ell_fg = ellipsoid(XF, YF)
            ec, et, eg = ell_f(Xc, Yc), ell_f(Xt, Yt), ell_fg(Xt)
        rep("ELLIPSOID", ec, et, eg, {"time_s": round(t.dt, 1)}); del eg

    # ================= auxiliary-step methods: base on N_A hours, auxiliary step on the N_REG regression hours
    fits = []
    if want("nn") or want("lp"):
        nn_a, lp_a, info["fit_A"] = fit_density(ctx, XA, YA, VI, seed, "A", want("nn"), want("lp"))
        if want("nn"):
            fits.append(("HPD-NN", nn_a, {"nn_epochs": info["fit_A"]["nn_epochs"]}))
        if want("lp"):
            fits.append(("HPD-LP", lp_a, {"lp_hx": info["fit_A"]["lp_hx"], "lp_hy": info["fit_A"]["lp_hy"]}))
        del nn_a, lp_a
    if want("cell"):
        cell_a, info["cell_A"] = fit_cell(ctx, XA, YA, VI, XV, YV)
        log(f"cell A: m {info['cell_A']['m']} hy {info['cell_A']['hy']} edge {info['cell_A']['cell_edge']}")
        fits.append(("CELL", cell_a, {"m": info["cell_A"]["m"], "hy": info["cell_A"]["hy"]}))
        del cell_a
    hf = h_fixed(len(XB)); info["h_fixed"] = hf
    while fits:
        name, fit, ex = fits.pop(0)
        r = one_stage(fit, name + "-A", ex)            # diagnostic: one-stage calibration of the same fit
        with Timer("aux_hpd") as ta:
            pgB, poB = grid_dens(fit, XB), fit.dens(np.hstack([XB, YB]))
            sB, _ = hpd_grid(pgB, poB)
        with Timer("rcp") as tq:                        # two-stage (quantile): RCP on the HPD score
            kq, qfun, kinfo = knn_quantile_cv(XB, sB, seed)
            qc, qt = np.maximum(qfun(Xc), 1e-6), np.maximum(qfun(Xt), 1e-6)
        rep(name + "-RCP", r["sc"] / qc, r["st"] / qt, r["sg"] / qt[:, None],
            {"time_s": round(r["t"] + ta.dt + tq.dt, 1), **kinfo, **ex})
        info[name + "-RCP"] = kinfo
        with Timer("loc") as tl:                        # two-stage (own covariate)
            hl, ll_loss = ind_cv(-sB, XB, seed)
            lc_, _ = loc_scores(sB, XB, hl, Xc, r["sc"]); lt_, lg_ = loc_scores(sB, XB, hl, Xt, r["st"], r["sg"])
        linfo = {"hx": hl, "cv_loss": [round(float(c), 7) for c in ll_loss], "cv_edge": grid_edge(ll_loss, H_GRID)}
        rep(name + "-LOC", lc_, lt_, lg_, {"time_s": round(r["t"] + ta.dt + tl.dt, 1), **linfo, **ex})
        info[name + "-LOC"] = linfo; del lg_
        if name == "HPD-NN":                            # PCP draws of the 8000-hour network: cal, regression, test
            with Timer("pcp") as tp:
                s_pcp_c, _ = pcp_scores(r["pgc"], Yc); s_pcp_B, _ = pcp_scores(pgB, YB)
                s_pcp_t, s_pcp_g = pcp_scores(r["pgt"], Yt, grid=True)
        del pgB
        with Timer("rank") as tr_:                      # Algorithm 1
            aB = fit.dens(np.hstack([XB, YB]))
            hx, cv_loss = ind_cv(aB, XB, seed)
            with Timer("rank_fcv") as tf:              # bandwidth of -RANKISO
                hfc, fc_loss, fc_nskip = SR.frozen_cv(XB, aB, fit.aux_vals, seed, grid=H_BIKE)
            hs = [hx, hf, hfc] + (list(H_BIKE) if SWEEP else [])
            S_c, _, kxc = rank_scores(fit, XB, hs, Xc, r["poc"])
            S_t, S_g, kxt = rank_scores(fit, XB, hs, Xt, r["pot"], r["pgt"])
            (sc1, sc2), (st1, st2), (sg1, sg2) = S_c[:2], S_t[:2], S_g[:2]
        kx = np.concatenate([kxc, kxt], axis=1)
        rinfo = {"hx": hx, "cv_loss": [round(float(c), 7) for c in cv_loss], "cv_edge": grid_edge(cv_loss, H_GRID),
                 "cv_grid": list(H_GRID), "hx_fix": hf, "kx_mean": float(kx[0].mean()), "kx_min": int(kx[0].min()),
                 "nzero": int((kx[0] == 0).sum())}
        rep(name + "-RANK", sc1, st1, sg1, {"time_s": round(tr_.dt, 1), **rinfo, **ex})
        rep(name + "-RANKFIX", sc2, st2, sg2, {"time_s": round(tr_.dt, 1), "hx": hf, "kx_mean": float(kx[1].mean()),
                                               "kx_min": int(kx[1].min()), "nzero": int((kx[1] == 0).sum()), **ex})
        info[name + "-RANK"] = rinfo
        finfo = {"hx": hfc, "fcv_loss": [round(float(c), 7) for c in fc_loss], "fcv_nskip": int(fc_nskip),
                 "fcv_grid": list(H_BIKE), "fcv_edge": grid_edge(fc_loss, H_BIKE), "fcv_s": round(tf.dt, 1),
                 "kx_mean": float(kx[2].mean()), "kx_min": int(kx[2].min()), "nzero": int((kx[2] == 0).sum())}
        rep(name + "-RANKISO", S_c[2], S_t[2], S_g[2], {"time_s": round(tr_.dt, 1), **finfo, **ex})
        info[name + "-RANKISO"] = finfo
        if SWEEP:
            for v, h in enumerate(H_BIKE):
                rep(f"{name}-RANKH{h}", S_c[3 + v], S_t[3 + v], S_g[3 + v],
                    {"hx": h, "kx_mean": float(kx[3 + v].mean()), "nzero": int((kx[3 + v] == 0).sum()), **ex})
        del r, sg1, sg2, sB, fit, S_g

    if want("ell"):
        with Timer("ellipsoid") as t:
            mnet_a, ell_a, ell_ag = ellipsoid(XA, YA)
            vB = ell_a(XB, YB); kq, qfun, kinfo = knn_quantile_cv(XB, vB, seed)
            qc, qt = qfun(Xc), qfun(Xt)
            sc_e, st_e, sg_e = ell_a(Xc, Yc) / qc, ell_a(Xt, Yt) / qt, ell_ag(Xt) / qt[:, None]
        rep("RCP-ELLIPSOID", sc_e, st_e, sg_e, {"time_s": round(t.dt, 1), **kinfo}); info["RCP-ELLIPSOID"] = kinfo; del sg_e
    if want("nn"):
        with Timer("pcp") as t:
            kq, qfun, kinfo = knn_quantile_cv(XB, s_pcp_B, seed)
            qc, qt = qfun(Xc), qfun(Xt)
        rep("RCP-PCP", s_pcp_c / qc, s_pcp_t / qt, s_pcp_g / qt[:, None], {"time_s": round(tp.dt + t.dt, 1), **kinfo})
        info["RCP-PCP"] = kinfo; del s_pcp_g
    if want("ell"):
        with Timer("otcp") as t:
            from scipy.optimize import linear_sum_assignment
            rv = YB - mnet_a(XB); m_ = len(rv)                  # all regression residuals
            ro = np.random.default_rng(seed + 11)
            ang = ro.uniform(0, 2 * np.pi, m_); rad = np.sqrt(ro.uniform(size=m_))
            Ub = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
            ri, ci = linear_sum_assignment(((rv[:, None, :] - Ub[None]) ** 2).sum(-1)); Tm = np.empty_like(Ub); Tm[ri] = Ub[ci]
            rtree = cKDTree(rv)
            ot_rank = lambda R: np.linalg.norm(Tm[rtree.query(R, k=5)[1]].mean(-2), axis=-1)
            mt = mnet_a(Xt); sg_o = np.empty((nT, NG))
            for s0 in range(0, nT, 500):
                sg_o[s0:s0 + 500] = ot_rank((YG[None] - mt[s0:s0 + 500, None]).reshape(-1, 2)).reshape(-1, NG)
            sc_o, st_o = ot_rank(Yc - mnet_a(Xc)), ot_rank(Yt - mt)
        rep("OT-CP", sc_o, st_o, sg_o, {"time_s": round(t.dt, 1), "n_ot": m_})

    ngrp = [int((D["gt"] == j).sum()) for j in rep.groups]
    info.update({"group_sizes": ngrp, "sampling_floor": sampling_floor(ngrp),
                 "stage_s": {k: round(v, 1) for k, v in sorted(STAGE.items())}, "total_s": round(time.perf_counter() - T0, 1)})
    print(json.dumps(info), flush=True)
    log(f"# total {time.perf_counter() - T0:.1f}s")


# ====================================================================== tests
def selftest():
    rng = np.random.default_rng(3)
    # 1. split = the split of realdata.py with train + validation as the 8000-hour fitting sample A
    D = split(0, {"N_A": 8000, "N_REG": 3000, "NVAL": 1500, "NCAL": 1500, "N_TEST": 0})
    x, y, grp, scale = RD.load(0)
    itr, iva, i2, ica, ite = np.split(np.random.default_rng(1).permutation(len(x)), np.cumsum([6500, 1500, 3000, 1500]))
    assert np.array_equal(D["iA"], np.r_[itr, iva]) and np.array_equal(D["iA"][D["VI"]], iva)
    assert np.array_equal(D["iB"], i2) and np.array_equal(D["iC"], ica) and np.array_equal(D["iT"], ite)
    assert np.array_equal(D["y"], y) and len(np.unique(np.r_[D["iA"], D["iB"], D["iC"], D["iT"]])) == len(x) == 17379
    print(f"split: A = old train + validation, tuning hours = old validation, D_reg = old D2, cal = old cal, "
          f"test = old test ({len(ite)} hours); identical responses: True")
    # 2. grid HPD values: vectorized = loop of realdata.py
    pg = np.maximum(rng.gamma(0.5, size=(40, NG)), 1e-12); po = pg[np.arange(40), rng.integers(NG, size=40)] * rng.uniform(0.5, 1.5, 40)
    so, sg = hpd_grid(pg, po)
    pn = pg / pg.sum(1, keepdims=True)
    so2 = np.array([pn[i][pg[i] > po[i]].sum() for i in range(40)]); sg2 = np.empty_like(pn)
    for i in range(40):
        o = np.argsort(-pn[i]); cs = np.cumsum(pn[i][o]); sg2[i, o] = cs - pn[i][o]
    print(f"hpd_grid vs realdata.py loop: max diff own {np.abs(so - so2).max():.1e}, grid {np.abs(sg - sg2).max():.1e}")
    assert np.allclose(so, so2) and np.allclose(sg, sg2)
    # 3. rank score = sim_rank's construction (weights over the window, rank_table, rank_eval) = brute force Qbar
    XB = rng.uniform(size=(400, 3)); YB = rng.normal(size=(400, 2)) * 0.8
    afun = lambda X, Y: np.round(np.exp(-((Y - X[:, :2]) ** 2).sum(1)), 2)            # rounding creates ties
    class Toy:
        def aux_vals(self, T, idxs):
            return [afun(np.repeat(T[t][None], len(i), 0), YB[i]) for t, i in enumerate(idxs)]
    T = np.vstack([rng.uniform(size=(30, 3)), [[5.0, 5.0, 5.0]]])                   # last: empty window
    yq = rng.normal(size=(len(T), 2)) * 0.8; own = afun(T, yq)
    grid = np.stack([afun(np.repeat(T[t][None], NG, 0), YG) for t in range(len(T))])
    hs = [0.3, 0.55]
    so_, sg_, kx = rank_scores(Toy(), XB, hs, T, own, grid, ch=7)
    worst = 0.0
    for v, h in enumerate(hs):
        W = ll_weights_batch(XB, T, h)
        for t in range(len(T)):
            I = np.flatnonzero(W[t] != 0); a = afun(np.repeat(T[t][None], len(I), 0), YB[I])
            u, R = rank_table(a, W[t][I])                                            # sim_rank.rank: tables() + rank_eval
            ref = rank_eval(u, R, np.r_[own[t], grid[t]])
            bf = SR.rank_bruteforce(a, W[t][I], np.r_[own[t], grid[t][:60]])
            worst = max(worst, abs(so_[v][t] - ref[0]), np.abs(sg_[v][t] - ref[1:]).max(), np.abs(ref[:61] - bf).max())
    print(f"rank_scores vs sim_rank construction and brute force: max diff {worst:.1e}; window sizes h=0.3: "
          f"mean {kx[0].mean():.0f}, empty windows {int((kx[0] == 0).sum())} (score 0, set = whole box)")
    assert worst < 1e-12 and so_[0][-1] == 0 and np.all(sg_[0][-1] == 0)
    # 4. tuning hours excluded from the tuning fits and included in the refits (recorders in place of the fitters)
    global fit_net_tuned, local_linear_fast, CKDE, CellFit
    saved = (fit_net_tuned, local_linear_fast, CKDE, CellFit)
    rec = defaultdict(list)
    def fake_net(Ztr, Ltr, Zv, Lv, Zall, Lall, seed):
        rec["net"].append((Ztr.copy(), Zv.copy(), Zall.copy())); return (lambda A: np.full(len(A), 0.5)), 7
    def fake_llf(C, L, E, sc, poly, ydim=2):
        rec["llf"].append((C.shape[0], E.shape[0])); return np.full(len(E), 0.5), None
    class RecKDE(saved[2]):
        def __init__(self, ctx, X, Y, hx, hy):
            rec["kde"].append(X.copy()); super().__init__(ctx, X, Y, hx, hy)
    class RecCell(saved[3]):
        def __init__(self, ctx, X, Y, m, hy):
            rec["cell"].append(X.copy()); super().__init__(ctx, X, Y, m, hy)
    fit_net_tuned, local_linear_fast, CKDE, CellFit = fake_net, fake_llf, RecKDE, RecCell
    try:
        D = split(0, {"N_A": 1000, "N_REG": 400, "NVAL": 300, "NCAL": 200, "N_TEST": 300})
        ctx = make_ctx(D["YB"]); VI = D["VI"]
        for tag, Xf, Yf in (("full", D["XF"], D["YF"]), ("A", D["XA"], D["YA"])):
            # covariates of the bike data repeat: a tiny row-specific shift makes every fitting point identifiable
            Xf = Xf + 1e-9 * (np.arange(len(Xf)) + 1)[:, None]; XV, YV = Xf[VI], Yf[VI]
            rec.clear()
            _, lp, inf = fit_density(ctx, Xf, Yf, VI, 0, tag)
            Ztr, Zv, Zall = rec["net"][0]
            key = lambda Z: set(map(tuple, np.round(Z[:, :DX], 12)))
            xs_v = set(map(tuple, np.round(Xf[VI], 12))); xs_tr = set(map(tuple, np.round(np.delete(Xf, VI, 0), 12)))
            n = len(Xf)
            assert len(Ztr) == (n - len(VI)) * (K + 1) and len(Zall) == n * (K + 1)
            assert key(Ztr) == xs_tr and key(Zv) <= xs_v and key(Zall) == xs_tr | xs_v and not (key(Ztr) & xs_v)
            assert all(c == (n - len(VI)) * (K + 1) and e == len(Zv) for c, e in rec["llf"])   # LP tuning fits
            lp.dens(np.hstack([D["Xc"][:5], D["Yc"][:5]])); assert rec["llf"][-1][0] == n * (K + 1)   # LP refit
            print(f"fit_density({tag}): NN/LP tuning rows {len(Ztr)} = ({n} - {len(VI)}) x {K + 1}, none from the tuning "
                  f"hours; held-out rows {len(Zv)} all from the tuning hours; refit rows {len(Zall)} = {n} x {K + 1}")
            for fun, nm in ((fit_kde, "kde"), (fit_cell, "cell")):
                rec[nm].clear(); fun(ctx, Xf, Yf, VI, XV, YV)
                tuned, refit = rec[nm][:-1], rec[nm][-1]
                assert all(len(t) == n - len(VI) and not (set(map(tuple, np.round(t, 12))) & xs_v) for t in tuned)
                assert len(refit) == n and np.array_equal(refit, Xf)
                print(f"   {nm}: {len(tuned)} tuning fits on {n - len(VI)} points without the tuning hours; refit on all {n}")
    finally:
        fit_net_tuned, local_linear_fast, CKDE, CellFit = saved
    print("sampling floor for 15 groups of 325:", round(sampling_floor([325] * 15), 4), "; of 200:", round(sampling_floor([200] * 15), 4))
    print("md5 sim_rank.py:", MD5_SR)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    else:
        run(int(sys.argv[1]))
