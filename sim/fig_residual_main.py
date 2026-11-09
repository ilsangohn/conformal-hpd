"""Figure fig:residual (Section 6) for the design of the main experiment: coverage residuals and conditional coverage
along x_1 for the LP density estimate, one-stage calibration and Algorithm 1 (alg:local).

Model SCEN (default moe, the multimodal model of Section 8, with MOE_K=4, MOE_S=0.02,0.5, MOE_G=3), d_X = 3, d_Y = 2; 80 covariates on the line
x_1 = 0.03, ..., 0.97 with x_2 = x_3 = 0.5. Data, sizes and tuning as sim_rank.py with run_main_n5k.sh (same draws for the
same seed): one-stage fits on N = 4000; two-stage fits on the first N - N_REG = 3000 and uses N_REG = 1000 points for
the regression; common NCAL = 1000 calibration points. The last NVAL = 750 of the first 3000 points tune each LP fit
(held-out contrastive log-loss over sim_rank.LP_HX x LP_HY), which is then refitted on its whole fitting sample.
Contrastive reference REF=gmix (sim_rank.Ref of the first 3000 responses).
  one-stage: HPD values of the one-stage LP on the 60x60 response grid, split conformal (tau_1).
  two-stage: rank calibration with the two-stage LP as ranking function (sim_rank.rank_table / ll_weights_batch;
             isotropic regression bandwidth by the frozen-target CV sim_rank.frozen_cv_vec with ISO_ONLY=1, as in the
             main experiment), split conformal on the calibration sample (tau_2). Reference: the fixed bandwidth
             sim_rank.h_fixed(N_REG) (*fix outputs).
Per covariate of the line: true residuals R_j(gamma, x) = P_x(HPD set of fit j at estimated mass gamma) - gamma,
the coverages of both output sets, and u_x = estimated mass (fit 2, normalized on the 60x60 grid) of the two-stage
output set {y: Qbar_x(a(y|x)) <= tau_2}. True probabilities by Monte Carlo: L = 20000 draws from the truncated
conditional distribution at each covariate of the line (the multimodal model has components of scale 0.02, which a
quadrature grid does not resolve). The hard region of the model along the line is saved as "hard".
Run: python sim/fig_residual_main.py SEED [SEED ...]   Outputs: $FIG1_OUT (default results/fig1_main/)
rank_seed{SEED}.npz/.json. Plot: sim/fig_residual_main_plot.py. (fig_residual_rank.py: the earlier design.)
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("ISO_ONLY", "1"); os.environ.setdefault("REF", "gmix")
for k_, v_ in (("MOE_K", "4"), ("MOE_S", "0.02,0.5"), ("MOE_G", "3")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from sim_contrastive import BOX, bce
from llfast_batch import _Setup, local_linear_fast
import sim_rank as SR
from sim_rank import (Model, Ref, crows, eta2dens, rank_table, rank_eval, ll_weights_batch, frozen_cv_vec, h_fixed,
                      LP_HX, LP_HY, H_AXES)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("FIG1_OUT", str(ROOT / "results/fig1_main")))
SCEN = os.environ.get("SCEN", "moe")
GAMMA, K, GRID, L_MC = .9, 5, 60, 20000
N = int(os.environ.get("N", "4000"))
N_REG = int(os.environ.get("N_REG", "1000"))
NCAL = int(os.environ.get("NCAL", "1000"))
NVAL = int(os.environ.get("NVAL", "750"))
X1 = np.linspace(0.03, 0.97, 80)


def hpd_at(pg, values):
    """Estimated mass strictly above each value, from the fitted density on the score grid."""
    order = np.argsort(pg)
    mass = np.r_[0., np.cumsum((pg[order] / pg.sum())[::-1])]
    return mass[len(pg) - np.searchsorted(pg[order], values, side="right")]


def grid(n):
    e = np.linspace(-BOX, BOX, n + 1)
    g = (e[:-1] + e[1:]) / 2
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2), (g[1] - g[0]) ** 2


def fit_lp(Xf, Yf, rng_seed, VI):
    """LP fit tuned on the points VI of its fitting sample (fitted on the rest), refitted on all of it (sim_rank)"""
    n = len(Xf); tr = np.setdiff1d(np.arange(n), VI)
    Z, L = crows(Xf, Yf, np.random.default_rng(rng_seed))
    rows = lambda I: np.concatenate([I, n + (I[:, None] * K + np.arange(K)).ravel()])
    rtr = rows(tr); rv = np.random.default_rng(rng_seed + 5).choice(rows(VI), size=min(3000, len(VI) * (K + 1)), replace=False)
    best = None; cv = {}
    for hx, hy in itertools.product(LP_HX, LP_HY):
        sc = np.array([hx] * 3 + [hy] * 2)
        loss = bce(local_linear_fast(Z[rtr], L[rtr], Z[rv], sc, list(range(5)), ydim=2)[0], L[rv]); cv[f"{hx},{hy}"] = round(float(loss), 6)
        if best is None or loss < best[0]:
            best = loss, sc
    return _Setup(Z, L, best[1], list(range(5)), 2, 480, 64, 64, 4_000_000), best[1], cv


def dens(fit, x, Y):
    rows = np.c_[np.repeat(x[None], len(Y), axis=0), Y]
    return np.maximum(eta2dens(fit.chunk(rows)[0], 2, rows[:, 3:]), 1e-12)


def dens_rows(fit, rows, ch=100000):
    return np.concatenate([np.maximum(eta2dens(fit.chunk(rows[i:i + ch])[0], 2, rows[i:i + ch, 3:]), 1e-12)
                           for i in range(0, len(rows), ch)])


def run(seed):
    start = time.time()

    def log(*args):
        print(f"seed={seed} {time.time() - start:.1f}s", *args, flush=True)

    OUT.mkdir(exist_ok=True)
    final = OUT / f"rank_seed{seed}.npz"
    if final.exists():
        log("completed output exists; skipping")
        return
    mo = Model(SCEN, 2)
    rng = np.random.default_rng(seed)
    X, Y = mo.sample(N, rng)
    Xc, Yc = mo.sample(NCAL, rng)
    nA = N - N_REG
    XA, YA, XB, YB = X[:nA], Y[:nA], X[nA:], Y[nA:]
    VI = np.arange(nA - NVAL, nA)                       # tuning points: the last NVAL of the first N - N_REG
    SR.REF_OBJ = Ref(YA) if SR.REF == "gmix" else None  # contrastive reference (as sim_rank.run)
    fit1, bw1, lpcv1 = fit_lp(X, Y, seed, VI)
    fit2, bw2, lpcv2 = fit_lp(XA, YA, seed + 77, VI)
    log("LP bandwidths one", bw1.tolist(), "two", bw2.tolist())

    Yg, cell = grid(GRID)
    Xe = np.c_[X1, np.full(len(X1), .5), np.full(len(X1), .5)]
    ev = np.random.default_rng(4343 + seed)
    Yd = [mo.draw_truncated(x, L_MC, ev) for x in Xe]  # draws from the truncated conditional distribution at each x
    k = int(np.ceil(GAMMA * (NCAL + 1)))

    # ---------------- one-stage: calibration on the 60x60 score grid
    t0 = time.time()
    s_cal1 = np.empty(NCAL)
    for t in range(NCAL):
        pd = dens(fit1, Xc[t], np.vstack([Yg, Yc[t:t + 1]]))
        s_cal1[t] = hpd_at(pd[:-1], pd[-1:])[0]
    tau1 = np.sort(s_cal1)[k - 1]
    time_one_cal = time.time() - t0
    log("one-stage calibration done, tau1", tau1)
    sf1 = np.empty((len(Xe), L_MC))
    for j, x in enumerate(Xe):
        pd = dens(fit1, x, np.vstack([Yg, Yd[j]]))
        sf1[j] = hpd_at(pd[:len(Yg)], pd[len(Yg):])
    log("one-stage evaluation scores done")

    # ---------------- two-stage: rank calibration (CV bandwidth; fixed bandwidth as reference)
    t0 = time.time()
    a_own = dens_rows(fit2, np.c_[XB, YB])
    aux = lambda T, idxs: [dens(fit2, T[t], YB[i]) if len(i) else np.empty(0) for t, i in enumerate(idxs)]
    hx, cv_loss, cv_nskip = frozen_cv_vec(XB, a_own, aux, seed)
    time_cv = time.time() - t0
    hf = h_fixed(N_REG)
    hs = (hx, hf)
    log("regression bandwidth", hx, "CV loss", np.round(cv_loss, 6).tolist(), "fixed", hf)

    def tables_and_values(T, extra):
        """rank tables (CV and fixed bandwidth) at the rows of T and fit-2 values at the extra points of each row"""
        tabs, vals, kx = ([], []), [], ([], [])
        for s0 in range(0, len(T), 50):
            Ws = [ll_weights_batch(XB, T[s0:s0 + 50], h) for h in hs]
            for i in range(len(Ws[0])):
                t = s0 + i
                idx = np.flatnonzero((Ws[0][i] != 0) | (Ws[1][i] != 0))
                pd = dens(fit2, T[t], np.vstack([YB[idx], extra(t)]))
                vals.append(pd[len(idx):])
                for v in (0, 1):
                    w = Ws[v][i][idx]; tabs[v].append(rank_table(pd[:len(idx)], w)); kx[v].append(int((w != 0).sum()))
        return tabs, vals, [np.array(k_) for k_ in kx]

    t0 = time.time()
    tabs_c, own_c, kx_c = tables_and_values(Xc, lambda t: Yc[t:t + 1])
    s_cal2 = [np.array([rank_eval(u, R, v)[0] for (u, R), v in zip(tabs_c[q], own_c)]) for q in (0, 1)]
    tau2 = [np.sort(s_cal2[q])[k - 1] for q in (0, 1)]
    time_two_cal = time.time() - t0 + time_cv
    log("two-stage calibration done, tau2", tau2, "mean k_x", kx_c[0].mean(), kx_c[1].mean())
    tabs_e, vals_e, kx_e = tables_and_values(Xe, lambda t: np.vstack([Yg, Yd[t]]))
    in2f = np.empty((2, len(Xe), L_MC), bool)
    in2g = np.empty((2, len(Xe), len(Yg)), bool)
    sf2 = np.empty((len(Xe), L_MC))
    u_x = np.empty((2, len(Xe)))
    for j in range(len(Xe)):
        pg, pfit = vals_e[j][:len(Yg)], vals_e[j][len(Yg):]
        for q in (0, 1):
            u, R = tabs_e[q][j]
            in2g[q, j] = rank_eval(u, R, pg) <= tau2[q]
            in2f[q, j] = rank_eval(u, R, pfit) <= tau2[q]
            u_x[q, j] = (pg / pg.sum())[in2g[q, j]].sum()     # estimated mass of the output set (fit 2, 60x60 grid)
        sf2[j] = hpd_at(pg, pfit)
    log("two-stage evaluation done")

    def cov(s, m):                                       # Monte Carlo coverage of {y: s(y) <= m(x)} at each x
        return np.mean(s <= np.asarray(m)[:, None], axis=1)

    res = dict(
        cov1=cov(sf1, np.full(len(Xe), tau1)), cov2=in2f[0].mean(1), cov2fix=in2f[1].mean(1),
        R1=cov(sf1, np.full(len(Xe), GAMMA)) - GAMMA, R2=cov(sf2, np.full(len(Xe), GAMMA)) - GAMMA,
        cov2_via_u=cov(sf2, u_x[0]))                        # F_2(u_x, x): check of the identity C_two = HPD set at u_x
    hard = mo.inR(Xe)
    e1, e2, e2f = abs(res["cov1"] - GAMMA), abs(res["cov2"] - GAMMA), abs(res["cov2fix"] - GAMMA)
    summary = dict(
        seed=seed, N=N, N_REG=N_REG, NCAL=NCAL, NVAL=NVAL, score_grid=GRID, eval_mc=L_MC,
        lp_bandwidth_one=bw1.tolist(), lp_bandwidth_two=bw2.tolist(), lp_cv_one=lpcv1, lp_cv_two=lpcv2,
        scen=SCEN, ref=SR.REF, regression_bandwidth=[float(v) for v in hx], cv_axes=list(H_AXES),
        cv_loss=[round(float(c), 7) for c in cv_loss], cv_nskip=int(cv_nskip),
        regression_bandwidth_fixed=hf,
        tau1=float(tau1), tau2=float(tau2[0]), tau2_fixed=float(tau2[1]),
        kx_mean_cal=float(kx_c[0].mean()), kx_mean_line=float(kx_e[0].mean()), kx_min=int(min(kx_c[0].min(), kx_e[0].min())),
        kx_mean_fixed=float(np.r_[kx_c[1], kx_e[1]].mean()),
        cerr_one=float(e1.mean()), cerr_two=float(e2.mean()), cerr_two_fixed=float(e2f.mean()),
        cerr_one_hard=float(e1[hard].mean()), cerr_two_hard=float(e2[hard].mean()), cerr_two_fixed_hard=float(e2f[hard].mean()),
        cov_one_min=float(res["cov1"].min()), cov_two_min=float(res["cov2"].min()), cov_two_fixed_min=float(res["cov2fix"].min()),
        x1_at_cov_two_min=float(X1[np.argmin(res["cov2"])]),
        mean_abs_gamma_minus_u=float(np.abs(GAMMA - u_x[0]).mean()),
        max_abs_cov2_minus_F2u=float(np.abs(res["cov2"] - res["cov2_via_u"]).max()),
        seconds_one_stage_calibration=time_one_cal, seconds_two_stage_calibration=time_two_cal,
        seconds=time.time() - start)
    np.savez(final, x1=X1, tau1=tau1, tau2=tau2[0], tau2fix=tau2[1], u_x=u_x[0], u_xfix=u_x[1], s_cal2=s_cal2[0],
             s_cal2fix=s_cal2[1], n1=N, n2=nA, n_reg=N_REG, ncal=NCAL, hx_reg=hx, hx_fix=hf, kx_line=kx_e[0], seed=seed,
             cv_loss=cv_loss, hard=hard, **res)
    (OUT / f"rank_seed{seed}.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("DONE", json.dumps(summary))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
