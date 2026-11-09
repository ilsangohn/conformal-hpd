"""Final-output (conformal) evaluation: coverage, conditional coverage by region, volume, shape.

Scenarios (dX = 3):
  smooth : 2-index Gaussian-mixture model (as in sim_metric)
  hetero : locally varying complexity. w(x1) = sigmoid((x1-0.5)/0.15);
           p(y|x) = (1-w) N((0, 0.3(x2-0.5)), 0.5^2 I) + w [0.5 N((-0.9,0),0.18^2 I) + 0.5 N((0.9,0),0.18^2 I)]
Methods: NN, NN_under, RF, LL (isotropic local linear), CERT(NN|LL, kappa), GLOBAL(NN vs LL by val loss).
Conformal: HPD value = estimated mass above p_hat(y|x) on a y-grid; split-conformal quantile on a
fresh calibration sample (n_cal = 500). True conditional coverage computed on the grid.
"""
import itertools, json, sys, time
import numpy as np

from sim_contrastive import BOX, contrastive_rows, eta_to_density, bce
from sim_metric import directions
from sim_sdr import local_linear_sd
from sim_robust import fit_net, fit_tree
import sim_robust

K, ALPHA, G = 5, 0.1, 30
NCAL = 500


def sig(u):
    return 1 / (1 + np.exp(-u))


import os as _os
_HW, _HC = float(_os.environ.get("HET_WIDTH", "0.15")), float(_os.environ.get("HET_CENTER", "0.5"))   # hetero transition in x1 (paper: width 0.15 at 0.5; the earlier design used 0.04 at 0.65)


def _w(x1):
    return sig((x1 - _HC) / _HW)


def sample(n, W, rng, scen):
    if scen == "smooth":
        return sim_robust.sample(n, 3, W, rng, "smooth")
    X = rng.uniform(size=(n, 3))
    w = _w(X[:, 0])
    isB = rng.uniform(size=n) < w
    comp = rng.choice([-1, 1], size=n)
    YA = np.stack([np.zeros(n), 0.3 * (X[:, 1] - 0.5)], 1) + 0.5 * rng.standard_normal((n, 2))
    YB = np.stack([0.9 * comp, np.zeros(n)], 1) + 0.18 * rng.standard_normal((n, 2))
    Y = np.where(isB[:, None], YB, YA)
    return X, np.clip(Y, -BOX + 1e-6, BOX - 1e-6)


def density(x, y, W, scen):
    if scen == "smooth":
        return sim_robust.density(x, y, W, "smooth")
    w = _w(x[:, 0])
    gA = lambda m0, m1, s: np.exp(-((y[:, 0] - m0) ** 2 + (y[:, 1] - m1) ** 2) / (2 * s**2)) / (2 * np.pi * s**2)
    pA = gA(0.0, 0.3 * (x[:, 1] - 0.5), 0.5)
    pB = 0.5 * gA(-0.9, 0.0, 0.18) + 0.5 * gA(0.9, 0.0, 0.18)
    return (1 - w) * pA + w * pB


def run(n, seed, scen):
    W = directions(3)
    rng = np.random.default_rng(seed)
    X, Y = sample(n, W, rng, scen)
    Xv, Yv = sample(max(500, n // 5), W, rng, scen)
    Xc, Yc = sample(NCAL, W, rng, scen)
    Z, L = contrastive_rows(X, Y, K, rng)
    Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False)
    Zvs, Lvs = Zv[sub], Lv[sub]

    g = np.linspace(-BOX, BOX, G + 1); g = (g[:-1] + g[1:]) / 2
    cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2); ng = len(Yg)
    er = np.random.default_rng(12345)
    Xs = np.vstack([np.c_[er.uniform(0.05, 0.55, 10), er.uniform(0.05, 0.95, (10, 2))],   # simple region
                    np.c_[er.uniform(0.72, 0.95, 10), er.uniform(0.05, 0.95, (10, 2))]])  # complex region
    Xm = er.uniform(size=(100, 3))                                                         # for marginal
    Xall = np.vstack([Xs, Xm])
    # evaluation rows: calibration (grid + own Y), evaluation x's (grid)
    Ecal = np.vstack([np.hstack([np.repeat(Xc, ng, 0), np.tile(Yg, (NCAL, 1))]), np.hstack([Xc, Yc])])
    Eev = np.hstack([np.repeat(Xall, ng, 0), np.tile(Yg, (len(Xall), 1))])
    E = np.vstack([Ecal, Eev])
    p_true = density(Eev[:, :3], Eev[:, 3:], W, scen).reshape(len(Xall), ng)
    p_true /= p_true.sum(1, keepdims=True) * cell

    def conformal(eta, name, extra):
        ph = np.maximum(eta_to_density(eta, K), 1e-12)
        pc_grid = ph[:NCAL * ng].reshape(NCAL, ng); pc_own = ph[NCAL * ng:NCAL * ng + NCAL]
        pe = ph[NCAL * ng + NCAL:].reshape(len(Xall), ng)
        pc_grid = pc_grid / pc_grid.sum(1, keepdims=True); pe = pe / pe.sum(1, keepdims=True)
        # HPD value: estimated mass strictly above the point's density
        s_cal = np.array([pc_grid[i][ph[:NCAL * ng].reshape(NCAL, ng)[i] > pc_own[i]].sum() for i in range(NCAL)])
        k = int(np.ceil((1 - ALPHA) * (NCAL + 1)))
        tau = np.sort(s_cal)[min(k, NCAL) - 1]
        res = {"cov_simple": [], "cov_complex": [], "cov_marg": [], "vol": [], "shape": [], "symdiff": []}
        for j in range(len(Xall)):
            order = np.argsort(-pe[j]); mass_above = np.empty(ng)
            cs = np.cumsum(pe[j][order]); mass_above[order] = cs - pe[j][order]
            Ch = mass_above <= tau
            pt = p_true[j]
            cov = float(pt[Ch].sum() * cell)
            ot = np.argsort(-pt); cst = np.cumsum(pt[ot]) * cell
            Cs_ = np.zeros(ng, bool); Cs_[ot[:np.searchsorted(cst, 1 - ALPHA) + 1]] = True
            Cc = np.zeros(ng, bool); Cc[ot[:max(np.searchsorted(cst, cov) + 1, 1)]] = True  # best set with same coverage
            if j < 10:
                res["cov_simple"].append(cov)
            elif j < 20:
                res["cov_complex"].append(cov)
            else:
                res["cov_marg"].append(cov)
            if j < 20:
                res["vol"].append(Ch.sum() / Cs_.sum())
                res["shape"].append((Ch.sum() - Cc.sum()) / Cs_.sum())
                res["symdiff"].append((Ch ^ Cs_).sum() / Cs_.sum())
        out = {"marg_cov": float(np.mean(res["cov_marg"])),
               "cerr_simple": float(np.mean(np.abs(np.array(res["cov_simple"]) - 0.9))),
               "cerr_complex": float(np.mean(np.abs(np.array(res["cov_complex"]) - 0.9))),
               "cov_simple": float(np.mean(res["cov_simple"])), "cov_complex": float(np.mean(res["cov_complex"])),
               "cerr_max": float(np.max(np.abs(np.array(res["cov_simple"] + res["cov_complex"]) - 0.9))),
               "vol": float(np.mean(res["vol"])), "shape": float(np.mean(res["shape"])),
               "symdiff": float(np.mean(res["symdiff"]))}
        out.update(extra)
        print(json.dumps({"n": n, "scen": scen, "seed": seed, "method": name, **out}), flush=True)

    ests, val = {}, {}
    for name, kw in (("NN", {}), ("NN_under", dict(width=8, depth=1)), ("NN_over", dict(early=False, epochs=150))):
        pr = fit_net(Z, L, Zvs, Lvs, seed, **kw); ests[name] = pr(E); val[name] = bce(pr(Zvs), Lvs)
    pr, _ = fit_tree("RF", Z, L, Zvs, Lvs, seed); ests["RF"] = pr(E); val["RF"] = bce(pr(Zvs), Lvs)
    shrink = (n / 4000) ** (-1 / 9); best = None
    for hx, hy in itertools.product([0.25, 0.4, 0.6], [0.3, 0.45, 0.65]):
        sc = np.array([hx * shrink] * 3 + [hy * shrink] * 2)
        loss = bce(local_linear_sd(Z, L, Zvs, sc, list(range(5)))[0], Lvs)
        if best is None or loss < best[0]:
            best = (loss, sc)
    est, sd = local_linear_sd(Z, L, E, best[1], list(range(5)))
    ests["LL"] = est; val["LL"] = best[0]
    est_u, sd_u = local_linear_sd(Z, L, E, 0.85 * best[1], list(range(5)))   # undersmoothed, for the test only

    for name in ("NN", "NN_under", "NN_over", "RF", "LL"):
        conformal(ests[name], name, {"val": val[name]})
    for nm in ("NN", "NN_under", "NN_over"):
        for kap in (1.0, 2.0):
            acc = np.abs(ests[nm] - est) <= kap * sd * np.sqrt(np.log(n))
            conformal(np.where(acc, ests[nm], est), f"CERT_{nm}_k{kap}", {"accept": float(acc[-len(Eev):].mean())})
            acc = np.abs(ests[nm] - est_u) <= kap * sd_u * np.sqrt(np.log(n))     # bias-aware test
            conformal(np.where(acc, ests[nm], est), f"BCERT_{nm}_k{kap}", {"accept": float(acc[-len(Eev):].mean())})
        pick = nm if val[nm] <= val["LL"] else "LL"
        conformal(ests[pick], f"GLOBAL_{nm}", {"pick": pick})


if __name__ == "__main__":
    n, seed, scen = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    run(n, seed, scen)
