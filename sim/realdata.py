"""Real-data application: UCI Bike Sharing (hourly).

Response  Y = (log(1 + casual + U1), log(1 + registered + U2)), U ~ U(0,1) (dequantization), affinely
mapped into the box [-BOX, BOX]^2.  Covariates X = (hour/23, temp, hum) in [0,1]^3.  The working-day
indicator is deliberately omitted, so that the conditional law of Y given X mixes commuting and leisure
patterns and is often bimodal.

One replication = one random split: train 6500 (density fit), validation 1500, D2 3000 (second stage /
local quantiles / transport map), calibration 1500, test = rest (about 4879).
Methods: HPD-NN, HPD-LP, HPD-CERT (switching, kappa=2), HPD-NN-2S, HPD-LP-2S (Algorithm 2), CHCDS,
ELLIPSOID, RCP-ELLIPSOID, PCP, OT-CP.
Metrics on the test set: marginal coverage; conditional coverage in 15 covariate groups
(5 hour blocks x 3 temperature terciles): mean and max |cov_g - 0.9|; mean set area in the original
log-count units.
"""
import itertools, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from sim_contrastive import BOX, contrastive_rows, eta_to_density, bce
from sim_sdr import local_linear_sd as _llsd


def local_linear_sd(C, L, E, scales, cols, chunk=50000):
    if __import__("os").environ.get("SLOWLP") != "1" and not getattr(local_linear_sd, "need_sd", False):
        from llfast_batch import local_linear_fast   # identical up to rounding; no standard errors
        return local_linear_fast(C, L, E, scales, cols)
    est = np.empty(len(E)); sd = np.empty(len(E))
    for s0 in range(0, len(E), chunk):
        est[s0:s0 + chunk], sd[s0:s0 + chunk] = _llsd(C, L, E[s0:s0 + chunk], scales, cols)
    return est, sd
from sim_robust import fit_net
from sim_final import fit_mean_net

K, ALPHA, G = 5, 0.1, 30
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "hour.csv")


def load(seed):
    raw = np.genfromtxt(DATA, delimiter=",", names=True)
    rng = np.random.default_rng(seed)
    y = np.c_[np.log1p(raw["casual"] + rng.uniform(size=len(raw))), np.log1p(raw["registered"] + rng.uniform(size=len(raw)))]
    x = np.c_[raw["hr"] / 23.0, raw["temp"], raw["hum"]]
    grp_hour = np.digitize(raw["hr"], [6, 10, 16, 20])                    # 0-5, 6-9, 10-15, 16-19, 20-23
    t1, t2 = np.quantile(raw["temp"], [1 / 3, 2 / 3])
    grp = grp_hour * 3 + np.digitize(raw["temp"], [t1, t2])
    # affine map of each response coordinate into the box (fixed across splits)
    lo, hi = np.quantile(y, 0.001, axis=0), np.quantile(y, 0.999, axis=0)
    scale = (hi - lo) / (2 * (BOX - 0.25)); centre = (hi + lo) / 2
    yb = np.clip((y - centre) / scale, -BOX + 1e-6, BOX - 1e-6)
    return x, yb, grp, scale


def run(seed):
    x, y, grp, scale = load(seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(x))
    sizes = [6500, 1500, 3000, 1500]
    cut = np.cumsum(sizes)
    itr, iva, i2, ica, ite = np.split(idx, cut)
    X, Y = x[itr], y[itr]; Xv, Yv = x[iva], y[iva]; X2, Y2 = x[i2], y[i2]; Xc, Yc = x[ica], y[ica]; Xt, Yt = x[ite], y[ite]
    ite = ite[:3000]; Xt, Yt = x[ite], y[ite]
    gt = grp[ite]
    Z, L = contrastive_rows(X, Y, K, rng); Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False); Zvs, Lvs = Zv[sub], Lv[sub]

    g = np.linspace(-BOX, BOX, G + 1); g = (g[:-1] + g[1:]) / 2; cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2); ng = len(Yg)
    area_unit = cell * scale[0] * scale[1]                                 # one grid cell in log-count units

    def rows(Xq, Yq):
        """grid rows for each query x, followed by the query's own (x, y) rows"""
        return np.vstack([np.hstack([np.repeat(Xq, ng, 0), np.tile(Yg, (len(Xq), 1))]), np.hstack([Xq, Yq])])

    out = []

    def report(name, s_cal, s_test, s_grid_test, extra=None):
        k = int(np.ceil((1 - ALPHA) * (len(s_cal) + 1)))
        tau = np.sort(s_cal)[min(k, len(s_cal)) - 1]
        cov = s_test <= tau
        cg = np.array([cov[gt == j].mean() for j in np.unique(gt)])
        area = (s_grid_test <= tau).sum(1) * area_unit
        res = {"seed": seed, "method": name, "marg": float(cov.mean()), "cerr_mean": float(np.mean(np.abs(cg - 0.9))),
               "cerr_max": float(np.max(np.abs(cg - 0.9))), "area": float(area.mean())}
        res.update(extra or {})
        print(json.dumps(res), flush=True)

    def hpd(pdens, nq):
        pg = pdens[:nq * ng].reshape(nq, ng); po = pdens[nq * ng:]
        pn = pg / pg.sum(1, keepdims=True)
        s_own = np.array([pn[i][pg[i] > po[i]].sum() for i in range(nq)])
        s_grid = np.empty((nq, ng))
        for i in range(nq):
            o = np.argsort(-pn[i]); cs = np.cumsum(pn[i][o]); s_grid[i, o] = cs - pn[i][o]
        return s_own, s_grid, pg, po

    # ---------------- density estimators
    prn = fit_net(Z, L, Zvs, Lvs, seed)
    shrink = (len(X) / 4000) ** (-1 / 9); best = None
    for hx, hy in itertools.product([0.15, 0.25, 0.4, 0.6], [0.2, 0.3, 0.45]):
        sc = np.array([hx * shrink] * 3 + [hy * shrink] * 2)
        loss = bce(local_linear_sd(Z, L, Zvs, sc, list(range(5)))[0], Lvs)
        if best is None or loss < best[0]:
            best = (loss, sc)
    lp = lambda E: local_linear_sd(Z, L, E, best[1], list(range(5)))[0]
    def cert(E):
        e_nn = prn(E); e_lp = lp(E)
        local_linear_sd.need_sd = True
        e_u, sd_u = local_linear_sd(Z, L, E, 0.85 * best[1], list(range(5)))
        local_linear_sd.need_sd = False
        return np.where(np.abs(e_nn - e_u) <= 2 * sd_u * np.sqrt(np.log(len(X))), e_nn, e_lp)
    Ec, Et, E2 = rows(Xc, Yc), rows(Xt, Yt), rows(X2, Y2)

    def ll_reg(Xtr, Ztr, Xq, hx):
        out_ = np.empty(len(Xq))
        for s0 in range(0, len(Xq), 500):
            D = (Xtr[None, :, :] - Xq[s0:s0 + 500, None, :]) / hx
            w = np.prod(np.clip(1 - D ** 2, 0, None) ** 3, axis=2)
            F = np.concatenate([np.ones(D.shape[:2] + (1,)), D], axis=2)
            Gm = np.einsum("qn,qna,qnb->qab", w, F, F) + 1e-8 * np.eye(F.shape[2])
            r = np.einsum("qn,qna,n->qa", w, F, Ztr)
            out_[s0:s0 + 500] = np.linalg.solve(Gm, r[:, :, None])[:, 0, 0]
        return out_
    def cv_bw(Xtr, Ztr, grid=(0.15, 0.22, 0.3, 0.4, 0.55, 0.75, 1.0)):
        perm = np.random.default_rng(seed + 2000).permutation(len(Xtr)); folds = np.array_split(perm, 5); bb = None
        for hx in grid:
            err = 0.0
            for f in folds:
                tr = np.setdiff1d(perm, f)
                err += np.mean((np.clip(ll_reg(Xtr[tr], Ztr[tr], Xtr[f][:300], hx), 0, 1) - Ztr[f][:300]) ** 2)
            if bb is None or err < bb[0]:
                bb = (err, hx)
        return bb[1]

    for name, fun in (("HPD-NN", prn), ("HPD-LP", lp), ("HPD-CERT", cert)):
        sc_, _, _, _ = hpd(np.maximum(eta_to_density(fun(Ec), K), 1e-12), len(Xc))
        st_, sg_, pg_t, po_t = hpd(np.maximum(eta_to_density(fun(Et), K), 1e-12), len(Xt))
        report(name, sc_, st_, sg_)
        if name == "HPD-NN":
            # CHCDS with the network density: score = level(x) - p_hat(y|x), normalized per x
            pgc = np.maximum(eta_to_density(prn(Ec), K), 1e-12)
            pc, poc = pgc[:len(Xc) * ng].reshape(len(Xc), ng), pgc[len(Xc) * ng:]
            def level(row):
                o = np.sort(row)[::-1]; cs = np.cumsum(o) / o.sum()
                return o[min(np.searchsorted(cs, 1 - ALPHA), len(o) - 1)]
            lc = np.array([level(r) for r in pc]); lt = np.array([level(r) for r in pg_t])
            zc, zt = pc.sum(1) * cell, pg_t.sum(1) * cell
            report("CHCDS", (lc - poc) / zc, (lt - po_t) / zt, (lt[:, None] - pg_t) / zt[:, None])
        if name in ("HPD-NN", "HPD-LP"):
            s2, _, _, _ = hpd(np.maximum(eta_to_density(fun(E2), K), 1e-12), len(X2))
            Z2 = (s2 <= 1 - ALPHA).astype(float); hx = cv_bw(X2, Z2)
            Rc = np.clip(ll_reg(X2, Z2, Xc, hx), 0, 1) - (1 - ALPHA); Rt = np.clip(ll_reg(X2, Z2, Xt, hx), 0, 1) - (1 - ALPHA)
            report(name + "-2S", sc_ + Rc, st_ + Rt, sg_ + Rt[:, None], {"hx": hx})
            if name == "HPD-NN":
                pt_nn = pg_t   # keep network grid densities for PCP

    # ---------------- standard baselines
    mnet = fit_mean_net(np.vstack([X, Xv]), np.vstack([Y, Yv]), seed)
    res = np.vstack([Y, Yv]) - mnet(np.vstack([X, Xv])); tree = cKDTree(np.vstack([X, Xv]))
    def cov_at(xq, k=150):
        _, ii = tree.query(xq, k=k)
        return np.array([np.cov(res[i].T) + 1e-4 * np.eye(2) for i in np.atleast_2d(ii)])
    def maha(Xq, Yq):
        m, S = mnet(Xq), cov_at(Xq); Si = np.linalg.inv(S)
        own = np.sqrt(np.einsum("ij,ijk,ik->i", Yq - m, Si, Yq - m))
        grid = np.array([np.sqrt(np.einsum("gj,jk,gk->g", Yg - m[i], Si[i], Yg - m[i])) for i in range(len(Xq))])
        return own, grid
    ec, _ = maha(Xc, Yc); et, egt = maha(Xt, Yt); e2, _ = maha(X2, Y2)
    report("ELLIPSOID", ec, et, egt)
    t2 = cKDTree(X2)
    lq = lambda Xq: np.quantile(e2[t2.query(Xq, k=100)[1]], 1 - ALPHA, axis=1)
    qc, qt = lq(Xc), lq(Xt)
    report("RCP-ELLIPSOID", ec / qc, et / qt, egt / qt[:, None])
    # PCP: 50 samples from the network density on the grid (+ jitter)
    rr = np.random.default_rng(seed + 7)
    pgc_nn = np.maximum(eta_to_density(prn(Ec), K), 1e-12)[:len(Xc) * ng].reshape(len(Xc), ng)
    def samples(prow):
        ii = rr.choice(ng, size=50, p=prow / prow.sum())
        return Yg[ii] + rr.uniform(-0.5, 0.5, size=(50, 2)) * (g[1] - g[0])
    sc_p = np.array([np.min(np.linalg.norm(samples(pgc_nn[i]) - Yc[i], axis=1)) for i in range(len(Xc))])
    st_p = np.empty(len(Xt)); sg_p = np.empty((len(Xt), ng))
    for i in range(len(Xt)):
        S_ = samples(pt_nn[i]); tr_ = cKDTree(S_)
        st_p[i] = tr_.query(Yt[i])[0]; sg_p[i] = tr_.query(Yg)[0]
    report("PCP", sc_p, st_p, sg_p)
    # OT-CP: transport of D2 residuals to the uniform disk, k-NN extension
    r2 = (Y2 - mnet(X2))[:1000]; ro = np.random.default_rng(seed + 11)
    ang = ro.uniform(0, 2 * np.pi, len(r2)); rad = np.sqrt(ro.uniform(size=len(r2)))
    U = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    ri, ci = linear_sum_assignment(((r2[:, None, :] - U[None]) ** 2).sum(-1)); T = np.empty_like(U); T[ri] = U[ci]
    rt = cKDTree(r2)
    otr = lambda r: np.linalg.norm(T[rt.query(r, k=5)[1]].mean(-2), axis=-1)
    mt = mnet(Xt)
    report("OT-CP", otr(Yc - mnet(Xc)), otr(Yt - mt), np.array([otr(Yg - mt[i]) for i in range(len(Xt))]))


if __name__ == "__main__":
    run(int(sys.argv[1]))
