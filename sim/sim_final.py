"""Final comparison with standard conformal baselines (d_X = 3, d_Y = 2).

Scenarios: smooth (two-index Gaussian mixture), hetero (locally varying complexity), minor (local
minor mode). Methods (all split-conformal with the same calibration sample, n_cal = 500):
  ours : HPD-NN   (HPD value of a contrastive MLP density; the procedure analyzed in the paper)
         HPD-LP   (HPD value of the contrastive local-linear anchor)
         HPD-CERT (bias-aware certified network, kappa = 2)
  base : ELLIPSOID (MLP mean + k-NN local residual covariance; Mahalanobis score)
         PCP       (Wang et al. 2022; union of balls around 50 samples from the MLP density)
         CHCDS     (Sampson & Chan 2024; additive shift of the estimated HPD density level, MLP density)
         HPD-KDE   (HPD-split with a Nadaraya-Watson conditional KDE; Izbicki et al. 2022)
Metrics: marginal coverage; conditional coverage error in region R (complex / minor-mode region) and
outside (O); volume ratio Leb(C_hat)/Leb(C*); relative symmetric difference; shape loss.
"""
import itertools, json, sys
import numpy as np
from scipy.spatial import cKDTree

from sim_contrastive import BOX, contrastive_rows, eta_to_density, bce, ckde
from sim_sdr import local_linear_sd as _llsd
from sim_robust import fit_net
import sim_cov, sim_local

K, ALPHA = 5, 0.1
NCAL = int(__import__("os").environ.get("NCAL", "500"))
G = int(__import__("os").environ.get("GRID", "30"))


def local_linear_sd(C, L, E, scales, poly_cols, chunk=50000, with_sd=False):
    """Chunked wrapper of sim_sdr.local_linear_sd (bounds the memory of the neighbour lists).
    Unless the standard errors are requested (with_sd=True) or SLOWLP=1, the estimate is computed by the
    batched-moment implementation llfast_batch.local_linear_fast, identical up to floating-point rounding;
    the second return value is then None."""
    if not with_sd and __import__("os").environ.get("SLOWLP") != "1":
        from llfast_batch import local_linear_fast
        return local_linear_fast(C, L, E, scales, poly_cols)
    out = [_llsd(C, L, E[i:i + chunk], scales, poly_cols) for i in range(0, len(E), chunk)]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def band_diag(est_u, sd_u, eta_true, band, n):
    """Coverage of the anchor's error by the band B = sd_u sqrt(log n) on the level band."""
    B = sd_u * np.sqrt(np.log(n)); err = np.abs(est_u - eta_true)
    return {"band_cov1": float((err <= B)[band].mean()), "band_cov2": float((err <= 2 * B)[band].mean()),
            "err_over_B": float(np.median((err / B)[band]))}


def model(scen):
    if scen in ("smooth", "hetero"):
        from sim_metric import directions
        W = directions(3)
        smp = lambda n, rng: sim_cov.sample(n, W, rng, scen)
        den = lambda x, y: sim_cov.density(x, y, W, scen)
        inR = (lambda x: x[:, 0] > 0.72) if scen == "hetero" else (lambda x: x[:, 0] > 0.5)
    elif scen == "gauss":
        import sim_gauss
        from sim_metric import directions
        W = directions(3)
        smp = lambda n, rng: sim_gauss.sample(n, W, rng)
        den = lambda x, y: sim_gauss.density(x, y, W)
        inR = lambda x: x[:, 0] > 0.5
    else:
        smp = lambda n, rng: sim_local.sample(n, rng, "minor")
        den = lambda x, y: sim_local.density(x, y, "minor")
        inR = lambda x: x[:, 1] > 0.75
    return smp, den, inR


def fit_mean_net(X, Y, seed):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    mu, sd = X.mean(0), X.std(0)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                              torch.nn.SiLU(), torch.nn.Linear(64, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(60):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 256):
            i = perm[b:b + 256]; opt.zero_grad()
            ((net(Xt[i]) - Yt[i]) ** 2).mean().backward(); opt.step()
    return lambda A: net(torch.tensor((A - mu) / sd, dtype=torch.float32)).detach().numpy()


def run(n, seed, scen):
    smp, den, inR = model(scen)
    rng = np.random.default_rng(seed)
    X, Y = smp(n, rng); Xv, Yv = smp(max(500, n // 5), rng); Xc, Yc = smp(NCAL, rng)
    Z, L = contrastive_rows(X, Y, K, rng); Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False); Zvs, Lvs = Zv[sub], Lv[sub]

    g = np.linspace(-BOX, BOX, G + 1); g = (g[:-1] + g[1:]) / 2; cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2); ng = len(Yg)
    er = np.random.default_rng(4242)
    cand = er.uniform(0.02, 0.98, size=(20000, 3)); m = inR(cand)
    Xs = np.vstack([cand[m][:10], cand[~m][:10]]); Xm = smp(100, er)[0]
    Xall = np.vstack([Xs, Xm]); nx = len(Xall)
    Ecal = np.vstack([np.hstack([np.repeat(Xc, ng, 0), np.tile(Yg, (NCAL, 1))]), np.hstack([Xc, Yc])])
    Eev = np.hstack([np.repeat(Xall, ng, 0), np.tile(Yg, (nx, 1))])
    E = np.vstack([Ecal, Eev])
    p_true = den(Eev[:, :3], Eev[:, 3:]).reshape(nx, ng); p_true /= p_true.sum(1, keepdims=True) * cell

    # ---------------- generic evaluation from scores (smaller score = more central)
    def evaluate(name, s_cal, s_grid, extra=None):
        tau = np.sort(s_cal)[min(int(np.ceil((1 - ALPHA) * (NCAL + 1))), NCAL) - 1]
        out = {k: [] for k in ("cR", "cO", "vol", "sym", "sh", "shld", "cM", "vh", "vs")}
        for j in range(nx):
            Ch = s_grid[j] <= tau; pt = p_true[j]; cov = float(pt[Ch].sum() * cell)
            ot = np.argsort(-pt); cst = np.cumsum(pt[ot]) * cell
            Cs_ = np.zeros(ng, bool); Cs_[ot[:np.searchsorted(cst, 1 - ALPHA) + 1]] = True
            Cc = np.zeros(ng, bool); Cc[ot[:max(np.searchsorted(cst, cov) + 1, 1)]] = True
            if j >= 20:
                out["cM"].append(cov); out["vh"].append(Ch.sum()); out["vs"].append(Cs_.sum()); continue
            out["cR" if j < 10 else "cO"].append(abs(cov - 0.9))
            out["vol"].append(Ch.sum() / Cs_.sum()); out["sym"].append((Ch ^ Cs_).sum() / Cs_.sum())
            kc = min(max(np.searchsorted(cst, cov) + 1, 1), ng); tc = pt[ot[kc - 1]]
            out["sh"].append(np.abs(pt - tc)[Ch ^ Cc].sum() / tc / Cs_.sum())  # integral form of the shape loss
            out["shld"].append((Ch.sum() - Cc.sum()) / Cs_.sum())
        res = {"marg": float(np.mean(out["cM"])), "cerrR": float(np.mean(out["cR"])), "cerrO": float(np.mean(out["cO"])),
               "vol": float(np.mean(out["vol"])), "volM": float(np.sum(out["vh"]) / np.sum(out["vs"])), "sym": float(np.mean(out["sym"])), "shape": float(np.mean(out["sh"])), "shape_ld": float(np.mean(out["shld"]))}
        res.update(extra or {})
        print(json.dumps({"n": n, "ncal": NCAL, "scen": scen, "seed": seed, "method": name, **res}), flush=True)

    def hpd_scores(pdens):
        """HPD value (estimated mass strictly above) from density values on E rows."""
        pc = pdens[:NCAL * ng].reshape(NCAL, ng); po = pdens[NCAL * ng:NCAL * ng + NCAL]
        pe = pdens[NCAL * ng + NCAL:].reshape(nx, ng)
        pcn = pc / pc.sum(1, keepdims=True); pen = pe / pe.sum(1, keepdims=True)
        s_cal = np.array([pcn[i][pc[i] > po[i]].sum() for i in range(NCAL)])
        s_grid = np.empty((nx, ng))
        for j in range(nx):
            o = np.argsort(-pen[j]); cs = np.cumsum(pen[j][o]); s_grid[j, o] = cs - pen[j][o]
        return s_cal, s_grid

    # ---------------- estimators
    prn = fit_net(Z, L, Zvs, Lvs, seed); eta_nn = prn(E)
    shrink = (n / 4000) ** (-1 / 9); best = None
    for hx, hy in itertools.product([0.15, 0.25, 0.4, 0.6], [0.2, 0.3, 0.45]):
        sc = np.array([hx * shrink] * 3 + [hy * shrink] * 2)
        loss = bce(local_linear_sd(Z, L, Zvs, sc, list(range(5)))[0], Lvs)
        if best is None or loss < best[0]:
            best = (loss, sc)
    eta_lp, _ = local_linear_sd(Z, L, E, best[1], list(range(5)))
    ONLY_NEW = __import__("os").environ.get("ONLY_NEW") == "1"
    if ONLY_NEW:
        eta_u = sd_u = eta_lp
    else:
        eta_u, sd_u = local_linear_sd(Z, L, E, 0.85 * best[1], list(range(5)), with_sd=True)
    acc = np.abs(eta_nn - eta_u) <= 2.0 * sd_u * np.sqrt(np.log(n))
    eta_cert = np.where(acc, eta_nn, eta_lp)
    if __import__("os").environ.get("CERTONLY") == "1":
        # same-anchor switching (test and replacement both use eta_u, as in Section 5) vs two-anchor variant
        ev = slice(len(Ecal), len(Ecal) + 20 * ng)
        pr_ = den(Eev[:20 * ng, :3], Eev[:20 * ng, 3:]); eta_true = pr_ / (pr_ + K / (2 * BOX) ** 2)
        tt = np.array([np.sort(p_true[j])[::-1][np.searchsorted(np.cumsum(np.sort(p_true[j])[::-1]) * cell, 1 - ALPHA)] for j in range(20)])
        band = (np.abs(p_true[:20] - tt[:, None]) <= 0.3 * tt[:, None]).ravel()
        diag = band_diag(eta_u[ev], sd_u[ev], eta_true, band, n)
        evaluate("HPD-NN", *hpd_scores(np.maximum(eta_to_density(eta_nn, K), 1e-12)))
        evaluate("HPD-LP", *hpd_scores(np.maximum(eta_to_density(eta_lp, K), 1e-12)))
        evaluate("HPD-CERT", *hpd_scores(np.maximum(eta_to_density(eta_cert, K), 1e-12)), {"accept": float(acc[-len(Eev):].mean()), **diag})
        evaluate("HPD-CERT-SAME", *hpd_scores(np.maximum(eta_to_density(np.where(acc, eta_nn, eta_u), K), 1e-12)), {"accept": float(acc[-len(Eev):].mean())})
        Bd = 2.0 * sd_u * np.sqrt(np.log(n))
        evaluate("HPD-PROJ2", *hpd_scores(np.maximum(eta_to_density(np.clip(eta_nn, eta_u - Bd, eta_u + Bd), K), 1e-12)))
        return

    p_nn = np.maximum(eta_to_density(eta_nn, K), 1e-12)
    Bd = sd_u * np.sqrt(np.log(n))
    eta_proj1 = np.clip(eta_nn, eta_u - Bd, eta_u + Bd)
    eta_proj2 = np.clip(eta_nn, eta_u - 2 * Bd, eta_u + 2 * Bd)
    for name, eta in (("HPD-NN", eta_nn), ("HPD-LP", eta_lp), ("HPD-CERT", eta_cert), ("HPD-PROJ1", eta_proj1), ("HPD-PROJ2", eta_proj2)):
        if ONLY_NEW: break
        evaluate(name, *hpd_scores(np.maximum(eta_to_density(eta, K), 1e-12)),
                 {"accept": float(acc[-len(Eev):].mean())} if name == "HPD-CERT" else None)

    if __import__("os").environ.get("TWOSTAGE") == "1":
        # Algorithm 2: independent D2 of size n2; HPD value s_hat(X_i, Y_i); local linear regression of
        # Z_i = 1{s_hat <= c*} on X_i (bandwidth by 5-fold CV); final split conformal on s_hat + R_hat(X)
        n2 = int(__import__("os").environ.get("N2", str(n)))
        X2, Y2 = smp(n2, np.random.default_rng(seed + 1000))
        def hpd_own(eta_fun_rows):
            out = np.empty(n2); ch = 400
            for s0 in range(0, n2, ch):
                Xb, Yb = X2[s0:s0 + ch], Y2[s0:s0 + ch]; nb = len(Xb)
                Eb = np.vstack([np.hstack([np.repeat(Xb, ng, 0), np.tile(Yg, (nb, 1))]), np.hstack([Xb, Yb])])
                pd = np.maximum(eta_to_density(eta_fun_rows(Eb), K), 1e-12)
                pg = pd[:nb * ng].reshape(nb, ng); po = pd[nb * ng:]
                pn = pg / pg.sum(1, keepdims=True)
                out[s0:s0 + nb] = [pn[i][pg[i] > po[i]].sum() for i in range(nb)]
            return out
        def ll_reg(Xtr, Ztr, Xq, hx):
            # local linear regression with a triweight product kernel (smooth), vectorized
            D = (Xtr[None, :, :] - Xq[:, None, :]) / hx
            w = np.prod(np.clip(1 - D ** 2, 0, None) ** 3, axis=2)
            F = np.concatenate([np.ones(D.shape[:2] + (1,)), D], axis=2)
            G = np.einsum("qn,qna,qnb->qab", w, F, F) + 1e-8 * np.eye(F.shape[2])
            r = np.einsum("qn,qna,n->qa", w, F, Ztr)
            return np.linalg.solve(G, r[:, :, None])[:, 0, 0]
        def cv_bw(Xtr, Ztr, grid=(0.15, 0.22, 0.3, 0.4, 0.55, 0.75, 1.0)):
            idx = np.random.default_rng(seed + 2000).permutation(len(Xtr)); folds = np.array_split(idx, 5); best = None
            for hx in grid:
                err = 0.0
                for f in folds:
                    tr = np.setdiff1d(idx, f)
                    pr = np.clip(ll_reg(Xtr[tr], Ztr[tr], Xtr[f][:300], hx), 0, 1)
                    err += np.mean((pr - Ztr[f][:300]) ** 2)
                if best is None or err < best[0]:
                    best = (err, hx)
            return best[1]
        def knn_quantile_cv(Xtr, Vtr, grid=(25, 50, 100, 200, 400)):
            # conditional (1-alpha)-quantile of the HPD value by k-NN in x; k by 5-fold pinball-loss CV
            idx = np.random.default_rng(seed + 3000).permutation(len(Xtr)); folds = np.array_split(idx, 5); bestk = None
            for k in grid:
                if k >= 0.8 * len(Xtr): continue
                loss = 0.0
                for f in folds:
                    tr = np.setdiff1d(idx, f)
                    _, ii = cKDTree(Xtr[tr]).query(Xtr[f], k=k); q = np.quantile(Vtr[tr][ii], 1 - ALPHA, axis=1)
                    u = Vtr[f] - q; loss += np.mean(np.maximum((1 - ALPHA) * u, -ALPHA * u))
                if bestk is None or loss < bestk[0]:
                    bestk = (loss, k)
            k = bestk[1]; tree = cKDTree(Xtr)
            return k, lambda Xq: np.quantile(Vtr[tree.query(Xq, k=k)[1]], 1 - ALPHA, axis=1)
        SKIP2S = __import__("os").environ.get("SKIP2S") == "1"
        for name, eta, fun in (("HPD-NN", eta_nn, prn),
                               ("HPD-LP", eta_lp, lambda A: local_linear_sd(Z, L, A, best[1], list(range(5)))[0])):
            sc, sg = hpd_scores(np.maximum(eta_to_density(eta, K), 1e-12))
            evaluate(name, sc, sg)
            if SKIP2S:
                continue
            s2 = hpd_own(fun); Z2 = (s2 <= 1 - ALPHA).astype(float)
            hx = cv_bw(X2, Z2)
            mc_, me_ = ll_reg(X2, Z2, Xc, hx), ll_reg(X2, Z2, Xall, hx)
            nclip = int(((mc_ < 0) | (mc_ > 1)).sum() + ((me_ < 0) | (me_ > 1)).sum())
            Rc = np.clip(mc_, 0, 1) - (1 - ALPHA); Re = np.clip(me_, 0, 1) - (1 - ALPHA)
            evaluate(name + "-2S", sc + Rc, sg + Re[:, None], {"hx": hx, "nclip": nclip, "maxR": float(np.abs(np.r_[Rc, Re]).max())})
            # same HPD score, rectified by an estimated conditional quantile (RCP-type) fitted on the same D_reg
            kq, qfun = knn_quantile_cv(X2, s2)
            qc, qe = np.maximum(qfun(Xc), 1e-6), np.maximum(qfun(Xall), 1e-6)
            evaluate(name + "-RCP", sc / qc, sg / qe[:, None], {"k": int(kq)})
        return

    if __import__("os").environ.get("LOCALCAL") == "1":
        # two-stage: local (k-NN in x) (1-alpha)-quantile of the HPD value on the validation fold, then
        # split-conformal on the rectified score s / tau_hat(x)   [Theorem B (c) proxy]
        nv = min(1000, len(Xv))
        Ev = np.vstack([np.hstack([np.repeat(Xv[:nv], ng, 0), np.tile(Yg, (nv, 1))]), np.hstack([Xv[:nv], Yv[:nv]])])
        eta_v_nn = prn(Ev)
        eta_v_lp, _ = local_linear_sd(Z, L, Ev, best[1], list(range(5)))
        def hpd_val(pd):
            pg = pd[:nv * ng].reshape(nv, ng); po = pd[nv * ng:]
            pn = pg / pg.sum(1, keepdims=True)
            return np.array([pn[i][pg[i] > po[i]].sum() for i in range(nv)])
        tree_v = cKDTree(Xv[:nv])
        for name, eta, eta_v in (("HPD-NN", eta_nn, eta_v_nn), ("HPD-LP", eta_lp, eta_v_lp)):
            sc, sg = hpd_scores(np.maximum(eta_to_density(eta, K), 1e-12))
            sv = hpd_val(np.maximum(eta_to_density(eta_v, K), 1e-12))
            for kk in (50, 150):
                _, ii = tree_v.query(Xc, k=kk); tc_ = np.quantile(sv[ii], 1 - ALPHA, axis=1)
                _, ii = tree_v.query(Xall, k=kk); te_ = np.quantile(sv[ii], 1 - ALPHA, axis=1)
                evaluate(f"{name}-LOC{kk}", sc / np.maximum(tc_, 1e-6), sg / np.maximum(te_, 1e-6)[:, None])
        return

    # HPD-KDE (conditional KDE, bandwidths by held-out conditional log-likelihood)
    bestk = None
    for hx, hy in itertools.product([0.1, 0.2, 0.35], [0.1, 0.18, 0.3]):
        ll = np.mean(np.log(np.maximum(ckde(X, Y, Xv[:800], Yv[:800], hx, hy), 1e-8)))
        if bestk is None or ll > bestk[0]:
            bestk = (ll, hx, hy)
    p_kde = np.maximum(ckde(X, Y, E[:, :3], E[:, 3:], bestk[1], bestk[2]), 1e-12)
    evaluate("HPD-KDE", *hpd_scores(p_kde))

    # CHCDS: score = c_hat(x) - p_hat(y|x), c_hat = HPD density level of p_hat(.|x) at 1 - alpha
    pc = p_nn[:NCAL * ng].reshape(NCAL, ng); po = p_nn[NCAL * ng:NCAL * ng + NCAL]
    pe = p_nn[NCAL * ng + NCAL:].reshape(nx, ng)
    def level(row):
        o = np.sort(row)[::-1]; cs = np.cumsum(o) / o.sum()
        return o[min(np.searchsorted(cs, 1 - ALPHA), len(o) - 1)]
    cc = np.array([level(r) for r in pc]); ce = np.array([level(r) for r in pe])
    # normalize densities per x so that levels are comparable across x
    zc = pc.sum(1) * cell; ze = pe.sum(1) * cell
    evaluate("CHCDS", (cc - po) / zc, (ce[:, None] - pe) / ze[:, None])

    # PCP: 50 samples from the MLP density (grid sampling + uniform jitter), score = distance to nearest
    def samples(prow, rr, M=50):
        idx = rr.choice(ng, size=M, p=prow / prow.sum())
        return Yg[idx] + rr.uniform(-0.5, 0.5, size=(M, 2)) * (g[1] - g[0])
    rr = np.random.default_rng(seed + 7)
    s_pcp_cal = np.array([np.min(np.linalg.norm(samples(pc[i], rr) - Yc[i], axis=1)) for i in range(NCAL)])
    s_pcp_grid = np.array([cKDTree(samples(pe[j], rr)).query(Yg)[0] for j in range(nx)])
    evaluate("PCP", s_pcp_cal, s_pcp_grid)

    # ELLIPSOID: MLP mean + local (k-NN in x) residual covariance, Mahalanobis score
    mnet = fit_mean_net(X, Y, seed)
    res = Y - mnet(X); tree = cKDTree(X)
    def cov_at(xq, k=150):
        _, ii = tree.query(xq, k=k)
        return np.array([np.cov(res[i].T) + 1e-4 * np.eye(2) for i in np.atleast_2d(ii)])
    mc, Sc = mnet(Xc), cov_at(Xc)
    s_ell_cal = np.array([np.sqrt((Yc[i] - mc[i]) @ np.linalg.solve(Sc[i], Yc[i] - mc[i])) for i in range(NCAL)])
    me, Se = mnet(Xall), cov_at(Xall)
    s_ell_grid = np.array([np.sqrt(np.einsum("ij,jk,ik->i", Yg - me[j], np.linalg.inv(Se[j]), Yg - me[j])) for j in range(nx)])
    evaluate("ELLIPSOID", s_ell_cal, s_ell_grid)
    # ---------------- RCP (rectified scores; conditional (1-alpha)-quantile of the base score by
    # k-NN local quantile on the validation fold, multiplicative rectification f_tau(v) = v / tau)
    def local_q(Xq, Xref, Vref, k=100):
        _, ii = cKDTree(Xref).query(Xq, k=k)
        return np.quantile(Vref[ii], 1 - ALPHA, axis=1)
    # base scores on the validation fold (used only to fit tau(x))
    mv = mnet(Xv); Sv = cov_at(Xv)
    Vv_ell = np.array([np.sqrt((Yv[i] - mv[i]) @ np.linalg.solve(Sv[i], Yv[i] - mv[i])) for i in range(len(Xv))])
    tc, te = local_q(Xc, Xv, Vv_ell), local_q(Xall, Xv, Vv_ell)
    evaluate("RCP-ELLIPSOID", s_ell_cal / tc, s_ell_grid / te[:, None])
    pv = np.maximum(eta_to_density(prn(np.vstack([np.hstack([np.repeat(Xv[i:i+1], ng, 0), Yg]) for i in range(len(Xv))])), K), 1e-12).reshape(len(Xv), ng)
    Vv_pcp = np.array([np.min(np.linalg.norm(samples(pv[i], rr) - Yv[i], axis=1)) for i in range(len(Xv))])
    tc, te = local_q(Xc, Xv, Vv_pcp), local_q(Xall, Xv, Vv_pcp)
    evaluate("RCP-PCP", s_pcp_cal / tc, s_pcp_grid / te[:, None])

    # ---------------- OT-CP (Thurin et al. 2025 / Klein et al. 2025): OT ranks of residuals
    from scipy.optimize import linear_sum_assignment
    mres = min(1000, len(Xv)); rv = (Yv - mv)[:mres]
    ro = np.random.default_rng(seed + 11)
    ang = ro.uniform(0, 2 * np.pi, mres); rad = np.sqrt(ro.uniform(size=mres))
    U = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)               # uniform on the unit disk
    cost = ((rv[:, None, :] - U[None]) ** 2).sum(-1)
    ri, ci = linear_sum_assignment(cost); T = np.empty_like(U); T[ri] = U[ci]
    rtree = cKDTree(rv)
    def ot_rank(r, k=5):
        _, ii = rtree.query(r, k=k)
        return np.linalg.norm(T[ii].mean(1), axis=-1)
    evaluate("OT-CP", ot_rank(Yc - mc), np.array([ot_rank(Yg - me[j]) for j in range(nx)]))

    # ---------------- oracles (reference): conditional HPD (vol ratio 1 by definition) and marginal oracle
    evaluate("ORACLE-COND", *hpd_scores(np.r_[den(Ecal[:, :3], Ecal[:, 3:]), den(Eev[:, :3], Eev[:, 3:])]))
    pm_cal = den(Ecal[:, :3], Ecal[:, 3:]); pm_ev = den(Eev[:, :3], Eev[:, 3:]).reshape(nx, ng)
    s_cal_m = -pm_cal[NCAL * ng:NCAL * ng + NCAL]                           # score = -p(y|x), global threshold
    evaluate("ORACLE-MARG", s_cal_m, -pm_ev)



if __name__ == "__main__":
    n, seed, scen = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    run(n, seed, scen)
