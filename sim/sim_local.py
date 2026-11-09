"""Search for REAL local failures of a neural conditional density estimator.

Scenarios (dX = 3, Y in R^2, base: symmetric bimodal Gaussian mixture, modes (+-a, 0), sd s):
  jump      : a = 0.5 if x1 < 0.5 else 1.2 (discontinuous in x); region R = {|x1 - 0.5| < 0.1}
  oscillate : a = 0.9 + 0.35 sin(2 pi 5 x1) for x1 > 0.6, a = 0.9 + 0.35 sin(2 pi 5 * 0.6) otherwise;
              region R = {x1 > 0.6}
  minor     : extra mode at (0, 1.4), sd 0.15, weight 0.2 * 1{x2 > 0.7} (smoothly switched);
              region R = {x2 > 0.75}
  sparse    : design density 10x lower for x1 > 0.85; there the modes rotate by 90 degrees
              (smooth switch); region R = {x1 > 0.88}
Methods: NN (early-stopped MLP), LL (validated bandwidth), BCERT (bias-aware certification with an
undersmoothed anchor), GLOBAL (validation-loss selection). Region-wise final conformal metrics and a
pointwise diagnostic near the level set: mean |e_NN| and |e_LL| on the level band, in R and outside.
"""
import itertools, json, sys
import numpy as np

from sim_contrastive import BOX, contrastive_rows, eta_to_density, bce
from sim_sdr import local_linear_sd
from sim_robust import fit_net

K, ALPHA, G, NCAL = 5, 0.1, 30, 500


def sig(u):
    return 1 / (1 + np.exp(-u))


def params(x, scen):
    x1, x2 = x[:, 0], x[:, 1]
    s = 0.25 * np.ones(len(x))
    rot = np.zeros(len(x)); wminor = np.zeros(len(x))
    if scen == "jump":
        a = np.where(x1 < 0.5, 0.5, 1.2)
    elif scen == "oscillate":
        a = 0.9 + 0.35 * np.sin(2 * np.pi * 5 * np.maximum(x1, 0.6))
    elif scen == "minor":
        a = 0.9 + 0.2 * x1
        wminor = 0.2 * sig((x2 - 0.7) / 0.02)
    else:  # sparse
        a = 0.9 + 0.2 * x2
        rot = sig((x1 - 0.85) / 0.02)
    return a, s, rot, wminor


def in_region(x, scen):
    return {"jump": np.abs(x[:, 0] - 0.5) < 0.1, "oscillate": x[:, 0] > 0.6,
            "minor": x[:, 1] > 0.75, "sparse": x[:, 0] > 0.88}[scen]


def sample_x(n, rng, scen):
    if scen != "sparse":
        return rng.uniform(size=(n, 3))
    X = rng.uniform(size=(3 * n, 3))
    keep = rng.uniform(size=3 * n) < np.where(X[:, 0] > 0.85, 0.1, 1.0)
    return X[keep][:n]


def sample(n, rng, scen):
    X = sample_x(n, rng, scen)
    a, s, rot, wm = params(X, scen)
    comp = rng.choice([-1, 1], size=n)
    ang = rot * np.pi / 2
    ctr = np.stack([comp * a * np.cos(ang), comp * a * np.sin(ang)], 1)
    Y = ctr + s[:, None] * rng.standard_normal((n, 2))
    minor = rng.uniform(size=n) < wm
    Y[minor] = np.array([0.0, 1.4]) + 0.15 * rng.standard_normal((minor.sum(), 2))
    return X, np.clip(Y, -BOX + 1e-6, BOX - 1e-6)


def density(x, y, scen):
    a, s, rot, wm = params(x, scen)
    ang = rot * np.pi / 2
    g = lambda m0, m1, sd: np.exp(-((y[:, 0] - m0) ** 2 + (y[:, 1] - m1) ** 2) / (2 * sd**2)) / (2 * np.pi * sd**2)
    base = 0.5 * g(a * np.cos(ang), a * np.sin(ang), s) + 0.5 * g(-a * np.cos(ang), -a * np.sin(ang), s)
    return (1 - wm) * base + wm * g(0.0, 1.4, 0.15)


def band_diag(est_u, sd_u, eta_true, band, n):
    """Coverage of the anchor's error by the band B = sd_u sqrt(log n) on the level band."""
    B = sd_u * np.sqrt(np.log(n)); err = np.abs(est_u - eta_true)
    return {"band_cov1": float((err <= B)[band].mean()), "band_cov2": float((err <= 2 * B)[band].mean()),
            "err_over_B": float(np.median((err / B)[band]))}


def run(n, seed, scen):
    rng = np.random.default_rng(seed)
    X, Y = sample(n, rng, scen)
    Xv, Yv = sample(max(500, n // 5), rng, scen)
    Xc, Yc = sample(NCAL, rng, scen)
    Z, L = contrastive_rows(X, Y, K, rng)
    Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False)
    Zvs, Lvs = Zv[sub], Lv[sub]

    g = np.linspace(-BOX, BOX, G + 1); g = (g[:-1] + g[1:]) / 2
    cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2); ng = len(Yg)
    er = np.random.default_rng(777)
    cand = er.uniform(0.02, 0.98, size=(20000, 3))
    inR = in_region(cand, scen)
    Xs = np.vstack([cand[inR][:10], cand[~inR][:10]])
    Xm = sample_x(100, er, scen)
    Xall = np.vstack([Xs, Xm])
    Ecal = np.vstack([np.hstack([np.repeat(Xc, ng, 0), np.tile(Yg, (NCAL, 1))]), np.hstack([Xc, Yc])])
    Eev = np.hstack([np.repeat(Xall, ng, 0), np.tile(Yg, (len(Xall), 1))])
    E = np.vstack([Ecal, Eev])
    p_raw = density(Eev[:, :3], Eev[:, 3:], scen).reshape(len(Xall), ng)
    p_true = p_raw / (p_raw.sum(1, keepdims=True) * cell)

    def conformal(eta, name, extra):
        ph = np.maximum(eta_to_density(eta, K), 1e-12)
        pcg = ph[:NCAL * ng].reshape(NCAL, ng); pown = ph[NCAL * ng:NCAL * ng + NCAL]
        pe = ph[NCAL * ng + NCAL:].reshape(len(Xall), ng)
        pcn = pcg / pcg.sum(1, keepdims=True); pe_n = pe / pe.sum(1, keepdims=True)
        s_cal = np.array([pcn[i][pcg[i] > pown[i]].sum() for i in range(NCAL)])
        tau = np.sort(s_cal)[min(int(np.ceil((1 - ALPHA) * (NCAL + 1))), NCAL) - 1]
        out = {k: [] for k in ("covR", "covO", "symR", "symO", "shR", "shO", "shldR", "shldO", "covM", "vol")}
        for j in range(len(Xall)):
            o = np.argsort(-pe_n[j]); ma = np.empty(ng); cs = np.cumsum(pe_n[j][o]); ma[o] = cs - pe_n[j][o]
            Ch = ma <= tau; pt = p_true[j]; cov = float(pt[Ch].sum() * cell)
            ot = np.argsort(-pt); cst = np.cumsum(pt[ot]) * cell
            Cs_ = np.zeros(ng, bool); Cs_[ot[:np.searchsorted(cst, 1 - ALPHA) + 1]] = True
            Cc = np.zeros(ng, bool); Cc[ot[:max(np.searchsorted(cst, cov) + 1, 1)]] = True
            if j >= 20:
                out["covM"].append(cov); continue
            tag = "R" if j < 10 else "O"
            out["cov" + tag].append(abs(cov - 0.9))
            out["sym" + tag].append((Ch ^ Cs_).sum() / Cs_.sum())
            kc = min(max(np.searchsorted(cst, cov) + 1, 1), ng); tc = pt[ot[kc - 1]]
            out["sh" + tag].append(np.abs(pt - tc)[Ch ^ Cc].sum() / tc / Cs_.sum())  # integral form
            out["shld" + tag].append((Ch.sum() - Cc.sum()) / Cs_.sum())
            out["vol"].append(Ch.sum() / Cs_.sum())
        res = {k: float(np.mean(v)) for k, v in out.items()}
        res.update(extra)
        print(json.dumps({"n": n, "scen": scen, "seed": seed, "method": name, **res}), flush=True)

    prn = fit_net(Z, L, Zvs, Lvs, seed); eta_nn = prn(E); val_nn = bce(prn(Zvs), Lvs)
    shrink = (n / 4000) ** (-1 / 9); best = None
    for hx, hy in itertools.product([0.15, 0.25, 0.4, 0.6], [0.2, 0.3, 0.45]):
        sc = np.array([hx * shrink] * 3 + [hy * shrink] * 2)
        loss = bce(local_linear_sd(Z, L, Zvs, sc, list(range(5)))[0], Lvs)
        if best is None or loss < best[0]:
            best = (loss, sc)
    eta_ll, sd = local_linear_sd(Z, L, E, best[1], list(range(5)))
    eta_u, sd_u = local_linear_sd(Z, L, E, 0.85 * best[1], list(range(5)))

    # pointwise diagnostic on the level band (|p - t| <= 0.3 t) at evaluation x's
    ev = slice(NCAL * ng + NCAL, None)
    p_t_raw = p_raw  # unnormalized true density at eval rows (grid) -> compare in eta scale
    eta_true = (p_raw / (p_raw + K / (2 * BOX) ** 2)).ravel()
    tt = np.array([np.sort(p_true[j])[::-1][np.searchsorted(np.cumsum(np.sort(p_true[j])[::-1]) * cell, 1 - ALPHA)]
                   for j in range(len(Xall))])
    band = (np.abs(p_true - tt[:, None]) <= 0.3 * tt[:, None]).ravel()
    rowR = np.repeat(np.r_[np.ones(10, bool), np.zeros(len(Xall) - 10, bool)], ng)
    rowO = np.repeat(np.r_[np.zeros(10, bool), np.ones(10, bool), np.zeros(len(Xall) - 20, bool)], ng)
    diag = {}
    for nm, e in (("NN", eta_nn[ev]), ("LL", eta_ll[ev])):
        err = np.abs(e - eta_true)
        diag[f"err_{nm}_R"] = float(err[band & rowR].mean()); diag[f"err_{nm}_O"] = float(err[band & rowO].mean())
    diag["bandB_R"] = float((sd_u[ev] * np.sqrt(np.log(n)))[band & rowR].mean())
    diag["bandB_O"] = float((sd_u[ev] * np.sqrt(np.log(n)))[band & rowO].mean())

    if __import__("os").environ.get("CERTONLY") == "1":
        band20 = band & (rowR | rowO)
        diag2 = band_diag(eta_u[ev], sd_u[ev], eta_true, band20, n)
        acc = np.abs(eta_nn - eta_u) <= 2.0 * sd_u * np.sqrt(np.log(n))
        accR = float(acc[ev][rowR].mean()); accO = float(acc[ev][rowO].mean())
        conformal(eta_nn, "NN", {"val": val_nn}); conformal(eta_ll, "LL", {"val": best[0]})
        conformal(np.where(acc, eta_nn, eta_ll), "BCERT_k2.0", {"accR": accR, "accO": accO, **diag2})
        conformal(np.where(acc, eta_nn, eta_u), "SCERT_k2.0", {"accR": accR, "accO": accO})
        for kap in (1.0, 2.0):
            Bd = kap * sd_u * np.sqrt(np.log(n))
            conformal(np.clip(eta_nn, eta_u - Bd, eta_u + Bd), f"PROJ_k{kap}", {})
        pick = "NN" if val_nn <= best[0] else "LL"
        conformal(eta_nn if pick == "NN" else eta_ll, "GLOBAL", {"pick": pick})
        return
    conformal(eta_nn, "NN", {"val": val_nn, **diag})
    conformal(eta_ll, "LL", {"val": best[0]})
    for kap in (1.0, 2.0):
        acc = np.abs(eta_nn - eta_u) <= kap * sd_u * np.sqrt(np.log(n))
        accR = float(acc[ev][rowR].mean()); accO = float(acc[ev][rowO].mean())
        conformal(np.where(acc, eta_nn, eta_ll), f"BCERT_k{kap}", {"accR": accR, "accO": accO})
        Bd = kap * sd_u * np.sqrt(np.log(n))
        conformal(np.clip(eta_nn, eta_u - Bd, eta_u + Bd), f"PROJ_k{kap}", {"accR": accR, "accO": accO})
    pick = "NN" if val_nn <= best[0] else "LL"
    conformal(eta_nn if pick == "NN" else eta_ll, "GLOBAL", {"pick": pick})


if __name__ == "__main__":
    n, seed, scen = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    run(n, seed, scen)
