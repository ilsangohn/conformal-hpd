"""Direction estimators (SDR) + structure-adapted local linear + certified NN.

Model (multi-index, non-axis-aligned): as in sim_metric.py; design uniform on the cube or
Gaussian with the same mean/variance (N(0.5, I/12)).
Direction estimators (k=2, all computed from the SAME training pairs, no splitting):
  SAVE, DR (directional regression), OPG (local-linear gradients of random Fourier features
  of Y), gKDR (gradient-based kernel dimension reduction), NNGOP, ORACLE.
For each B_hat: contrastive local linear in (B_hat'x, y) (bandwidths by validation loss).
Certified NN: use NN at z iff |eta_NN - eta_LL| <= kappa * sd_LL(z) * sqrt(log n), with LL on DR
directions; report acceptance rates for several kappa.
"""
import itertools, json, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import sqrtm

from sim_contrastive import BOX, GRID, contrastive_rows, eta_to_density, bce, evaluate
from sim_metric import directions, params, true_density, fit_nn_with_grad, gop_directions

K = 5
KAPPAS = (0.5, 1.0, 2.0)


def sample(n, dX, W, rng, design):
    if design == "unif":
        X = rng.uniform(size=(n, dX))
    else:
        X = 0.5 + rng.standard_normal((n, dX)) / np.sqrt(12)
    a, s = params(X, W)
    comp = rng.choice([-1, 1], size=n)
    Y = np.stack([comp * a, np.zeros(n)], 1) + s[:, None] * rng.standard_normal((n, 2))
    return X, np.clip(Y, -BOX + 1e-6, BOX - 1e-6)


def orth(B):
    q, _ = np.linalg.qr(B)
    return q


def angle(B, W):
    return float(np.linalg.norm(orth(B) @ orth(B).T - W @ W.T) / np.sqrt(2))


# ------------------------------------------------------------ SDR direction estimators
def slices(Y, H, rng):
    from scipy.cluster.vq import kmeans2
    _, lab = kmeans2(Y, H, seed=int(rng.integers(1 << 30)), minit="++")
    return lab


def standardize(X):
    mu, S = X.mean(0), np.cov(X.T)
    Sih = np.real(np.linalg.inv(sqrtm(S)))
    return (X - mu) @ Sih, Sih


def save_dirs(X, Y, k, rng, H=10):
    Z, Sih = standardize(X)
    lab = slices(Y, H, rng)
    d = X.shape[1]; M = np.zeros((d, d))
    for h in np.unique(lab):
        Zh = Z[lab == h]; ph = len(Zh) / len(Z)
        A = np.eye(d) - np.cov(Zh.T)
        M += ph * A @ A
    v = np.linalg.eigh(M)[1][:, ::-1][:, :k]
    return orth(Sih @ v)


def dr_dirs(X, Y, k, rng, H=10):
    Z, Sih = standardize(X)
    lab = slices(Y, H, rng)
    d = X.shape[1]
    A1 = np.zeros((d, d)); A2 = np.zeros((d, d)); a3 = 0.0
    for h in np.unique(lab):
        Zh = Z[lab == h]; ph = len(Zh) / len(Z)
        Ezz = Zh.T @ Zh / len(Zh); m = Zh.mean(0)
        A1 += ph * Ezz @ Ezz
        A2 += ph * np.outer(m, m)
        a3 += ph * m @ m
    M = 2 * A1 + 2 * A2 @ A2 + 2 * a3 * A2 - 2 * np.eye(d)
    v = np.linalg.eigh(M)[1][:, ::-1][:, :k]
    return orth(Sih @ v)


def rff_y(Y, m, rng, scale=0.5):
    Om = rng.standard_normal((2, m)) / scale
    b = rng.uniform(0, 2 * np.pi, m)
    return np.cos(Y @ Om + b)


def opg_dirs(X, Y, k, rng, m=20, nsub=800):
    n, d = X.shape
    T = rff_y(Y, m, rng)
    Z, Sih = standardize(X)
    h = 1.2 * n ** (-1 / (d + 4)) * np.sqrt(d)
    idx = rng.choice(n, size=min(nsub, n), replace=False)
    M = np.zeros((d, d))
    for i in idx:
        D = Z - Z[i]
        w = np.exp(-np.sum(D**2, 1) / (2 * h**2))
        F = np.hstack([np.ones((n, 1)), D])
        G = F.T @ (F * w[:, None]) + 1e-6 * np.eye(d + 1)
        coef = np.linalg.solve(G, F.T @ (w[:, None] * T))   # (d+1, m)
        Bg = coef[1:]                                         # gradients (d, m)
        M += Bg @ Bg.T
    v = np.linalg.eigh(M)[1][:, ::-1][:, :k]
    return orth(Sih @ v)


def gkdr_dirs(X, Y, k, rng, nsub=1500, eps=1e-3):
    n, d = X.shape
    idx = rng.choice(n, size=min(nsub, n), replace=False)
    Z, Sih = standardize(X[idx]); Yy = Y[idx]; m = len(idx)
    sx = np.sqrt(d) * 1.0; sy = np.median(np.linalg.norm(Yy[:, None] - Yy[None], axis=2))
    D2 = np.sum((Z[:, None] - Z[None]) ** 2, 2)
    Gx = np.exp(-D2 / (2 * sx**2))
    Gy = np.exp(-np.sum((Yy[:, None] - Yy[None]) ** 2, 2) / (2 * sy**2))
    R = np.linalg.solve(Gx + m * eps * np.eye(m), Gy)
    R = np.linalg.solve(Gx + m * eps * np.eye(m), R.T).T     # (Gx+)^{-1} Gy (Gx+)^{-1}
    M = np.zeros((d, d))
    for i in range(m):
        dk = -(Z[i] - Z) / sx**2 * Gx[i][:, None]             # (m, d): grad_x k(Z_j, x) at x=Z_i
        M += dk.T @ R @ dk
    v = np.linalg.eigh(M)[1][:, ::-1][:, :k]
    return orth(Sih @ v)


# ------------------------------------------------------------ local linear with sd
def local_linear_sd(C, L, E, scales, poly_cols):
    Cs, Es = C / scales, E / scales
    tree = cKDTree(Cs)
    nb_all = tree.query_ball_point(Es, r=1.0, p=np.inf, workers=int(__import__("os").environ.get("KD_WORKERS", "1")))
    p_dim = 1 + len(poly_cols)
    est = np.empty(len(E)); sd = np.empty(len(E))
    for i, nb in enumerate(nb_all):
        if len(nb) < 3 * p_dim:
            if not nb:
                est[i], sd[i] = 0.0, 1.0
                continue
            w = np.prod(0.75 * (1 - (Cs[nb] - Es[i]) ** 2), axis=1)
            est[i] = np.dot(w, L[nb]) / max(w.sum(), 1e-12)
            e = min(max(est[i], 1e-3), 1 - 1e-3)
            sd[i] = np.sqrt(np.sum(w**2) * e * (1 - e)) / max(w.sum(), 1e-12)   # local-constant (NW) s.e.
            continue
        D = Cs[nb] - Es[i]
        w = np.prod(0.75 * (1 - D**2), axis=1)
        F = np.hstack([np.ones((len(nb), 1)), C[nb][:, poly_cols] - E[i, poly_cols]])
        G = F.T @ (F * w[:, None]) + 1e-8 * np.eye(p_dim)
        ell = np.linalg.solve(G, (F * w[:, None]).T)[0]      # equivalent weights
        est[i] = ell @ L[nb]
        e = min(max(est[i], 1e-3), 1 - 1e-3)
        sd[i] = np.sqrt(np.sum(ell**2) * e * (1 - e))
    return est, sd


def run(n, dX, seed, design):
    W = directions(dX)
    rng = np.random.default_rng(seed)
    X, Y = sample(n, dX, W, rng, design)
    Xv, Yv = sample(max(500, n // 5), dX, W, rng, design)
    Z, L = contrastive_rows(X, Y, K, rng)
    Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False)
    Zvs, Lvs = Zv[sub], Lv[sub]

    Xte = np.random.default_rng(999).uniform(0.1, 0.9, size=(5, dX))
    g = np.linspace(-BOX, BOX, GRID + 1); g = (g[:-1] + g[1:]) / 2
    cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    Ex, Ey = np.repeat(Xte, len(Yg), 0), np.tile(Yg, (len(Xte), 1))
    p_true = true_density(Ex, Ey, W).reshape(len(Xte), -1)
    p_true /= p_true.sum(1, keepdims=True) * cell
    E = np.hstack([Ex, Ey])

    def emit(name, eta, extra):
        ph = np.maximum(eta_to_density(eta, K).reshape(len(Xte), -1), 0)
        ph /= ph.sum(1, keepdims=True) * cell
        r = evaluate(ph, p_true, cell); r.update(extra)
        print(json.dumps({"n": n, "dX": dX, "design": design, "seed": seed, "method": name, **r}), flush=True)

    t0 = time.time()
    predict, grad_x = fit_nn_with_grad(Z, L, Zvs, Lvs, dX, seed)
    t_nn = time.time() - t0
    eta_nn = predict(E)
    emit("NN", eta_nn, {"sec": round(t_nn, 1)})

    dirs = {}
    for name, fn in (("SAVE", lambda: save_dirs(X, Y, 2, rng)), ("DR", lambda: dr_dirs(X, Y, 2, rng, H=max(10, n // 160))),
                     ("OPG", lambda: opg_dirs(X, Y, 2, rng)), ("gKDR", lambda: gkdr_dirs(X, Y, 2, rng)),
                     ("NNGOP", lambda: gop_directions(grad_x, Z[:6000, :dX], Z[:6000, dX:])[0]),
                     ("ORACLE", lambda: W)):
        t1 = time.time(); B = fn(); dirs[name] = (B, time.time() - t1)

    su = 1 / np.sqrt(12)
    shrink = (n / 4000) ** (-1 / 8)
    ll_cache = {}
    for name, (B, tdir) in dirs.items():
        t1 = time.time()
        C = np.hstack([(Z[:, :dX] - 0.5) @ B, Z[:, dX:]])
        Cv = np.hstack([(Zvs[:, :dX] - 0.5) @ B, Zvs[:, dX:]])
        Ce = np.hstack([(E[:, :dX] - 0.5) @ B, E[:, dX:]])
        best = None
        for hu, hy in itertools.product([0.15, 0.25, 0.4], [0.3, 0.45, 0.65]):
            sc = np.array([hu * su * 3 * shrink] * 2 + [hy * shrink] * 2)
            loss = bce(local_linear_sd(C, L, Cv, sc, [0, 1, 2, 3])[0], Lvs)
            if best is None or loss < best[0]:
                best = (loss, sc)
        est, sd = local_linear_sd(C, L, Ce, best[1], [0, 1, 2, 3])
        ll_cache[name] = (est, sd)
        emit("LL_" + name, est, {"angle": angle(B, W), "sec_dir": round(tdir, 1), "sec": round(time.time() - t1, 1)})

    # certified NN on DR directions (theory-backed candidate) and on ORACLE for reference
    for base in ("DR", "gKDR"):
        est, sd = ll_cache[base]
        for kap in KAPPAS:
            acc = np.abs(eta_nn - est) <= kap * sd * np.sqrt(np.log(n))
            emit(f"CERT_{base}_k{kap}", np.where(acc, eta_nn, est),
                 {"accept": float(acc.mean()), "angle": angle(dirs[base][0], W)})


if __name__ == "__main__":
    dX, n, seed, design = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    run(n, dX, seed, design)
