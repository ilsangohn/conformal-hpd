"""Pilot: neural-metric local polynomial (Lepski-proxy) and neural-metric GP (empirical Bayes)
for contrastive conditional density estimation under a multi-index model.

Model: X ~ U[0,1]^{dX}; x acts only through two NON-axis-aligned directions w1, w2:
  t_j = sqrt(12) w_j'(x - 0.5),  s_j = Phi(t_j) in (0,1),
  Y | X=x ~ 0.5 N((-a,0), s^2 I) + 0.5 N((a,0), s^2 I),  a = 0.6 + 0.6 s_1,  s = 0.30 + 0.15 s_2.
No sample splitting in estimation: the network, its gradient-outer-product directions, and the
final local fit all use the same training data (a validation fold is used only for tuning).

Methods
  NN      : contrastive MLP (CINDES-type baseline)
  LLiso   : contrastive local linear in (x, y), isotropic in x
  LLmet   : contrastive local linear in (B_hat'x, y) [+ optional wide window in B_perp'x]
  GPmet   : Gaussian-working-likelihood GP (random Fourier features) in (B_hat'x, B_perp'x, y),
            per-group lengthscales chosen by marginal likelihood (empirical Bayes)
  LLora   : local linear in (B'x, y) with the TRUE directions (oracle ceiling)
"""
import itertools, json, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

from sim_contrastive import (BOX, Q, ALPHA, GRID, contrastive_rows, poly_features,
                             eta_to_density, bce, evaluate, fit_nn)

K = 5


def directions(dX):
    rng = np.random.default_rng(7)
    W, _ = np.linalg.qr(rng.standard_normal((dX, 2)))
    return W  # dX x 2, orthonormal


def params(x, W):
    t = np.sqrt(12) * (x - 0.5) @ W
    s1, s2 = norm.cdf(t[:, 0]), norm.cdf(t[:, 1])
    return 0.6 + 0.6 * s1, 0.30 + 0.15 * s2


def true_density(x, y, W):
    a, s = params(x, W)
    out = 0.0
    for sign in (-1, 1):
        d2 = (y[:, 0] - sign * a) ** 2 + y[:, 1] ** 2
        out = out + 0.5 * np.exp(-d2 / (2 * s**2)) / (2 * np.pi * s**2)
    return out


def sample(n, dX, W, rng):
    X = rng.uniform(size=(n, dX))
    a, s = params(X, W)
    comp = rng.choice([-1, 1], size=n)
    Y = np.stack([comp * a, np.zeros(n)], 1) + s[:, None] * rng.standard_normal((n, 2))
    return X, np.clip(Y, -BOX + 1e-6, BOX - 1e-6)


# ------------------------------------------------------------ local linear in given coordinates
def local_linear(C, L, E, scales, poly_cols):
    """Local linear LS at points E. C, E: coordinates (already transformed).
    scales: per-column half-widths (np.inf = column ignored in the kernel).
    poly_cols: columns entering the linear part."""
    use = np.isfinite(scales)
    Cs, Es = C[:, use] / scales[use], E[:, use] / scales[use]
    tree = cKDTree(Cs)
    nb_all = tree.query_ball_point(Es, r=1.0, p=np.inf, workers=-1)
    p_dim = 1 + len(poly_cols)
    out = np.empty(len(E))
    for i, nb in enumerate(nb_all):
        if len(nb) == 0:
            out[i] = 0.0
            continue
        D = Cs[nb] - Es[i]
        w = np.prod(0.75 * (1 - D**2), axis=1)
        if len(nb) < 3 * p_dim:
            out[i] = np.dot(w, L[nb]) / max(w.sum(), 1e-12)
            continue
        F = np.hstack([np.ones((len(nb), 1)), C[nb][:, poly_cols] - E[i, poly_cols]])
        G = F.T @ (F * w[:, None]) + 1e-8 * np.eye(p_dim)
        out[i] = np.linalg.solve(G, F.T @ (w * L[nb]))[0]
    return out


# ------------------------------------------------------------ GOP directions from the network
def gop_directions(net_logit_grad, Xs, Ys, k=2):
    G = net_logit_grad(Xs, Ys)            # (m, dX) gradients wrt x
    M = G.T @ G / len(G)
    vals, vecs = np.linalg.eigh(M)
    B = vecs[:, ::-1][:, :k]
    Bperp = vecs[:, ::-1][:, k:]
    return B, Bperp, vals[::-1]


def fit_nn_with_grad(Z, L, Zv, Lv, dX, seed):
    """Same MLP as sim_contrastive.fit_nn, but also returns d logit / d x."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mu, sd = Z.mean(0), Z.std(0)
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    Xt, Yt, Xv, Yv = f(Z), torch.tensor(L, dtype=torch.float32), f(Zv), torch.tensor(Lv, dtype=torch.float32)
    d = Z.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                              torch.nn.SiLU(), torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad = np.inf, None, 0
    for epoch in range(200):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 1024):
            idx = perm[b:b + 1024]
            opt.zero_grad()
            lossf(net(Xt[idx]).squeeze(1), Yt[idx]).backward()
            opt.step()
        with torch.no_grad():
            v = lossf(net(Xv).squeeze(1), Yv).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k_: t.clone() for k_, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    predict = lambda A: torch.sigmoid(net(f(A)).squeeze(1)).detach().numpy()

    def grad_x(Xs, Ys):
        z = f(np.hstack([Xs, Ys])).requires_grad_(True)
        net(z).sum().backward()
        return (z.grad.numpy()[:, :dX]) / sd[:dX]   # chain rule for standardization
    return predict, grad_x


# ------------------------------------------------------------ RFF GP with empirical Bayes
def rff(C, ls, Om, b):
    use = np.isfinite(ls)
    Zs = C[:, use] / ls[use]
    return np.sqrt(2.0 / Om.shape[1]) * np.cos(Zs @ Om[use] + b)


def gp_fit_predict(C, L, E, ls, Om, b, sf2=0.25, s2=0.12):
    P = rff(C, ls, Om, b).astype(np.float64)
    m = L.mean()
    A = P.T @ P + (s2 / sf2) * np.eye(P.shape[1])
    w = np.linalg.solve(A, P.T @ (L - m))
    return m + rff(E, ls, Om, b) @ w


def gp_log_marglik(C, L, ls, Om, b, sf2=0.25, s2=0.12):
    P = rff(C, ls, Om, b)
    y = L - L.mean()
    D = P.shape[1]
    A = P.T @ P / s2 + np.eye(D) / sf2
    cf = np.linalg.cholesky(A)
    v = np.linalg.solve(cf, P.T @ y / s2)
    logdetA = 2 * np.log(np.diag(cf)).sum()
    N = len(y)
    return -0.5 * (y @ y / s2 - v @ v) - 0.5 * (logdetA + D * np.log(sf2) + N * np.log(s2))


# ------------------------------------------------------------ main experiment
def run(n, dX, seed, methods=("NN", "LLiso", "LLmet", "NNLP", "GPmet", "LLora")):
    W = directions(dX)
    rng = np.random.default_rng(seed)
    X, Y = sample(n, dX, W, rng)
    Xv, Yv = sample(max(500, n // 5), dX, W, rng)
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

    out, t0 = {}, time.time()
    # --- network (always fitted: needed for directions)
    predict, grad_x = fit_nn_with_grad(Z, L, Zvs, Lvs, dX, seed)
    t_nn = time.time() - t0
    idx = rng.choice(len(Z), size=min(6000, len(Z)), replace=False)
    B, Bp, ev = gop_directions(grad_x, Z[idx, :dX], Z[idx, dX:])
    angle = float(np.linalg.norm(B @ B.T - W @ W.T) / np.sqrt(2))

    def coords(A, Bm, Bpm):
        x, y = A[:, :dX], A[:, dX:]
        cols = [(x - 0.5) @ Bm]
        if Bpm is not None and Bpm.shape[1] > 0:
            cols.append((x - 0.5) @ Bpm)
        return np.hstack(cols + [y])

    def record(name, eta_or_p, info, t, is_eta=True):
        ph = eta_to_density(eta_or_p, K) if is_eta else eta_or_p
        ph = np.maximum(ph.reshape(len(Xte), -1), 0)
        ph /= ph.sum(1, keepdims=True) * cell
        r = evaluate(ph, p_true, cell)
        r.update(info, sec=round(t, 1), angle=angle)
        out[name] = r
        print(json.dumps({"n": n, "dX": dX, "seed": seed, "method": name, **r}), flush=True)

    record("NN", predict(E), {}, t_nn)

    su = np.std((X - 0.5) @ W[:, 0])  # scale of index coordinates
    hy_grid = [0.3, 0.45, 0.65]
    shrink = (n / 4000) ** (-1 / (4 + 4))

    # --- isotropic LL in (x, y)
    if "LLiso" in methods:
        t1 = time.time(); best = None
        for hx, hy in itertools.product([0.25, 0.4, 0.6], hy_grid):
            sc = np.array([hx * (n / 4000) ** (-1 / (4 + dX + 2))] * dX + [hy * shrink] * 2)
            loss = bce(local_linear(Z, L, Zvs, sc, list(range(dX + 2))), Lvs)
            if best is None or loss < best[0]:
                best = (loss, sc)
        record("LLiso", local_linear(Z, L, E, best[1], list(range(dX + 2))), {}, time.time() - t1 + 0)

    # --- neural-metric LL: candidates (index window, perp window in {inf, wide}) selected on validation
    for name, Bm, Bpm in (("LLmet", B, Bp), ("LLora", W, None)):
        if name not in methods:
            continue
        t1 = time.time()
        C, Cv, Ce = coords(Z, Bm, Bpm), coords(Zvs, Bm, Bpm), coords(E, Bm, Bpm)
        nperp = 0 if Bpm is None else Bpm.shape[1]
        poly = [0, 1, 2 + nperp, 3 + nperp]            # linear part only in (index, y)
        best = None
        perp_opts = [np.inf] if nperp == 0 else [np.inf, 3.0]
        for hu, hy, cp in itertools.product([0.15, 0.25, 0.4], hy_grid, perp_opts):
            sc = np.array([hu * su * 3 * shrink] * 2 + [cp * hu * su * 3] * nperp + [hy * shrink] * 2)
            loss = bce(local_linear(C, L, Cv, sc, poly), Lvs)
            if best is None or loss < best[0]:
                best = (loss, sc, cp)
        eta = local_linear(C, L, Ce, best[1], poly)
        record(name, eta, {"perp": str(best[2])}, time.time() - t1 + (t_nn if name == "LLmet" else 0))

    # --- NN + local-linear residual correction in neural-metric coordinates (same data)
    if "NNLP" in methods:
        t1 = time.time()
        C, Cv, Ce = coords(Z, B, Bp), coords(Zvs, B, Bp), coords(E, B, Bp)
        nperp = Bp.shape[1]
        poly = [0, 1, 2 + nperp, 3 + nperp]
        R = L - predict(Z)
        base_v, base_e = predict(Zvs), predict(E)
        best = (bce(base_v, Lvs), None, "NNonly")
        for hu, hy, cp in itertools.product([0.25, 0.5, 1.0], [0.45, 0.9, 1.8], [np.inf, 3.0]):
            sc = np.array([hu * su * 3 * shrink] * 2 + [cp * hu * su * 3] * nperp + [hy * shrink] * 2)
            loss = bce(np.clip(base_v + local_linear(C, R, Cv, sc, poly), 1e-4, 1), Lvs)
            if loss < best[0]:
                best = (loss, sc, (hu, hy, cp))
        eta = base_e if best[1] is None else base_e + local_linear(C, R, Ce, best[1], poly)
        record("NNLP", eta, {"sel": str(best[2])}, time.time() - t1 + t_nn)

    # --- neural-metric GP with empirical Bayes lengthscales
    if "GPmet" in methods:
        t1 = time.time()
        C, Ce = coords(Z, B, Bp), coords(E, B, Bp)
        nperp = Bp.shape[1]
        Dfeat = 800
        rr = np.random.default_rng(seed + 100)
        Om = rr.standard_normal((C.shape[1], Dfeat)); bb = rr.uniform(0, 2 * np.pi, Dfeat)
        subi = rr.choice(len(C), size=min(8000, len(C)), replace=False)
        best = None
        for lu, ly, lp, sf2, s2 in itertools.product([0.2, 0.4, 0.8, 1.6, 3.2], [0.4, 0.8, 1.6], [np.inf, 1.0], [0.05, 0.25], [0.06, 0.12]):
            ls = np.array([lu] * 2 + [lp] * nperp + [ly] * 2)
            lml = gp_log_marglik(C[subi], L[subi], ls, Om, bb, sf2=sf2, s2=s2)
            if best is None or lml > best[0]:
                best = (lml, ls, (lu, ly, lp, sf2, s2))
        eta = gp_fit_predict(C, L, Ce, best[1], Om, bb, sf2=best[2][3], s2=best[2][4])
        record("GPmet", eta, {"ls": str(best[2])}, time.time() - t1 + t_nn)
    return out


if __name__ == "__main__":
    dX, n, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    run(n, dX, seed)
