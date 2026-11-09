"""Does contrastive local-polynomial conditional density estimation work in practice?

Model: X ~ U[0,1]^{dX}; Y | X=x in R^2 is a symmetric two-component Gaussian mixture
with means (+-a(x), 0), a(x) = 0.6 + 0.6 x1, sd s(x) = 0.35 (dX=1) or 0.30 + 0.15 x2 (dX>=2).
Response box Y = [-2.5, 2.5]^2, reference q = Uniform(box), K negatives per positive.

Methods: contrastive local linear (LL), contrastive local quadratic (LQ),
conditional KDE (CKDE), contrastive MLP (NN). Bandwidths / early stopping chosen on a
held-out fold (contrastive log-loss for LL/LQ/NN, conditional log-likelihood for CKDE).

Metrics at test x's on a 60x60 y-grid: sup-norm error (whole box and on the level-set
band {|p - t| <= t/2}), L1 error, plug-in 90% HPD set: Leb(C_hat ^ C*)/Leb(C*) and
|coverage - 0.9|.
"""
import itertools, json, sys, time
import numpy as np
from scipy.spatial import cKDTree

BOX = 2.5
Q = 1.0 / (2 * BOX) ** 2
ALPHA = 0.1
GRID = 40


def params(x):
    x = np.atleast_2d(x)
    a = 0.6 + 0.6 * x[:, 0]
    s = 0.35 * np.ones(len(x)) if x.shape[1] == 1 else 0.30 + 0.15 * x[:, 1]
    return a, s


def true_density(x, y):
    """x: (m, dX), y: (m, 2) -> p(y|x)."""
    a, s = params(x)
    out = 0.0
    for sign in (-1, 1):
        d2 = (y[:, 0] - sign * a) ** 2 + y[:, 1] ** 2
        out = out + 0.5 * np.exp(-d2 / (2 * s**2)) / (2 * np.pi * s**2)
    return out


def sample(n, dX, rng):
    X = rng.uniform(size=(n, dX))
    a, s = params(X)
    comp = rng.choice([-1, 1], size=n)
    Y = np.stack([comp * a, np.zeros(n)], 1) + s[:, None] * rng.standard_normal((n, 2))
    Y = np.clip(Y, -BOX + 1e-6, BOX - 1e-6)
    return X, Y


def contrastive_rows(X, Y, K, rng):
    n = len(X)
    Yneg = rng.uniform(-BOX, BOX, size=(n * K, 2))
    Z = np.vstack([np.hstack([X, Y]), np.hstack([np.repeat(X, K, 0), Yneg])])
    L = np.concatenate([np.ones(n), np.zeros(n * K)])
    return Z, L


# ---------------------------------------------------------------- local polynomial
def poly_features(D, degree):
    cols = [np.ones(len(D))]
    d = D.shape[1]
    cols += [D[:, j] for j in range(d)]
    if degree >= 2:
        for j, k in itertools.combinations_with_replacement(range(d), 2):
            cols.append(D[:, j] * D[:, k])
    return np.stack(cols, 1)


def local_poly(Z, L, E, hx, hy, dX, degree):
    """Local polynomial LS of L on Z at points E, product Epanechnikov kernel."""
    scale = np.array([hx] * dX + [hy] * 2)
    tree = cKDTree(Z / scale)
    out = np.empty(len(E))
    nb_all = tree.query_ball_point(E / scale, r=1.0, p=np.inf, workers=-1)
    p_dim = 1 + Z.shape[1] + (Z.shape[1] * (Z.shape[1] + 1) // 2 if degree >= 2 else 0)
    for i, nb in enumerate(nb_all):
        if len(nb) == 0:
            out[i] = 0.0
            continue
        D = (Z[nb] - E[i]) / scale
        w = np.prod(0.75 * (1 - D**2), axis=1)
        if len(nb) < 3 * p_dim:  # fall back to local constant
            out[i] = np.dot(w, L[nb]) / max(w.sum(), 1e-12)
            continue
        F = poly_features(D, degree)
        G = F.T @ (F * w[:, None]) + 1e-8 * np.eye(F.shape[1])
        out[i] = np.linalg.solve(G, F.T @ (w * L[nb]))[0]
    return out


def eta_to_density(eta, K):
    eta = np.clip(eta, 0.0, 0.995)
    return K * Q * eta / (1 - eta)


def bce(eta, L):
    eta = np.clip(eta, 1e-4, 1 - 1e-4)
    return -np.mean(L * np.log(eta) + (1 - L) * np.log(1 - eta))


# ---------------------------------------------------------------- conditional KDE
def ckde(X, Y, Ex, Ey, hx, hy):
    """p_hat(Ey_j | Ex_j) for paired query arrays (grouped by x when queries share x; identical output)."""
    from llfast import ckde_grouped
    if len(np.unique(Ex, axis=0)) < len(Ex) / 20:
        return ckde_grouped(X, Y, Ex, Ey, hx, hy, Q)
    tree = cKDTree(X / hx)
    nb_all = tree.query_ball_point(Ex / hx, r=1.0, p=np.inf, workers=1)
    out = np.empty(len(Ex))
    for i, nb in enumerate(nb_all):
        if len(nb) == 0:
            out[i] = Q
            continue
        D = (X[nb] - Ex[i]) / hx
        w = np.prod(0.75 * (1 - D**2), axis=1)
        k = np.exp(-np.sum((Y[nb] - Ey[i]) ** 2, 1) / (2 * hy**2)) / (2 * np.pi * hy**2)
        out[i] = np.dot(w, k) / max(w.sum(), 1e-12)
    return out


# ---------------------------------------------------------------- contrastive MLP
def fit_nn(Ztr, Ltr, Zva, Lva, seed):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mu, sd = Ztr.mean(0), Ztr.std(0)
    f = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xt, Yt, Xv, Yv = f(Ztr), torch.tensor(Ltr, dtype=torch.float32), f(Zva), torch.tensor(Lva, dtype=torch.float32)
    d = Ztr.shape[1]
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
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    return lambda Z: torch.sigmoid(net(f(Z)).squeeze(1)).detach().numpy()


# ---------------------------------------------------------------- evaluation
def evaluate(p_hat_grid, p_true_grid, cell):
    """Both (m, GRID*GRID); p_hat already normalized per row."""
    res = {k: [] for k in ["sup", "sup_band", "L1", "symdiff", "coverr"]}
    for ph, pt in zip(p_hat_grid, p_true_grid):
        order = np.argsort(-pt)
        cum = np.cumsum(pt[order]) * cell
        t = pt[order][np.searchsorted(cum, 1 - ALPHA)]
        Cstar = pt >= t
        oh = np.argsort(-ph)
        cumh = np.cumsum(ph[oh]) * cell
        th = ph[oh][min(np.searchsorted(cumh, 1 - ALPHA), len(ph) - 1)]
        Ch = ph >= th
        band = np.abs(pt - t) <= t / 2
        res["sup"].append(np.max(np.abs(ph - pt)))
        res["sup_band"].append(np.max(np.abs(ph - pt)[band]))
        res["L1"].append(np.sum(np.abs(ph - pt)) * cell)
        res["symdiff"].append(np.sum(Ch ^ Cstar) / np.sum(Cstar))
        res["coverr"].append(abs(np.sum(pt[Ch]) * cell - (1 - ALPHA)))
    return {k: float(np.mean(v)) for k, v in res.items()}


def run(n, dX, seed, K=10, methods=("LL", "LQ", "CKDE", "NN")):
    rng = np.random.default_rng(seed)
    X, Y = sample(n, dX, rng)
    Xv, Yv = sample(max(300, n // 10), dX, rng)
    Z, L = contrastive_rows(X, Y, K, rng)
    Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False)
    Zvs, Lvs = Zv[sub], Lv[sub]
    # test points
    Xte = np.random.default_rng(999).uniform(0.1, 0.9, size=(5, dX))
    g = np.linspace(-BOX, BOX, GRID + 1)
    g = (g[:-1] + g[1:]) / 2
    cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    Ex = np.repeat(Xte, len(Yg), 0)
    Ey = np.tile(Yg, (len(Xte), 1))
    p_true = true_density(Ex, Ey).reshape(len(Xte), -1)
    p_true = p_true / (p_true.sum(1, keepdims=True) * cell)

    shrink = (n / 4000) ** (-1 / (4 + dX + 2))
    hx_grid = [c * shrink for c in (0.25, 0.4, 0.6)]
    hy_grid = [c * shrink for c in (0.4, 0.6, 0.9)]
    out = {}
    for m in methods:
        t0 = time.time()
        if m in ("LL", "LQ"):
            deg = 1 if m == "LL" else 2
            best = None
            for hx, hy in itertools.product(hx_grid, hy_grid):
                loss = bce(local_poly(Z, L, Zvs, hx, hy, dX, deg), Lvs)
                if best is None or loss < best[0]:
                    best = (loss, hx, hy)
            _, hx, hy = best
            ph = eta_to_density(local_poly(Z, L, np.hstack([Ex, Ey]), hx, hy, dX, deg), K)
            info = {"hx": hx, "hy": hy}
        elif m == "CKDE":
            best = None
            for hx, hy in itertools.product(hx_grid, [c * 0.6 for c in hy_grid]):
                ll = np.mean(np.log(np.maximum(ckde(X, Y, Xv[:1000], Yv[:1000], hx, hy), 1e-8)))
                if best is None or ll > best[0]:
                    best = (ll, hx, hy)
            _, hx, hy = best
            ph = ckde(X, Y, Ex, Ey, hx, hy)
            info = {"hx": hx, "hy": hy}
        else:
            pred = fit_nn(Z, L, Zv, Lv, seed)
            ph = eta_to_density(pred(np.hstack([Ex, Ey])), K)
            info = {}
        ph = np.maximum(ph.reshape(len(Xte), -1), 0)
        ph = ph / (ph.sum(1, keepdims=True) * cell)
        r = evaluate(ph, p_true, cell)
        r.update(info, sec=round(time.time() - t0, 1))
        out[m] = r
        print(json.dumps({"n": n, "dX": dX, "seed": seed, "method": m, **r}), flush=True)
    return out


if __name__ == "__main__":
    dX, n, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    methods = ("LL", "LQ", "CKDE", "NN") if dX == 1 else ("LL", "CKDE", "NN")
    run(n, dX, seed, methods=methods)
