"""Robustness of the certified estimator + tree baselines (no direction learning).

Scenarios
  smooth : Gaussian-mixture conditionals (as in sim_metric), non-axis-aligned 2-index model
  tent   : same geometry but pyramid (L_inf tent) components -> Lipschitz density (beta = 1),
           kinks crossing the HPD level set
Methods (all on contrastive rows, K = 5; bandwidth / hyperparameters by validation loss)
  NN, NN_under (1 hidden layer, width 8), NN_over (no early stopping, 150 epochs, no val),
  RF, ET (extremely randomized trees), HGB (hist. gradient boosting),
  LLiso (isotropic local linear; dX = 3 only),
  CERT_<nn>_k<kappa> : certified network with LLiso fallback (dX = 3 only).
"""
import itertools, json, sys, time
import numpy as np
from scipy.stats import norm

from sim_contrastive import BOX, GRID, contrastive_rows, eta_to_density, bce, evaluate
from sim_metric import directions, params
from sim_sdr import local_linear_sd

K = 5


# ------------------------------------------------------------------ models
def sample(n, dX, W, rng, scen):
    X = rng.uniform(size=(n, dX))
    a, s = params(X, W)
    comp = rng.choice([-1, 1], size=n)
    ctr = np.stack([comp * a, np.zeros(n)], 1)
    if scen == "smooth":
        Y = ctr + s[:, None] * rng.standard_normal((n, 2))
    else:  # pyramid with L_inf radius R = 2.2 s: radius ~ R * Beta(2,2), uniform on the square shell
        R = 2.2 * s
        rad = R * rng.beta(2, 2, size=n)
        u = rng.uniform(-1, 1, size=n); side = rng.integers(4, size=n)
        one = np.ones(n)
        e = np.select([side[:, None] == 0, side[:, None] == 1, side[:, None] == 2],
                      [np.stack([one, u], 1), np.stack([-one, u], 1), np.stack([u, one], 1)],
                      np.stack([u, -one], 1))
        Y = ctr + rad[:, None] * e
    return X, np.clip(Y, -BOX + 1e-6, BOX - 1e-6)


def density(x, y, W, scen):
    a, s = params(x, W)
    out = 0.0
    for sign in (-1, 1):
        d0, d1 = y[:, 0] - sign * a, y[:, 1]
        if scen == "smooth":
            out = out + 0.5 * np.exp(-(d0**2 + d1**2) / (2 * s**2)) / (2 * np.pi * s**2)
        else:
            R = 2.2 * s
            out = out + 0.5 * 3 / (4 * R**2) * np.maximum(1 - np.maximum(np.abs(d0), np.abs(d1)) / R, 0)
    return out


# ------------------------------------------------------------------ networks
def fit_net(Z, L, Zv, Lv, seed, width=128, depth=3, early=True, epochs=200):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    mu, sd = Z.mean(0), Z.std(0)
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    Xt, Yt, Xv, Yv = f(Z), torch.tensor(L, dtype=torch.float32), f(Zv), torch.tensor(Lv, dtype=torch.float32)
    layers, dim = [], Z.shape[1]
    for _ in range(depth):
        layers += [torch.nn.Linear(dim, width), torch.nn.SiLU()]; dim = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(dim, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512 if not early else 1024):
            idx = perm[b:b + 1024]
            opt.zero_grad(); lossf(net(Xt[idx]).squeeze(1), Yt[idx]).backward(); opt.step()
        if early:
            with torch.no_grad():
                v = lossf(net(Xv).squeeze(1), Yv).item()
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
    if early:
        net.load_state_dict(state)
    return lambda A: torch.sigmoid(net(f(A)).squeeze(1)).detach().numpy()


def fit_tree(kind, Z, L, Zvs, Lvs, seed):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    best = None
    grid = {"RF": [5, 20, 60], "ET": [5, 20, 60], "HGB": [(0.05, 20), (0.1, 40), (0.05, 80)]}[kind]
    for g in grid:
        if kind == "RF":
            m = RandomForestClassifier(n_estimators=200, min_samples_leaf=g, max_features=0.6, n_jobs=1, random_state=seed)
        elif kind == "ET":
            m = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=g, max_features=0.8, n_jobs=1, random_state=seed)
        else:
            m = HistGradientBoostingClassifier(learning_rate=g[0], max_leaf_nodes=g[1], max_iter=600,
                                               early_stopping=True, random_state=seed)
        m.fit(Z, L)
        loss = bce(m.predict_proba(Zvs)[:, 1], Lvs)
        if best is None or loss < best[0]:
            best = (loss, m, g)
    return (lambda A: best[1].predict_proba(A)[:, 1]), best[2]


def run(n, dX, seed, scen):
    W = directions(dX)
    rng = np.random.default_rng(seed)
    X, Y = sample(n, dX, W, rng, scen)
    Xv, Yv = sample(max(500, n // 5), dX, W, rng, scen)
    Z, L = contrastive_rows(X, Y, K, rng)
    Zv, Lv = contrastive_rows(Xv, Yv, K, rng)
    sub = rng.choice(len(Zv), size=min(3000, len(Zv)), replace=False)
    Zvs, Lvs = Zv[sub], Lv[sub]

    Xte = np.random.default_rng(999).uniform(0.1, 0.9, size=(5, dX))
    g = np.linspace(-BOX, BOX, GRID + 1); g = (g[:-1] + g[1:]) / 2
    cell = (g[1] - g[0]) ** 2
    Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    Ex, Ey = np.repeat(Xte, len(Yg), 0), np.tile(Yg, (len(Xte), 1))
    p_true = density(Ex, Ey, W, scen).reshape(len(Xte), -1)
    p_true /= p_true.sum(1, keepdims=True) * cell
    E = np.hstack([Ex, Ey])

    def emit(name, eta, extra):
        ph = np.maximum(eta_to_density(eta, K).reshape(len(Xte), -1), 0)
        ph /= ph.sum(1, keepdims=True) * cell
        r = evaluate(ph, p_true, cell); r.update(extra)
        print(json.dumps({"n": n, "dX": dX, "scen": scen, "seed": seed, "method": name, **r}), flush=True)

    nets = {}
    for name, kw in (("NN", {}), ("NN_under", dict(width=8, depth=1)), ("NN_over", dict(early=False, epochs=150))):
        t0 = time.time(); pr = fit_net(Z, L, Zvs, Lvs, seed, **kw); nets[name] = pr(E)
        emit(name, nets[name], {"sec": round(time.time() - t0, 1)})
    for kind in ("RF", "ET", "HGB"):
        t0 = time.time(); pr, hp = fit_tree(kind, Z, L, Zvs, Lvs, seed)
        emit(kind, pr(E), {"hp": str(hp), "sec": round(time.time() - t0, 1)})

    if dX == 3:
        t0 = time.time(); shrink = (n / 4000) ** (-1 / (4 + dX + 2)); best = None
        for hx, hy in itertools.product([0.25, 0.4, 0.6], [0.3, 0.45, 0.65]):
            sc = np.array([hx * shrink] * dX + [hy * shrink] * 2)
            loss = bce(local_linear_sd(Z, L, Zvs, sc, list(range(dX + 2)))[0], Lvs)
            if best is None or loss < best[0]:
                best = (loss, sc)
        est, sd = local_linear_sd(Z, L, E, best[1], list(range(dX + 2)))
        emit("LLiso", est, {"sec": round(time.time() - t0, 1)})
        for nm, eta_nn in nets.items():
            for kap in (1.0, 2.0):
                acc = np.abs(eta_nn - est) <= kap * sd * np.sqrt(np.log(n))
                emit(f"CERT_{nm}_k{kap}", np.where(acc, eta_nn, est), {"accept": float(acc.mean())})


if __name__ == "__main__":
    dX, n, seed, scen = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    run(n, dX, seed, scen)
