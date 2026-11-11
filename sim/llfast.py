"""Local linear estimates of the contrastive regression: the direct implementation local_linear_sd and its batched
(vectorized) version local_linear_sd_fast with identical output; conditional KDE (ckde_grouped)."""
import numpy as np
from scipy.spatial import cKDTree


def local_linear_sd(C, L, E, scales, poly_cols):
    """Direct local linear estimate with product Epanechnikov weights and its standard error (one query at a time)."""
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


def local_linear_sd_fast(C, L, E, scales, poly_cols, chunk=4000, workers=1):
    Cs, Es = C / scales, E / scales
    tree = cKDTree(Cs)
    p_dim = 1 + len(poly_cols)
    est = np.empty(len(E)); sd = np.empty(len(E))
    Cp = C[:, poly_cols]
    for s0 in range(0, len(E), chunk):
        idx = np.arange(s0, min(s0 + chunk, len(E)))
        nbl = tree.query_ball_point(Es[idx], r=1.0, p=np.inf, workers=workers)
        lens = np.array([len(v) for v in nbl]); Lm = max(int(lens.max()), 1)
        NB = np.zeros((len(idx), Lm), dtype=np.int64); M = np.zeros((len(idx), Lm), bool)
        for i, v in enumerate(nbl):
            if v:
                NB[i, :len(v)] = v; M[i, :len(v)] = True
        D = Cs[NB] - Es[idx][:, None, :]
        w = np.prod(0.75 * (1 - D ** 2), axis=2) * M
        Lab = L[NB] * M
        # local-constant fallback (few neighbours)
        ws = w.sum(1)
        nw = (w * Lab).sum(1) / np.maximum(ws, 1e-12)
        e_nw = np.clip(nw, 1e-3, 1 - 1e-3)
        sd_nw = np.sqrt((w ** 2).sum(1) * e_nw * (1 - e_nw)) / np.maximum(ws, 1e-12)
        # local linear
        F = np.concatenate([np.ones((len(idx), Lm, 1)), Cp[NB] - E[idx][:, None, poly_cols]], axis=2)
        Fw = F * w[:, :, None]
        G = np.einsum("blp,blq->bpq", Fw, F) + 1e-8 * np.eye(p_dim)
        e1 = np.zeros((len(idx), p_dim, 1)); e1[:, 0, 0] = 1.0
        a = np.linalg.solve(G, e1)[:, :, 0]                       # G symmetric: e1' G^{-1}
        ell = np.einsum("blp,bp->bl", Fw, a)
        ll = (ell * Lab).sum(1)
        e_ll = np.clip(ll, 1e-3, 1 - 1e-3)
        sd_ll = np.sqrt((ell ** 2).sum(1) * e_ll * (1 - e_ll))
        few = lens < 3 * p_dim
        est[idx] = np.where(few, nw, ll); sd[idx] = np.where(few, sd_nw, sd_ll)
        empty = lens == 0
        est[idx[empty]] = 0.0; sd[idx[empty]] = 1.0
    return est, sd


def _groups(Ex):
    u, inv = np.unique(Ex, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv.ravel(), minlength=len(u)))]
    return u, order, bounds


def local_linear_grouped(C, L, E, scales, poly_cols, dx=3):
    """Exact local linear (same output as local_linear_sd) for query sets in which many rows share x.
    For each distinct x, the x-window rows are found once and all y-queries are solved via moment sums."""
    assert list(poly_cols) == list(range(C.shape[1]))
    Ex, Ey = E[:, :dx], E[:, dx:]
    u, order, bounds = _groups(Ex)
    if len(u) > len(E) / 20:
        return local_linear_sd_fast(C, L, E, scales, poly_cols)
    sx, sy = scales[:dx], scales[dx:]
    tree = cKDTree(C[:, :dx] / sx)
    p = C.shape[1] + 1
    est = np.empty(len(E)); sd = np.empty(len(E))
    for g, x0 in enumerate(u):
        rows = order[bounds[g]:bounds[g + 1]]
        nb = np.array(tree.query_ball_point(x0 / sx, r=1.0, p=np.inf), dtype=np.int64)
        yq = Ey[rows]
        if len(nb) == 0:
            est[rows], sd[rows] = 0.0, 1.0; continue
        Dx = C[nb, :dx] - x0                                  # (m, dx)
        wx = np.prod(0.75 * (1 - (Dx / sx) ** 2), axis=1)     # (m,)
        Yn = C[nb, dx:]                                       # (m, dy)
        Dy = (Yn[None, :, :] - yq[:, None, :]) / sy           # (k, m, dy)
        inside = np.all(np.abs(Dy) <= 1.0, axis=2)
        W = wx[None, :] * np.prod(0.75 * (1 - Dy ** 2), axis=2) * inside    # (k, m)
        cnt = inside.sum(1)
        Lb = L[nb]
        # features f_j(y) = [1, Dx_j, Yn_j - y]; build per-query moment matrices via broadcasting in chunks
        F0 = np.concatenate([np.ones((len(nb), 1)), Dx, Yn], axis=1)   # y-independent part (m, p)
        shift = np.zeros((len(rows), p)); shift[:, 1 + dx:] = yq          # f_j(y) = F0_j - shift(y)
        def mom(Wm):
            A = Wm @ np.einsum("mp,mq->mpq", F0, F0).reshape(len(nb), -1)  # sum w F0 F0'
            A = A.reshape(len(rows), p, p)
            b = Wm @ F0                                                    # sum w F0
            s0 = Wm.sum(1)
            return A - np.einsum("kp,kq->kpq", b, shift) - np.einsum("kp,kq->kpq", shift, b) + s0[:, None, None] * np.einsum("kp,kq->kpq", shift, shift), b, s0
        G, b1, s0 = mom(W)
        G = G + 1e-8 * np.eye(p)
        a = np.linalg.solve(G, np.tile(np.eye(p)[:, :1], (len(rows), 1, 1)))[:, :, 0]
        WL = W * Lb[None, :]
        rhs = WL @ F0 - (WL.sum(1))[:, None] * shift                       # sum w L f
        ll = np.einsum("kp,kp->k", a, rhs)
        G2, _, _ = mom(W ** 2)
        s2 = np.einsum("kp,kpq,kq->k", a, G2, a)
        e_ll = np.clip(ll, 1e-3, 1 - 1e-3); sd_ll = np.sqrt(np.maximum(s2, 0) * e_ll * (1 - e_ll))
        nw = WL.sum(1) / np.maximum(s0, 1e-12)
        e_nw = np.clip(nw, 1e-3, 1 - 1e-3)
        sd_nw = np.sqrt((W ** 2).sum(1) * e_nw * (1 - e_nw)) / np.maximum(s0, 1e-12)
        few = cnt < 3 * p
        e_out = np.where(few, nw, ll); s_out = np.where(few, sd_nw, sd_ll)
        e_out[cnt == 0] = 0.0; s_out[cnt == 0] = 1.0
        est[rows] = e_out; sd[rows] = s_out
    return est, sd


def ckde_grouped(X, Y, Ex, Ey, hx, hy, Q):
    u, order, bounds = _groups(Ex)
    tree = cKDTree(X / hx)
    out = np.empty(len(Ex))
    for g, x0 in enumerate(u):
        rows = order[bounds[g]:bounds[g + 1]]
        nb = np.array(tree.query_ball_point(x0 / hx, r=1.0, p=np.inf), dtype=np.int64)
        if len(nb) == 0:
            out[rows] = Q; continue
        w = np.prod(0.75 * (1 - ((X[nb] - x0) / hx) ** 2), axis=1)
        d2 = ((Y[nb][None, :, :] - Ey[rows][:, None, :]) ** 2).sum(2)
        k = np.exp(-d2 / (2 * hy ** 2)) / (2 * np.pi * hy ** 2)
        out[rows] = (k @ w) / max(w.sum(), 1e-12)
    return out
