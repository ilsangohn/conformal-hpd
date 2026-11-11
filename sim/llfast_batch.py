"""Batched-moment version of llfast.local_linear_sd (estimate only; identical up to floating-point rounding).

Queries are grouped by their covariate x (the first d - ydim columns of E).

Large groups (>= min_group queries sharing x; the y-grid and auxiliary-Y blocks): the contrastive rows (presorted
by scaled y_1) in the x-box |Cs_x - Es_x|_inf <= 1 are found once by a vectorised box test, with their x-weights
prod_k 0.75 (1 - Dx_k^2). The y's of that x are processed in tiles that are compact in y (sorted by y_1
into bands, by y_2 within a band); a tile of B queries meets only the rows of its y-bounding box (+1), and for
those the B x R weight matrix W = wx * prod_k 0.75 (1 - Dy_k^2) * 1{|Dy|_inf <= 1} is formed (for product-structured
tiles such as the y-grid, from per-dimension factors on the distinct y_1 and y_2 values). Neighbour counts are
the row sums of the indicator. One product W @ P, with P the 21 Gram and 6 response features of the basis
Phi = [1, C_poly - c0] (c0 = x exactly in the x-coordinates, the tile centre y0 in the y-coordinates), gives the
moments; they are shifted algebraically to the centre E[i, poly_cols] of each query (F = Phi - s, s = E_y - y0):
G = M - s M_.0' - M_.0 s' + M_00 s s', b = m - s m_0. Then G + 1e-8 I is solved for all queries at once and
est = (G^{-1} b)_0 = e1' G^{-1} F' W L. Branches as in the original: #nb == 0 -> 0, #nb < 3 p -> NW.

Small groups (singletons etc.): the original 5-d KD-tree ball query, padded per-query arrays, the centred design
built directly and a batched solve.
"""
import numpy as np
from scipy.spatial import cKDTree


def local_linear_fast(C, L, E, scales, poly_cols, chunk=100000, ydim=2, band=480, tile=64, min_group=64,
                      max_elems=4_000_000, return_counts=False):
    """Drop-in for llfast.local_linear_sd(...)[0]; returns (est, None) (or (est, None, #neighbours)).

    chunk      : queries per outer chunk (per-query moment arrays: ~100 doubles per query)
    band, tile : queries per y_1-band and per tile inside a band (large groups)
    min_group  : groups with fewer queries go to the padded per-query path
    max_elems  : cap on B x R (tiles) and B x maxlen x p (padded path) per block; 8 bytes each
    """
    C = np.asarray(C, float); E = np.asarray(E, float); L = np.asarray(L, float)
    scales = np.asarray(scales, float)
    S = _Setup(C, L, scales, list(poly_cols), ydim, band, tile, min_group, max_elems)
    est = np.empty(len(E)); cnt = np.empty(len(E), np.int64)
    for s0 in range(0, len(E), chunk):
        est[s0:s0 + chunk], cnt[s0:s0 + chunk] = S.chunk(E[s0:s0 + chunk])
    return (est, None, cnt) if return_counts else (est, None)


class _Setup:
    def __init__(self, C, L, scales, poly, ydim, band, tile, min_group, max_elems):
        d = C.shape[1]
        self.xc, self.yc = np.arange(d - ydim), np.arange(d - ydim, d)
        Cs = C / scales
        o = np.argsort(Cs[:, self.yc[0]], kind="stable")                     # rows sorted by scaled y_1
        self.C, self.L, self.Cs, self.scales, self.poly = C[o], L[o], Cs[o], scales, poly
        self.Csx = np.ascontiguousarray(self.Cs[:, self.xc])
        self.p = 1 + len(poly)
        self.iu = np.triu_indices(self.p); self.nf = len(self.iu[0])
        xset, yset = set(self.xc.tolist()), set(self.yc.tolist())
        self.xpos = [j + 1 for j, c in enumerate(poly) if c in xset]          # basis positions of x-columns
        self.ypos = [j + 1 for j, c in enumerate(poly) if c in yset]          # ... and of y-columns
        self.ypos_dim = [int(np.flatnonzero(self.yc == poly[j - 1])[0]) for j in self.ypos]
        self.band, self.tile, self.min_group, self.max_elems = band, tile, min_group, max_elems
        self.kfac = 0.75 ** ydim
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.Cs)
        return self._tree

    # ------------------------------------------------------------------ one chunk of queries
    def chunk(self, E):
        Es = E / self.scales
        nq, p = len(E), self.p
        est = np.zeros(nq); cnt = np.zeros(nq, np.int64)
        ux, inv = np.unique(E[:, self.xc], axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        order = np.argsort(inv, kind="stable"); bounds = np.searchsorted(inv[order], np.arange(len(ux) + 1))
        sizes = np.diff(bounds)
        big = np.flatnonzero(sizes >= self.min_group)
        small = np.flatnonzero(sizes < self.min_group)
        if len(small):
            qs = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in small])
            est[qs], cnt[qs] = self._padded(E[qs], Es[qs])
        if len(big):
            qb = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in big])
            self.M = np.zeros((nq, self.nf)); self.m = np.zeros((nq, p))
            self.cnt = np.zeros(nq, np.int64); self.shift = np.zeros((nq, p))
            self.E, self.Es = E, Es
            for g in big:
                q = order[bounds[g]:bounds[g + 1]]
                self._group(q, Es[q[0], self.xc])
            est[qb] = self._solve(qb); cnt[qb] = self.cnt[qb]
            del self.M, self.m, self.cnt, self.shift, self.E, self.Es
        return est, cnt

    def _solve(self, q):
        p, iu = self.p, self.iu
        Mq, mq, s, cnt = self.M[q], self.m[q], self.shift[q], self.cnt[q]
        G = np.empty((len(q), p, p)); G[:, iu[0], iu[1]] = Mq; G[:, iu[1], iu[0]] = Mq
        c0 = G[:, :, 0].copy(); M00 = c0[:, 0].copy(); m0 = mq[:, 0].copy()
        G -= s[:, :, None] * c0[:, None, :]
        G -= c0[:, :, None] * s[:, None, :]
        G += M00[:, None, None] * (s[:, :, None] * s[:, None, :])
        b = mq - s * m0[:, None]
        G += 1e-8 * np.eye(p)
        est = np.zeros(len(q))
        ll = cnt >= 3 * p
        if ll.any():
            est[ll] = np.linalg.solve(G[ll], b[ll][:, :, None])[:, 0, 0]
        nw = (cnt > 0) & ~ll
        est[nw] = m0[nw] / np.maximum(M00[nw], 1e-12)
        return est

    # ------------------------------------------------------------------ large groups (shared x)
    def _group(self, q, ex):
        Cs, yc = self.Cs, self.yc
        Dx = self.Csx - ex
        nb = np.flatnonzero(np.all(np.abs(Dx) <= 1, axis=1))                # x-box, still sorted by y_1
        if len(nb) == 0:
            return                                                            # cnt = 0 -> est = 0
        Dx = Dx[nb]
        wpre = np.prod(0.75 * (1 - Dx ** 2), axis=1) * self.kfac
        Cy = Cs[nb][:, yc]
        Phi = np.ones((len(nb), self.p)); Phi[:, 1:] = self.C[nb][:, self.poly]          # [1, C_poly - c0]
        x0 = self.E[q[0]]
        for j in self.xpos:
            Phi[:, j] -= x0[self.poly[j - 1]]                                  # exact centring in x
        Lr = self.L[nb]
        Esq = self.Es[q]
        o1 = q[np.argsort(Esq[:, 0 + yc[0]], kind="stable")]
        for b0 in range(0, len(o1), self.band):
            qb = o1[b0:b0 + self.band]
            if len(yc) > 1:
                qb = qb[np.argsort(self.Es[qb, yc[1]], kind="stable")]
            e1 = self.Es[qb, yc[0]]
            lo = np.searchsorted(Cy[:, 0], e1.min() - 1 - 1e-9, "left")
            hi = np.searchsorted(Cy[:, 0], e1.max() + 1 + 1e-9, "right")
            for t0 in range(0, len(qb), self.tile):
                self._tile(qb[t0:t0 + self.tile], lo, hi, Cy, Phi, Lr, wpre)

    def _tile(self, qt, lo, hi, Cy, Phi, Lr, wpre):
        yc = self.yc
        Ey = self.Es[qt][:, yc]
        y0 = self.E[qt][:, yc].mean(0)
        for j, k in zip(self.ypos, self.ypos_dim):
            self.shift[qt, j] = self.E[qt, yc[k]] - y0[k]
        if hi <= lo:
            return
        r = np.arange(lo, hi)
        if len(yc) > 1:
            c = Cy[lo:hi, 1:]
            elo, ehi = Ey[:, 1:].min(0) - 1 - 1e-9, Ey[:, 1:].max(0) + 1 + 1e-9
            r = r[np.all((c >= elo) & (c <= ehi), axis=1)]
        if len(r) == 0:
            return
        if len(qt) * len(r) > self.max_elems and len(qt) > 1:
            h = len(qt) // 2
            self._tile(qt[:h], lo, hi, Cy, Phi, Lr, wpre); self._tile(qt[h:], lo, hi, Cy, Phi, Lr, wpre)
            return
        u1, ia = np.unique(Ey[:, 0], return_inverse=True)
        u2, ib = np.unique(Ey[:, 1], return_inverse=True) if len(yc) == 2 else (None, None)
        if len(yc) == 2 and len(u1) + len(u2) <= len(qt) // 2:
            # product-structured tile (grid): per-dimension factors on the distinct y_1, y_2 values only
            ia, ib = np.ravel(ia), np.ravel(ib)
            K1 = self._factor(Cy[r, 0], u1); K2 = self._factor(Cy[r, 1], u2)
            self.cnt[qt] = ((K1 >= 0).astype(float) @ (K2 >= 0).astype(float).T)[ia, ib].astype(np.int64)
            np.maximum(K1, 0.0, out=K1); np.maximum(K2, 0.0, out=K2)
            T = np.multiply(K1[ia], K2[ib])
        else:
            T = np.subtract(Cy[r, 0][None, :], Ey[:, 0][:, None])
            np.multiply(T, T, out=T); np.subtract(1.0, T, out=T)            # 1 - Dy_1^2 (>= 0 iff |Dy_1| <= 1)
            ind = T >= 0
            for k in range(1, len(yc)):
                U = np.subtract(Cy[r, k][None, :], Ey[:, k][:, None])
                np.multiply(U, U, out=U); np.subtract(1.0, U, out=U)
                ind &= U >= 0
                T *= U
            self.cnt[qt] = np.count_nonzero(ind, axis=1)
            T *= ind
        Ph = Phi[r]
        for j, k in zip(self.ypos, self.ypos_dim):
            Ph[:, j] -= y0[k]                                                  # centre the y-features at the tile
        p, nf = self.p, self.nf
        Pw = Ph * wpre[r][:, None]
        P = np.empty((len(r), nf + p)); c = 0
        for j in range(p):                                                     # upper-triangle order (triu_indices)
            np.multiply(Pw[:, j:j + 1], Ph[:, j:], out=P[:, c:c + p - j]); c += p - j
        np.multiply(Pw, Lr[r][:, None], out=P[:, nf:])
        Mt = T @ P
        self.M[qt] = Mt[:, :nf]; self.m[qt] = Mt[:, nf:]

    @staticmethod
    def _factor(c, u):
        """1 - (c - u)^2 for rows c and distinct query values u (>= 0 exactly when |c - u| <= 1)."""
        T = np.subtract(c[None, :], u[:, None]); np.multiply(T, T, out=T); np.subtract(1.0, T, out=T)
        return T

    # ------------------------------------------------------------------ small groups: padded per-query path
    def _padded(self, E, Es):
        p, cap = self.p, self.max_elems
        est = np.zeros(len(E)); cnt = np.zeros(len(E), np.int64)
        for s0 in range(0, len(E), 1024):
            idx = np.arange(s0, min(s0 + 1024, len(E)))
            nbl = self.tree.query_ball_point(Es[idx], r=1.0, p=np.inf)      # the original neighbour sets
            lens = np.array([len(v) for v in nbl], dtype=np.int64); cnt[idx] = lens
            o = np.argsort(lens, kind="stable"); i = 0
            while i < len(o):                                                # batches of similar length
                j = min(i + 256, len(o))
                while j > i + 1 and (j - i) * lens[o[j - 1]] * p > cap:
                    j = i + (j - i) // 2
                sub = o[i:j]; i = j
                est[idx[sub]] = self._padded_batch(E[idx[sub]], Es[idx[sub]], [nbl[k] for k in sub], lens[sub])
        return est, cnt

    def _padded_batch(self, E, Es, nbl, lens):
        p = self.p
        e = np.zeros(len(E))
        Lm = int(lens.max())
        if Lm == 0:
            return e
        NB = np.zeros((len(E), Lm), np.int64); Mk = np.arange(Lm)[None, :] < lens[:, None]
        NB[Mk] = np.concatenate([np.asarray(v, dtype=np.int64) for v in nbl if len(v)])
        D = self.Cs[NB] - Es[:, None, :]
        w = np.prod(0.75 * (1 - D ** 2), axis=2) * Mk
        F = np.ones((len(E), Lm, p))
        F[:, :, 1:] = self.C[NB][:, :, self.poly] - E[:, None, self.poly]
        Fw = F * w[:, :, None]
        G = np.matmul(Fw.transpose(0, 2, 1), F) + 1e-8 * np.eye(p)
        b = np.einsum("blp,bl->bp", Fw, self.L[NB])
        ll = lens >= 3 * p
        if ll.any():
            e[ll] = np.linalg.solve(G[ll], b[ll][:, :, None])[:, 0, 0]
        nw = (lens > 0) & ~ll
        e[nw] = b[nw, 0] / np.maximum(w[nw].sum(1), 1e-12)
        return e
