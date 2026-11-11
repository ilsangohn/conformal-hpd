"""Table 2 with rank calibration (Algorithm 1, alg:local) for d_Y in {2, 3}; grid-free computation and evaluation.

usage: python sim_rank.py N seed scen dY          (scen in gauss, smooth, hetero; dY in 2, 3)
       python sim_rank.py selftest                (unit tests: rank score, models, LP evaluator in d_Y = 3)
       python sim_rank.py mccheck scen dY M hp    (Monte Carlo error of the SNIS mass at u = 0.9, true density)

Sample sizes (environment): N (positional) = fitting sample of a one-stage method; N_REG (default N // 6) = auxiliary
(regression) sample; methods with an auxiliary step fit on the first N - N_REG points and use the last N_REG points;
NCAL (default 1000) = common calibration sample. Tuning: the last NVAL (default N // 6) points of the first N - N_REG
are held out from each fitting sample; every density estimate is tuned on them (bandwidths, partition, number of
epochs of the network) with the rest of its fitting sample, and then refitted on its whole fitting sample with the
selected tuning parameters (the same bandwidths, number of cubes and number of epochs).

Models. d_X = 3, X ~ U[0,1]^3, Y-box [-2.5, 2.5]^{d_Y}. For d_Y = 2 the samplers reproduce those of the
earlier two-dimensional experiments draw for draw (same rng calls); for d_Y = 3 each model gets a third coordinate:
  gauss : N(ctr(x), sd(x)^2 I_3), ctr(x) = (0.6 Phi(idx_1) - 0.3, 0, 0), sd(x) = 0.30 + 0.30 Phi(idx_2);
  smooth: 1/2 N(-ctr, sd^2 I_3) + 1/2 N(ctr, sd^2 I_3), ctr(x) = (0.6 + 0.6 Phi(idx_1), 0, 0), sd = 0.30 + 0.15 Phi(idx_2);
  hetero: (1 - wt) N((0, 0.3(x_2 - 0.5), 0), 0.5^2 I_3) + wt [1/2 N((-0.9,0,0), 0.18^2 I_3) + 1/2 N((0.9,0,0), 0.18^2 I_3)].
Fitting and calibration draws come from the truncated law by rejection in the gauss, moe and curve models (other models clip them to the box); evaluation uses the truncated
distribution (exact draws by rejection) and its density p / P_x(box), with P_x(box) in closed form.

One-stage HPD values (method computation): self-normalized importance sampling with M proposal points U_m shared by
all x, drawn from q = 0.9 * (Gaussian-smoothed marginal of 3000 responses of the first N - N_REG points, bandwidth MC_HP, truncated to the
box) + 0.1 * Unif(box) (density known exactly). HPD value of y at x = sum_m w_m 1{p(U_m|x) > p(y|x)} / sum_m w_m,
w_m = p(U_m|x) / q(U_m). The HPD values of calibration and auxiliary points use the first MC_M_CAL (default M / 4)
proposal points; the evaluation covariates use all M.

"-LOC" (ablation): the same regression with the HPD values s(X_i, Y_i) of the regression points at their own
covariates (sets {s <= u}), as in the HPD-level recalibration of Dey et al. (2025, App. D.2), then split conformal.

Rank calibration ("-RANK"): local linear weights w_i(x) (triweight product kernel, equivalent-kernel weights) over
D_reg, a_i = a(Y_i | x) at x, Qbar_x = clip_[0,1] of the running maximum of the cumulative weights over tie groups in
decreasing order, score Qbar_x(a(y|x)), split conformal on the calibration sample. Coordinatewise regression
bandwidths by frozen-target cross-validation (frozen_cv_vec): 5 folds over D_reg, product grid H_AXES^d_X; for a
held-out point j the candidate sets are frozen at X_j (a_i = a(Y_i | X_j) for the other folds, a_j = a(Y_j | X_j)),
and the loss is the integrated Brier score mean_j int (1{a_j >= t} - Qbar_j^h(t))^2 w_j(dt), w_j uniform between the
levels at which the pilot fit (h = 1.0 in every coordinate) has coverage 0.97 and 0.80, computed exactly over the step
functions. "-RANKFIX": the same procedure with the fixed bandwidth (log N_REG / N_REG)^(1/(2 beta + d_X)), beta = 2;
"-RANKIND": the isotropic bandwidth of ind_cv (5-fold CV of the regression of Z_i = 1{a(Y_i|X_i) >= q}, q the
0.1-quantile of a(Y_i|X_i), on the grid H_GRID); both are reference rows.
Bandwidth grids (selection losses are recorded in the output lines): LP hx in {0.6, 0.9, 1.35, 2.0}, hy in
{0.2, 0.3, 0.45, 0.65} (tuned directly; the same bandwidths in the refit); regression H_GRID = (0.22, 0.3, 0.4, 0.55,
0.75, 1.0, 1.5).
OT-CP is run for d_Y = 2 only.

Evaluation (common random numbers across methods): at the 20 evaluation covariates (10 hard, 10 easy) and 100
marginal covariates of sim_eq.py, L exact draws Y_l ~ P_x (truncated) and L uniform draws on the box (shared by all
x) form an importance sample from q_x = 1/2 P_x + 1/2 Unif(box); L = L_TRUE (20 covariates), L_MARG (100 covariates).
Coverage = mean over the exact draws; Leb(A) = mean over the pooled sample of 1_A / q_x; oracle level t_c =
(1 - c)-quantile of p(Y_l); shape S = (1/t_c) int_{C delta C*_c} |p - t_c| by the same importance sample.
Output keys as sim_eq.py (marg, cerrR, cerrO, vol, volM, sym, shape, shape_ld) plus dY, sizes, tau, time_s
(method computation, fitting excluded), cal_s (calibration scores only), fit_s (fitting of the base model).
"""
import itertools, json, math, os, sys, time
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gammaln, i0e, ndtr
from scipy.stats import t as student_t

from sim_contrastive import BOX, bce
from sim_metric import directions


# ====================================================================== helpers of the models and baselines
def sig(u):
    return 1 / (1 + np.exp(-u))


_HW, _HC = float(os.environ.get("HET_WIDTH", "0.15")), float(os.environ.get("HET_CENTER", "0.5"))   # transition in x1 of the hetero weight


def _w(x1):
    """Weight of the two-component part of the hetero model and of the manifold (curve) model."""
    return sig((x1 - _HC) / _HW)


def fit_mean_net(X, Y, seed):
    """Mean network of the ELLIPSOID baseline (2 x 64 SiLU, Adam 2e-3, 60 epochs, batches of 256)."""
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

K, ALPHA = 5, 0.1
GAMMA = 1 - ALPHA
DX = 3
EPS_DEF = 0.1                                   # defensive uniform component of the proposal
LP_HX, LP_HY = (0.6, 0.9, 1.35, 2.0), (0.2, 0.3, 0.45, 0.65)    # LP bandwidth grids (covariate, response)
STAGE = defaultdict(float)                      # wall time per stage (printed to stderr at the end)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Timer:
    """with Timer("stage") as t: ...  -> t.dt, and STAGE["stage"] += t.dt"""
    def __init__(self, stage=None):
        self.stage = stage
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        if self.stage:
            STAGE[self.stage] += self.dt


# ====================================================================== models
MS_S0, MS_LOG = float(os.environ.get("MS_S0", "0.1")), float(os.environ.get("MS_LOG", "1.0"))   # mscale: sd = MS_S0 10^(MS_LOG x1)
MF_DELTA, MF_R0, MF_R1 = (float(v) for v in (os.environ.get("MF_DELTA", "0.01"), os.environ.get("MF_R0", "0.8"),
                                             os.environ.get("MF_R1", "0.4")))         # manifold: shell sd, radius
TM_DF, TM_SCALE, TM_W = (float(v) for v in (os.environ.get("TM_DF", "3"), os.environ.get("TM_SCALE", "0.5"),
                                            os.environ.get("TM_W", "1.0")))         # tailmix: tail df, tail scale, max weight
RR_THIN, RR_WIDE, RR_OMEGA = (float(v) for v in (os.environ.get("RR_THIN", "0.05"), os.environ.get("RR_WIDE", "0.4"),
                                                  os.environ.get("RR_OMEGA", "0.6")))   # rotell: thin / wide sd, angle per x1
PL_R, PL_B0, PL_B1 = (float(v) for v in (os.environ.get("PL_R", "0.8"), os.environ.get("PL_B0", "2"),
                                         os.environ.get("PL_B1", "10")))       # plateau: radius, shape at x1 = 0 and 1
PL_SG = float(os.environ.get("PL_SG", "0.4"))     # platmix: sd of the Gaussian component
CV_D0, CV_D1 = float(os.environ.get("CV_D0", "0.02")), float(os.environ.get("CV_D1", "0.02"))  # curve: noise sd at x1 = 0, 1
CV_PURE = os.environ.get("CV_PURE") == "1"        # curve: the tube alone (no broad component)
SDB = float(os.environ.get("SDB", "0.18"))         # sd of the two concentrated components of the hetero model
REF = os.environ.get("REF", "unif")               # contrastive reference: unif (box) or gmix (Gaussian + uniform)
REF_OBJ = None                                    # the gmix reference of the current run (set in run)
ISO_ONLY = os.environ.get("ISO_ONLY") == "1"      # isotropic tuning only: frozen CV and ind_cv over the diagonal grid


class Model:
    """Mixture of isotropic Gaussians in d_Y dimensions with covariate-dependent weights, means and sds."""

    def __init__(self, scen, dY):
        self.scen, self.dY = scen, dY
        self.W = directions(3)
        self.inR = (lambda x: x[:, 0] > 0.72) if scen == "hetero" else (lambda x: x[:, 0] > 0.5)
        self.Q = 1.0 / (2 * BOX) ** dY
        if scen in ("noisedim", "tailmix", "rotell", "plateau", "platmix", "curve"):
            self.inR = lambda x: x[:, 0] > 0.72          # the side where the mixture weight w(x) is large
        # plateau: generalized normal exp(-(|y| / PL_R)^beta(x)) with beta(x) = PL_B0 + (PL_B1 - PL_B0) x_1, Gaussian at
        # beta = 2 and close to the uniform ball for large beta (flat top, steep edge); R = the flat side x_1 > 0.72
        # platmix: (1 - w(x)) N(0, PL_SG^2 I) + w(x) exp(-(|y| / PL_R)^PL_B1) with w the hetero transition: the covariates
        # move only the weight of the flat-topped component; R = x_1 > 0.72
        # curve: (1 - w(x)) N(0, 0.5^2 I) + w(x) [uniform on the circle of radius 0.8 in (y_1, y_2) + N(0, delta(x)^2 I)],
        # delta(x) = CV_D0 (CV_D1 / CV_D0)^{x_1}, w the hetero transition (w = 1 with CV_PURE); a tube of codimension
        # d_Y - 1 around a fixed curve. R = x_1 > 0.72
        if scen == "mscale":
            # multi-scale heteroscedasticity: Gaussian with mean (0, 0.3 (x_2 - 1/2), 0, ...) and sd MS_S0 * 10^(MS_LOG x_1),
            # so the sd varies by 10^MS_LOG over the covariate; R = the sharp (small-sd) side x_1 < 0.3
            self.inR = lambda x: x[:, 0] < 0.3
        if scen == "manifold":
            # thin shell: with probability w(x) (the hetero transition) |Y| ~ N(r(x), MF_DELTA^2) and Y/|Y| uniform on the
            # sphere, otherwise N(0, 0.6^2 I); r(x) = MF_R0 + MF_R1 x_2. R = the shell side x_1 > 0.72
            self.inR = lambda x: x[:, 0] > 0.72
        if scen == "moe":
            # latent classes: MOE_K isotropic Gaussian components at fixed means (uniform on [-1.6, 1.6]^d_Y, pairwise
            # distance >= 1.2) with sds geometric over MOE_S (randomly assigned); the covariates move only the class
            # probabilities, softmax(MOE_G B (x - 1/2)), B with N(0, 1) entries. R: the sharper half carries > 1/2
            # (x_1 > 1/2 when MOE_G = 0).
            K_, (s0, s1), G = int(os.environ.get("MOE_K", "8")), map(float, os.environ.get("MOE_S", "0.1,0.4").split(",")), \
                float(os.environ.get("MOE_G", "3"))
            r = np.random.default_rng(2026 + dY); mk = []
            while len(mk) < K_:
                c = r.uniform(-1.6, 1.6, size=dY)
                if all(np.linalg.norm(c - m) >= 1.2 for m in mk):
                    mk.append(c)
            self.mk = np.array(mk); self.sk = r.permutation(np.geomspace(s0, s1, K_)); self.Bg = G * r.standard_normal((K_, 3))
            sharp = self.sk <= np.median(self.sk)
            self.inR = lambda x: self.comps(x)[0][:, sharp].sum(1) > 0.5
            if G == 0:                                   # no covariate effect, so no sharp-dominated side: x_1 > 1/2
                self.inR = lambda x: x[:, 0] > 0.5

    def comps(self, X):
        """weights (n, J), means (n, J, dY), sds (n, J)"""
        n, d = len(X), self.dY
        if self.scen == "mscale":
            mu = np.zeros((n, 1, d)); mu[:, 0, 1 % d] = 0.3 * (X[:, 1] - 0.5)
            return np.ones((n, 1)), mu, (MS_S0 * 10.0 ** (MS_LOG * X[:, 0]))[:, None]
        if self.scen == "moe":
            L = (X - 0.5) @ self.Bg.T; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            return P, np.broadcast_to(self.mk, (n,) + self.mk.shape), np.broadcast_to(self.sk, (n, len(self.sk)))
        if self.scen == "gauss":
            import sim_gauss
            m, s = sim_gauss.params(X, self.W)
            mu = np.zeros((n, 1, d)); mu[:, 0, 0] = m
            return np.ones((n, 1)), mu, s[:, None]
        if self.scen == "smooth":
            import sim_metric
            a, s = sim_metric.params(X, self.W)
            mu = np.zeros((n, 2, d)); mu[:, 0, 0] = -a; mu[:, 1, 0] = a
            return np.full((n, 2), 0.5), mu, np.stack([s, s], 1)
        w = _w(X[:, 0])
        mu = np.zeros((n, 3, d)); mu[:, 0, 1] = 0.3 * (X[:, 1] - 0.5); mu[:, 1, 0] = -0.9; mu[:, 2, 0] = 0.9
        return np.stack([1 - w, w / 2, w / 2], 1), mu, np.tile([0.5, SDB, SDB], (n, 1))

    def draw_y(self, X, rng):
        """Y | X with the rng calls of the earlier two-dimensional samplers (identical draws for d_Y = 2); not clipped."""
        n, d = len(X), self.dY
        if self.scen == "plateau":                   # (|Y| / PL_R)^beta ~ Gamma(d / beta), direction uniform
            b = self.pl_beta(X); u = rng.standard_normal((n, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            return (PL_R * rng.gamma(d / b) ** (1 / b))[:, None] * u
        if self.scen == "curve":
            isT = rng.uniform(size=n) < self.cv_w(X); th = rng.uniform(0, 2 * np.pi, size=n)
            c = np.zeros((n, d)); c[:, 0] = 0.8 * np.cos(th); c[:, 1] = 0.8 * np.sin(th)
            T = c + self.cv_delta(X)[:, None] * rng.standard_normal((n, d))
            return np.where(isT[:, None], T, 0.5 * rng.standard_normal((n, d)))
        if self.scen == "platmix":
            isP = rng.uniform(size=n) < self.shell_w(X); u = rng.standard_normal((n, d))
            g = PL_SG * rng.standard_normal((n, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            return np.where(isP[:, None], (PL_R * rng.gamma(d / PL_B1, size=n) ** (1 / PL_B1))[:, None] * u, g)
        if self.scen == "rotell":                    # Y = RR_WIDE z + (RR_THIN - RR_WIDE) (e . z) e, z ~ N(0, I)
            z = rng.standard_normal((n, d)); e = self.rot_e(X)
            return RR_WIDE * z + (RR_THIN - RR_WIDE) * (e * z).sum(1)[:, None] * e
        if self.scen == "noisedim":
            isB = rng.uniform(size=n) < self.shell_w(X); comp = rng.choice([-1, 1], size=n)
            muA = np.zeros((n, 2)); muA[:, 1] = 0.3 * (X[:, 1] - 0.5)
            muB = np.zeros((n, 2)); muB[:, 0] = 0.9 * comp
            info = np.where(isB[:, None], muB + SDB * rng.standard_normal((n, 2)), muA + 0.5 * rng.standard_normal((n, 2)))
            return np.hstack([info, rng.standard_normal((n, d - 2))])
        if self.scen == "tailmix":
            isT = rng.uniform(size=n) < TM_W * self.shell_w(X)
            return np.where(isT[:, None], TM_SCALE * rng.standard_t(TM_DF, size=(n, d)), rng.standard_normal((n, d)))
        if self.scen == "manifold":
            isS = rng.uniform(size=n) < self.shell_w(X)
            rho = self.shell_r(X) + MF_DELTA * rng.standard_normal(n)
            u = rng.standard_normal((n, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            return np.where(isS[:, None], rho[:, None] * u, 0.6 * rng.standard_normal((n, d)))
        if self.scen == "moe":
            pi, mu, s = self.comps(X)
            k = (rng.uniform(size=(n, 1)) > np.cumsum(pi, 1)).sum(1)
            return self.mk[k] + self.sk[k][:, None] * rng.standard_normal((n, d))
        if self.scen in ("gauss", "mscale"):
            pi, mu, s = self.comps(X)
            return mu[:, 0] + s[:, 0:1] * rng.standard_normal((n, d))
        if self.scen == "smooth":
            pi, mu, s = self.comps(X)
            comp = rng.choice([-1, 1], size=n)
            ctr = np.zeros((n, d)); ctr[:, 0] = comp * mu[:, 1, 0]
            return ctr + s[:, 0:1] * rng.standard_normal((n, d))
        w = _w(X[:, 0])
        isB = rng.uniform(size=n) < w
        comp = rng.choice([-1, 1], size=n)
        muA = np.zeros((n, d)); muA[:, 1] = 0.3 * (X[:, 1] - 0.5)
        YA = muA + 0.5 * rng.standard_normal((n, d))
        muB = np.zeros((n, d)); muB[:, 0] = 0.9 * comp
        YB = muB + SDB * rng.standard_normal((n, d))
        return np.where(isB[:, None], YB, YA)

    def sample(self, n, rng):
        X = rng.uniform(size=(n, 3))
        if self.scen not in ("noisedim", "tailmix", "plateau", "platmix", "curve", "moe", "gauss"):   # moe: rejection since 10-31, gauss since 11-03
            return X, np.clip(self.draw_y(X, rng), -BOX + 1e-6, BOX - 1e-6)
        Y = self.draw_y(X, rng); bad = np.flatnonzero(np.any(np.abs(Y) > BOX, axis=1))
        while len(bad):          # rejection: given x, the law is the density truncated to the box (no clipping atoms)
            Y[bad] = self.draw_y(X[bad], rng); bad = bad[np.any(np.abs(Y[bad]) > BOX, axis=1)]
        return X, Y

    def _info2(self, X, Y2):
        """informative two-dimensional part of noisedim: the hetero mixture (fixed locations, weight w(x))"""
        w = self.shell_w(X); mA = np.zeros((len(X), 2)); mA[:, 1] = 0.3 * (X[:, 1] - 0.5)
        g = lambda mu, s: np.exp(-0.5 * ((Y2 - mu) ** 2).sum(1) / s ** 2) / (2 * np.pi * s ** 2)
        B = 0.5 * g(np.array([-0.9, 0.0]), SDB) + 0.5 * g(np.array([0.9, 0.0]), SDB)
        return (1 - w) * g(mA, 0.5) + w * B

    def _box2(self, X):
        """box mass of the informative two-dimensional part of noisedim"""
        w = self.shell_w(X); sd = SDB
        cdf = lambda mu, s: ndtr((BOX - mu) / s) - ndtr((-BOX - mu) / s)
        mA = np.zeros((len(X), 2)); mA[:, 1] = 0.3 * (X[:, 1] - 0.5)
        A = cdf(mA[:, 0], 0.5) * cdf(mA[:, 1], 0.5)
        B = 0.5 * cdf(-0.9, sd) * cdf(0.0, sd) + 0.5 * cdf(0.9, sd) * cdf(0.0, sd)
        return (1 - w) * A + w * B

    def shell_w(self, X):
        return _w(X[:, 0])

    def shell_r(self, X):
        return MF_R0 + MF_R1 * X[:, 1]

    def cv_w(self, X):
        return np.ones(len(X)) if CV_PURE else self.shell_w(X)

    def cv_delta(self, X):
        return CV_D0 * (CV_D1 / CV_D0) ** X[:, 0]

    def pl_beta(self, X):
        return PL_B0 + (PL_B1 - PL_B0) * X[:, 0]

    def rot_e(self, X):
        """unit thin direction e(x) = (cos th, sin th, 0, ...), th = RR_OMEGA (x1 - 1/2)"""
        th = RR_OMEGA * (X[:, 0] - 0.5); e = np.zeros((len(X), self.dY)); e[:, 0] = np.cos(th); e[:, 1] = np.sin(th)
        return e

    def density(self, X, Y):
        """untruncated conditional density, rows paired"""
        if self.scen == "curve":                     # circle * N(0, delta^2 I_2): exp(-(rho^2 + 0.8^2) / 2 delta^2) I_0(0.8 rho / delta^2)
            d = self.dY; w = self.cv_w(X); dl = self.cv_delta(X); rho = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
            tube = np.exp(-0.5 * ((rho - 0.8) / dl) ** 2) * i0e(0.8 * rho / dl ** 2) / (2 * np.pi * dl ** 2)
            tube = tube * np.exp(-0.5 * (Y[:, 2:] ** 2).sum(1) / dl ** 2) / (2 * np.pi * dl ** 2) ** ((d - 2) / 2)
            return (1 - w) * np.exp(-0.5 * (Y ** 2).sum(1) / 0.25) / (2 * np.pi * 0.25) ** (d / 2) + w * tube
        if self.scen in ("plateau", "platmix"):      # normalizing constant A_d PL_R^d Gamma(d / beta) / beta
            d = self.dY; rho = np.sqrt((Y ** 2).sum(1)); b = self.pl_beta(X) if self.scen == "plateau" else PL_B1
            lz = math.log(2) + 0.5 * d * math.log(np.pi) - gammaln(d / 2) + d * math.log(PL_R) + gammaln(d / b) - np.log(b)
            flat = np.exp(-(rho / PL_R) ** b - lz)
            if self.scen == "plateau":
                return flat
            w = self.shell_w(X)
            return (1 - w) * np.exp(-0.5 * (Y ** 2).sum(1) / PL_SG ** 2) / (2 * np.pi * PL_SG ** 2) ** (d / 2) + w * flat
        if self.scen == "rotell":                    # N(0, RR_WIDE^2 I + (RR_THIN^2 - RR_WIDE^2) e e^T)
            e = self.rot_e(X); q = (Y * e).sum(1); d = self.dY
            quad = (Y ** 2).sum(1) / RR_WIDE ** 2 + (1 / RR_THIN ** 2 - 1 / RR_WIDE ** 2) * q ** 2
            return np.exp(-0.5 * quad) / (2 * np.pi) ** (d / 2) / (RR_THIN * RR_WIDE ** (d - 1))
        if self.scen == "noisedim":                  # informative (y_1, y_2) times independent N(0, 1) noise coordinates
            return self._info2(X, Y[:, :2]) * np.prod(np.exp(-0.5 * Y[:, 2:] ** 2) / np.sqrt(2 * np.pi), axis=1)
        if self.scen == "tailmix":                   # (1 - w) N(0, I) + w prod_j t_df(y_j / TM_SCALE) / TM_SCALE
            w = TM_W * self.shell_w(X)
            core = np.exp(-0.5 * (Y ** 2).sum(1)) / (2 * np.pi) ** (self.dY / 2)
            tail = np.prod(student_t.pdf(Y / TM_SCALE, TM_DF) / TM_SCALE, axis=1)
            return (1 - w) * core + w * tail
        if self.scen == "manifold":
            w = self.shell_w(X); d = self.dY; rho = np.sqrt((Y ** 2).sum(1)); r = self.shell_r(X)
            broad = np.exp(-0.5 * (Y ** 2).sum(1) / 0.36) / (2 * np.pi * 0.36) ** (d / 2)
            Ad = 2 * np.pi ** (d / 2) / math.gamma(d / 2)                       # surface area of the unit sphere in R^d
            shell = np.exp(-0.5 * ((rho - r) / MF_DELTA) ** 2) / (np.sqrt(2 * np.pi) * MF_DELTA) / (Ad * np.maximum(rho, 1e-12) ** (d - 1))
            return (1 - w) * broad + w * shell
        pi, mu, s = self.comps(X)
        d2 = ((Y[:, None, :] - mu) ** 2).sum(-1)
        return (pi * np.exp(-d2 / (2 * s ** 2)) / (2 * np.pi * s ** 2) ** (self.dY / 2)).sum(1)

    def box_mass(self, X):
        if self.scen == "plateau":                   # beta >= 2 and PL_R = 0.8: the box holds all but < 1e-4
            return np.ones(len(X))
        if self.scen == "curve":                     # the tube lies inside the box (0.8 + 6 delta < 2.5)
            w = self.cv_w(X)
            return (1 - w) * (2 * ndtr(BOX / 0.5) - 1) ** self.dY + w
        if self.scen == "platmix":                   # the flat-topped component lies inside the box
            w = self.shell_w(X)
            return (1 - w) * (2 * ndtr(BOX / PL_SG) - 1) ** self.dY + w
        if self.scen == "rotell":                    # every coordinate has sd <= RR_WIDE: the box holds all but ~1e-9
            return np.ones(len(X))
        if self.scen == "noisedim":
            return self._box2(X) * (ndtr(BOX) - ndtr(-BOX)) ** (self.dY - 2)
        if self.scen == "tailmix":
            w = TM_W * self.shell_w(X)
            mc = (ndtr(BOX) - ndtr(-BOX)) ** self.dY
            mt = (student_t.cdf(BOX / TM_SCALE, TM_DF) - student_t.cdf(-BOX / TM_SCALE, TM_DF)) ** self.dY
            return (1 - w) * mc + w * mt
        if self.scen == "manifold":
            w = self.shell_w(X)
            return (1 - w) * (2 * ndtr(BOX / 0.6) - 1) ** self.dY + w              # the shell lies inside the box
        pi, mu, s = self.comps(X)
        s = s[:, :, None]
        return (pi * np.prod(ndtr((BOX - mu) / s) - ndtr((-BOX - mu) / s), axis=2)).sum(1)

    def tdensity(self, X, Y):
        """density of the truncated distribution (restricted to the box and renormalized)"""
        return self.density(X, Y) / self.box_mass(X)

    def draw_truncated(self, x, L, rng):
        out, have = [], 0
        while have < L:
            Yd = self.draw_y(np.repeat(x[None], int(1.05 * (L - have)) + 16, 0), rng)
            Yd = Yd[np.all(np.abs(Yd) <= BOX, axis=1)]
            out.append(Yd); have += len(Yd)
        return np.vstack(out)[:L]


class Ref:
    """gmix contrastive reference q_ref = 0.9 N_box(m, diag(s^2)) + 0.1 Unif(box), m and s = coordinatewise mean and
    1.25 x sd of the responses Yfit, N_box the Gaussian truncated to the box (as sim_dyscale.Ref)."""

    def __init__(self, Yfit, eps=0.1, infl=1.25):
        self.dY = Yfit.shape[1]; self.eps = eps; self.Q = 1.0 / (2 * BOX) ** self.dY
        self.m = Yfit.mean(0); self.s = infl * Yfit.std(0)
        self.Z = float(np.prod(ndtr((BOX - self.m) / self.s) - ndtr((-BOX - self.m) / self.s)))
        self.lnorm = -0.5 * self.dY * np.log(2 * np.pi) - np.log(self.s).sum() - np.log(self.Z)

    def density(self, Y):
        out = np.empty(len(Y))
        for s0 in range(0, len(Y), 500000):
            y = Y[s0:s0 + 500000]
            out[s0:s0 + 500000] = (1 - self.eps) * np.exp(self.lnorm - 0.5 * (((y - self.m) / self.s) ** 2).sum(1)) + self.eps * self.Q
        return out

    def draw(self, n, rng):
        uni = rng.uniform(size=n) < self.eps
        Y = self.m + self.s * rng.standard_normal((n, self.dY)); bad = np.flatnonzero(np.any(np.abs(Y) > BOX, axis=1))
        while len(bad):
            Y[bad] = self.m + self.s * rng.standard_normal((len(bad), self.dY)); bad = bad[np.any(np.abs(Y[bad]) > BOX, axis=1)]
        Y[uni] = rng.uniform(-BOX, BOX, size=(int(uni.sum()), self.dY))
        return Y


def crows(Xf, Yf, rng):
    """contrastive rows with K references per fitting point: uniform on the box (= sim_contrastive.contrastive_rows
    for d_Y = 2) or, with REF=gmix, drawn from REF_OBJ"""
    n, d = Yf.shape
    Yneg = rng.uniform(-BOX, BOX, size=(n * K, d)) if REF_OBJ is None else REF_OBJ.draw(n * K, rng)
    Z = np.vstack([np.hstack([Xf, Yf]), np.hstack([np.repeat(Xf, K, 0), Yneg])])
    return Z, np.concatenate([np.ones(n), np.zeros(n * K)])


def eta2dens(eta, dY, Y=None, cap=0.995):
    """q_ref(y) K eta / (1 - eta), eta clipped to [0, cap]; q_ref uniform on the box unless REF=gmix"""
    eta = np.clip(np.asarray(eta, float), 0.0, cap)
    q = (1.0 / (2 * BOX) ** dY) if REF_OBJ is None else REF_OBJ.density(Y)
    return K * q * eta / (1 - eta)


# ====================================================================== proposal (shared by all x)
class Proposal:
    def __init__(self, Yfit, M, hp, seed, dY):
        """hp: one smoothing bandwidth or a tuple of bandwidths, each draw using one of them uniformly at random"""
        pr = np.random.default_rng(999 + seed)
        self.C = Yfit[pr.choice(len(Yfit), size=min(len(Yfit), 3000), replace=False)]
        self.hps = tuple(hp) if isinstance(hp, (tuple, list)) else (hp,); self.dY = dY
        H = np.array(self.hps); out, have = [], 0
        while have < M:
            m = 2 * (M - have) + 64
            uni = pr.uniform(size=m) < EPS_DEF
            hb = H[pr.integers(len(H), size=m)][:, None] if len(H) > 1 else H[0]
            U = self.C[pr.integers(len(self.C), size=m)] + hb * pr.standard_normal((m, dY))
            U[uni] = pr.uniform(-BOX, BOX, size=(int(uni.sum()), dY))
            U = U[np.all(np.abs(U) <= BOX, axis=1)]
            out.append(U); have += len(U)
        # the rejection step is applied to each mixture draw: the accepted Gaussian part has density g / Z_g
        self.U = np.vstack(out)[:M]
        self.Zg = float(np.mean([np.mean(np.prod(ndtr((BOX - self.C) / h) - ndtr((-BOX - self.C) / h), axis=1)) for h in H]))
        self.q = self.density(self.U)
        self.wU = 1.0 / self.q

    def density(self, Uq):
        """exact density of the accepted draws; the acceptance probability of a draw is
        a = (1 - eps) Z_g + eps, and q = [(1 - eps) g + eps / vol] / a."""
        g = np.zeros(len(Uq))
        for s0 in range(0, len(Uq), 2000):
            u = Uq[s0:s0 + 2000]
            d2 = np.maximum((u ** 2).sum(1)[:, None] + (self.C ** 2).sum(1)[None] - 2 * u @ self.C.T, 0)
            for h in self.hps:
                g[s0:s0 + 2000] += np.exp(-d2 / (2 * h ** 2)).mean(1) / (2 * np.pi * h ** 2) ** (self.dY / 2) / len(self.hps)
        a = (1 - EPS_DEF) * self.Zg + EPS_DEF
        return ((1 - EPS_DEF) * g + EPS_DEF / (2 * BOX) ** self.dY) / a


# ====================================================================== contrastive network
def fit_net_tuned(Ztr, Ltr, Zv, Lv, Zall, Lall, seed, width=128, depth=3, max_epochs=200, patience=8):
    """Contrastive network (3 x 128 SiLU, Adam 2e-3, batches of 1024): the number of epochs is
    the epoch with the smallest held-out loss on (Zv, Lv) when training on (Ztr, Ltr) (early stopping, patience 8);
    the network is then retrained from the same initialization on all rows (Zall, Lall) for that many epochs.
    Returns (eta, number of epochs)."""
    import torch

    def train(Z, L, epochs, Zh=None, Lh=None):
        torch.manual_seed(seed); torch.set_num_threads(1)
        mu, sd = Z.mean(0), Z.std(0)
        f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
        Xt, Yt = f(Z), torch.tensor(L, dtype=torch.float32)
        layers, dim = [], Z.shape[1]
        for _ in range(depth):
            layers += [torch.nn.Linear(dim, width), torch.nn.SiLU()]; dim = width
        net = torch.nn.Sequential(*layers, torch.nn.Linear(dim, 1))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = torch.nn.BCEWithLogitsLoss()
        if Zh is not None:
            Xh, Yh = f(Zh), torch.tensor(Lh, dtype=torch.float32)
        best, best_ep, bad = np.inf, 1, 0
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 1024):
                idx = perm[b:b + 1024]
                opt.zero_grad(); lossf(net(Xt[idx]).squeeze(1), Yt[idx]).backward(); opt.step()
            if Zh is not None:
                with torch.no_grad():
                    v = lossf(net(Xh).squeeze(1), Yh).item()
                if v < best - 1e-5:
                    best, best_ep, bad = v, ep + 1, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        return net, f, best_ep

    _, _, n_ep = train(Ztr, Ltr, max_epochs, Zv, Lv)
    net, f, _ = train(Zall, Lall, n_ep)
    return (lambda A: torch.sigmoid(net(f(A)).squeeze(1)).detach().numpy()), n_ep


# ====================================================================== rank calibration
def rank_table(a_aux, w_aux):
    """(u, R): distinct values u (ascending) of a_aux and R[j] = clip_[0,1](max_{0<=j'<=j} S_j'), S_j the cumulative
    weight of the top j tie groups (S_0 = 0). Qbar(t) = R[#{groups with value >= t}]."""
    u, inv = np.unique(a_aux, return_inverse=True)
    gw = np.bincount(np.ravel(inv), weights=w_aux, minlength=len(u))
    S = np.concatenate([[0.0], np.cumsum(gw[::-1])])
    return u, np.clip(np.maximum.accumulate(S), 0.0, 1.0)


def rank_eval(u, R, t):
    return R[len(u) - np.searchsorted(u, t, side="left")]


def rank_bruteforce(a_aux, w_aux, t):
    """Qbar(t) = clip(sup_{v >= t} sum_i w_i 1{a_i >= v}); sup over v in {t} U {a_i >= t} U {+inf}"""
    out = np.empty(len(t))
    for k, tk in enumerate(t):
        cand = [0.0, float(np.sum(w_aux[a_aux >= tk]))]
        cand += [float(np.sum(w_aux[a_aux >= v])) for v in a_aux[a_aux >= tk]]
        out[k] = min(1.0, max(0.0, max(cand)))
    return out


def ll_weights_batch(XB, T, h, min_pts=2 * (DX + 1)):
    """Equivalent-kernel weights of local linear regression (triweight product kernel) at the rows of T:
    (nT, nB) array, zero outside the window and for windows with fewer than min_pts points."""
    D = (XB[None, :, :] - T[:, None, :]) / h
    inside = np.all(np.abs(D) < 1, axis=2)
    k = np.prod(np.clip(1 - D ** 2, 0, None) ** 3, axis=2) * inside
    F = np.concatenate([np.ones(D.shape[:2] + (1,)), D], axis=2)
    G = np.einsum("tn,tna,tnb->tab", k, F, F) + 1e-8 * np.eye(DX + 1)
    e1 = np.zeros((len(T), DX + 1, 1)); e1[:, 0, 0] = 1.0
    a = np.linalg.solve(G, e1)[:, :, 0]
    w = k * np.einsum("tna,ta->tn", F, a)
    w[inside.sum(1) < min_pts] = 0.0
    return w


def ll_reg(Xtr, Ztr, Xq, hx):
    """local linear regression (triweight product kernel) of Ztr on Xtr, evaluated at Xq (as sim_eq.py)"""
    outv = np.empty(len(Xq))
    for s0 in range(0, len(Xq), 200):
        D = (Xtr[None, :, :] - Xq[s0:s0 + 200, None, :]) / hx
        w = np.prod(np.clip(1 - D ** 2, 0, None) ** 3, axis=2)
        F = np.concatenate([np.ones(D.shape[:2] + (1,)), D], axis=2)
        Gm = np.einsum("qn,qna,qnb->qab", w, F, F) + 1e-8 * np.eye(F.shape[2])
        r = np.einsum("qn,qna,n->qa", w, F, Ztr)
        outv[s0:s0 + 200] = np.linalg.solve(Gm, r[:, :, None])[:, 0, 0]
    return outv


def cv_bw(Xtr, Ztr, seed, grid=(0.15, 0.22, 0.3, 0.4, 0.55, 0.75, 1.0)):
    """five-fold CV of the regression bandwidth (as cv_bw of sim_eq.py: at most 300 held-out points per fold,
    predictions clipped to [0, 1], squared error)"""
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


def rank_cv_bw(a_own, XB, seed):
    """regression bandwidth of rank calibration: CV of the regression of Z_i = 1{a(Y_i|X_i) >= q} on X_i, with q the
    empirical 0.1-quantile of a(Y_i|X_i) over the auxiliary points"""
    qz = np.quantile(a_own, ALPHA)
    return cv_bw(XB, (a_own >= qz).astype(float), seed)


H_GRID = (0.22, 0.3, 0.4, 0.55, 0.75, 1.0, 1.5)    # regression bandwidths of the frozen-target CV
H_PILOT = 1.0                                     # pilot bandwidth that fixes the levels of the CV loss
CV_LEV = (0.97, 0.80)                             # coverages of the pilot fit at the ends of the level range


def level_at(u, R, c):
    """inf{t: Qbar(t) <= c} for the table (u, R) of rank_table; -inf if Qbar <= c everywhere. Qbar equals R[G] on
    (-inf, u_1], R[G - g + 1] on (u_{g-1}, u_g] and R[0] = 0 on (u_G, inf)."""
    k = int(np.argmax(R[::-1] <= c))                 # first interval (0-based) with value <= c; exists since R[0] = 0
    return -np.inf if k == 0 else float(u[k - 1])


def frozen_point_loss(a_tr, W, a_j, pilot, lev=CV_LEV):
    """Integrated Brier losses of the frozen-target CV at one held-out point, for all bandwidths at once.
    a_tr: values a(Y_i | X_j) of the training points; W: (n_h, n_tr) local linear weights at X_j; a_j = a(Y_j | X_j);
    pilot: row of W of the pilot bandwidth. Returns (losses, (t_lo, t_hi)) or (None, None) if the level range is empty.
    Loss_h = (t_hi - t_lo)^{-1} int_{t_lo}^{t_hi} (1{a_j >= t} - Qbar^h(t))^2 dt, exact for the step functions."""
    if len(a_tr) == 0:
        return None, None
    u, inv = np.unique(a_tr, return_inverse=True); inv = np.ravel(inv); G = len(u)
    gw = np.stack([np.bincount(inv, weights=w, minlength=G) for w in W])
    S = np.concatenate([np.zeros((len(W), 1)), np.cumsum(gw[:, ::-1], axis=1)], axis=1)
    R = np.clip(np.maximum.accumulate(S, axis=1), 0.0, 1.0)
    tl, th = level_at(u, R[pilot], lev[0]), level_at(u, R[pilot], lev[1])
    if tl == -np.inf:
        tl = float(u[0])                              # the pilot never exceeds lev[0]: start at the smallest value
    if not th > tl:
        return None, None
    b = np.unique(np.r_[tl, th, u[(u > tl) & (u < th)], [a_j] if tl < a_j < th else []])
    right = b[1:]                                     # both functions are left-continuous: constant on (b_k, b_k+1]
    Q = R[:, G - np.searchsorted(u, right, side="left")]
    f = ((a_j >= right).astype(float)[None] - Q) ** 2
    return (f * np.diff(b)[None]).sum(1) / (th - tl), (tl, th)


def frozen_cv(XB, a_own, aux_vals, seed, grid=H_GRID, pilot=H_PILOT, lev=CV_LEV, chunk=50):
    """Frozen-target 5-fold CV of the regression bandwidth over the regression sample (each point held out once).
    aux_vals(T, idxs) -> [a(Y_i | T[t]) for i in idxs[t]] (indices into the regression sample); a_own[j] = a(Y_j|X_j).
    Returns (selected bandwidth, losses on the grid, number of held-out points without a level range)."""
    hs = list(grid) + ([pilot] if pilot not in grid else [])
    ip = hs.index(pilot); hmax = max(hs)
    idx = np.random.default_rng(seed + 2000).permutation(len(XB)); folds = np.array_split(idx, 5)
    tot = np.zeros(len(hs)); cnt = 0; nskip = 0
    for f in folds:
        tr = np.sort(np.setdiff1d(idx, f)); Xtr = XB[tr]
        for s0 in range(0, len(f), chunk):
            jj = f[s0:s0 + chunk]; T = XB[jj]
            inwin = np.all(np.abs(Xtr[None, :, :] - T[:, None, :]) < hmax, axis=2)
            Ws = [ll_weights_batch(Xtr, T, h) for h in hs]
            av = aux_vals(T, [tr[inwin[i]] for i in range(len(T))])
            for i in range(len(T)):
                L, _ = frozen_point_loss(av[i], np.stack([w[i][inwin[i]] for w in Ws]), a_own[jj[i]], ip, lev)
                if L is None:
                    nskip += 1; continue
                tot += L; cnt += 1
    loss = tot[:len(grid)] / max(cnt, 1)
    return float(grid[int(np.argmin(loss))]), loss, nskip


H_AXES = (0.3, 0.4, 0.55, 0.75, 1.0, 1.5)        # per-coordinate bandwidths of the coordinatewise CVs


def frozen_cv_vec(XB, a_own, aux_vals, seed, axes=H_AXES, pilot=H_PILOT, lev=CV_LEV, chunk=50, wfun=None):
    """frozen_cv over coordinatewise bandwidths, the product grid axes^d_X (wfun: weight function, default
    ll_weights_batch). Returns (selected bandwidth vector, losses in itertools.product order, number skipped)."""
    wfun = wfun or ll_weights_batch
    d = XB.shape[1]
    combos = [np.array(c, float) for c in (itertools.product(axes, repeat=d) if not ISO_ONLY else
                                             [(a,) * d for a in axes])]
    hs = combos + [np.full(d, float(pilot))]
    ip = len(hs) - 1; hmax = np.max(np.array(hs), axis=0)
    idx = np.random.default_rng(seed + 2000).permutation(len(XB)); folds = np.array_split(idx, 5)
    tot = np.zeros(len(hs)); cnt = 0; nskip = 0
    for f in folds:
        tr = np.sort(np.setdiff1d(idx, f)); Xtr = XB[tr]
        for s0 in range(0, len(f), chunk):
            jj = f[s0:s0 + chunk]; T = XB[jj]
            inwin = np.all(np.abs(Xtr[None, :, :] - T[:, None, :]) < hmax, axis=2)
            Ws = [wfun(Xtr, T, h) for h in hs]
            av = aux_vals(T, [tr[inwin[i]] for i in range(len(T))])
            for i in range(len(T)):
                L, _ = frozen_point_loss(av[i], np.stack([w[i][inwin[i]] for w in Ws]), a_own[jj[i]], ip, lev)
                if L is None:
                    nskip += 1; continue
                tot += L; cnt += 1
    loss = tot[:len(combos)] / max(cnt, 1)
    return combos[int(np.argmin(loss))], loss, nskip


def ind_cv_vec(a_own, XB, seed, axes=H_AXES):
    """ind_cv over coordinatewise bandwidths (product grid axes^d_X); returns (bandwidth vector, losses)"""
    d = XB.shape[1]
    combos = [np.array(c, float) for c in (itertools.product(axes, repeat=d) if not ISO_ONLY else
                                             [(a,) * d for a in axes])]
    loss = np.array([ind_cv(a_own, XB / c, seed, grid=(1.0,))[1][0] for c in combos])
    return combos[int(np.argmin(loss))], loss


def axes_edge(h, axes=H_AXES):
    """per coordinate: 'low' / 'high' end of the axis grid or 'interior'"""
    return ["low" if v == axes[0] else ("high" if v == axes[-1] else "interior") for v in h]


ICI_LEV = (0.85, 0.90, 0.95)                      # pilot coverages of the levels compared by the ICI rule
ICI_GAMMA = 2.0                                   # half-width of the ICI intervals in standard deviations


def ici_point(a, W, pilot, lev=ICI_LEV, gam=ICI_GAMMA):
    """Intersection-of-confidence-intervals (Lepski-type) choice of the regression bandwidth at one covariate x.
    a: values a(Y_i | x) of the points in the largest window; W: (n_h, n) local linear weights on the increasing
    bandwidth grid (zero rows where the window is too small); pilot: row whose envelope fixes the levels t_c with
    Qbar_pilot(t_c) = c, c in lev. Raw fits Qhat_h(t) = sum_i W[h, i] 1{a_i >= t}, standard deviation
    sqrt(c (1 - c) sum_i W[h, i]^2). Returns the largest row k such that the intervals Qhat_h(t_c) +- gam sd_h,
    over the valid rows h <= k, intersect at every level (None if no row is valid)."""
    ks = np.flatnonzero(np.any(W != 0, axis=1))
    if len(ks) == 0:
        return None
    u, R = rank_table(a, W[pilot] if np.any(W[pilot] != 0) else W[ks[-1]])
    ts = [(level_at(u, R, c), c) for c in lev]
    ts = [(t, c) for t, c in ts if t > -np.inf]
    if not ts:
        return int(ks[0])
    ind = np.stack([(a >= t).astype(float) for t, _ in ts])          # (n_lev, n)
    sdc = np.sqrt(np.array([c * (1 - c) for _, c in ts]))
    lo = np.full(len(ts), -np.inf); hi = np.full(len(ts), np.inf); best = int(ks[0])
    for k in ks:
        q = ind @ W[k]; s = gam * sdc * np.sqrt(np.sum(W[k] ** 2))
        lo = np.maximum(lo, q - s); hi = np.minimum(hi, q + s)
        if np.any(lo > hi):
            break
        best = int(k)
    return best


def ici_bandwidths(XB, T, aux_vals, grid=H_GRID, pilot=H_PILOT, chunk=50):
    """ICI bandwidths (values of grid) at the covariates T, with all regression points in the regression."""
    hs = sorted(grid); ip = hs.index(pilot); out = []
    for s0 in range(0, len(T), chunk):
        Tc = T[s0:s0 + chunk]; Ws = np.stack([ll_weights_batch(XB, Tc, h) for h in hs], axis=1)   # (nT, n_h, nB)
        idxs = [np.flatnonzero(np.any(Ws[i] != 0, axis=0)) for i in range(len(Tc))]
        for i, a in enumerate(aux_vals(Tc, idxs)):
            k = ici_point(a, Ws[i][:, idxs[i]], ip)
            out.append(np.nan if k is None else hs[k])
    return np.array(out)


def ind_cv(a_own, XB, seed, grid=H_GRID):
    """Regression bandwidth of rank calibration: five-fold CV (squared error) of the local linear regression of
    Z_i = 1{a(Y_i | X_i) >= q} on X_i over D_reg, q the alpha-quantile of a(Y_i | X_i). Returns (h, losses)."""
    Z = (a_own >= np.quantile(a_own, ALPHA)).astype(float)
    idx = np.random.default_rng(seed + 2000).permutation(len(XB)); folds = np.array_split(idx, 5); loss = []
    for h in grid:
        err = 0.0
        for f in folds:
            tr = np.setdiff1d(idx, f)
            err += np.mean((np.clip(ll_reg(XB[tr], Z[tr], XB[f], h), 0, 1) - Z[f]) ** 2)
        loss.append(err / len(folds))
    return float(grid[int(np.argmin(loss))]), np.array(loss)


def h_fixed(n_reg, beta=2.0):
    """rate bandwidth (log n / n)^(1 / (2 beta + d_X)) of rank calibration (reference rows -RANKFIX)"""
    return float((np.log(n_reg) / n_reg) ** (1 / (2 * beta + DX)))


def grid_edge(losses, keys):
    """'interior' or 'low'/'high' edge of the argmin on a one-dimensional grid"""
    k = int(np.argmin(losses))
    return "interior" if 0 < k < len(keys) - 1 else ("low" if k == 0 else "high")


# ====================================================================== fitted ranking functions / densities
class Fit:
    """A fitted conditional density (or ranking function); dens(rows) on rows (x, y). Caches values at the
    evaluation points of each evaluation covariate."""
    name = "fit"

    def __init__(self, ctx):
        self.ctx = ctx; self.cache = {}; self.t_dens_ev = 0.0; self.fit_s = 0.0

    def dens(self, rows):
        raise NotImplementedError

    def prop_dens(self, T, m=None):
        """(len(T), m) densities at the first m proposal points (all M if m is None)"""
        U = self.ctx.prop.U[:m]; M = len(U)
        E = np.empty((len(T) * M, DX + U.shape[1])); E[:, :DX] = np.repeat(T, M, 0); E[:, DX:] = np.tile(U, (len(T), 1))
        return self.dens(E).reshape(len(T), M)

    def aux_vals(self, T, idxs):
        """[a(Y_i | T[t]) for i in idxs[t]] for the auxiliary points (one batched call)"""
        YB = self.ctx.YB
        if sum(len(i) for i in idxs) == 0:
            return [np.empty(0) for _ in idxs]
        v = self.dens(np.vstack([np.hstack([np.repeat(T[t][None], len(i), 0), YB[i]]) for t, i in enumerate(idxs)]))
        return np.split(v, np.cumsum([len(i) for i in idxs])[:-1])

    def pts(self, j, kind):
        """densities at the evaluation points ("ev") or at the response grid ("grid") of evaluation covariate j"""
        key = (j, kind)
        if key not in self.cache:
            t0 = time.perf_counter()
            P = self.ctx.evpts(j, kind)
            self.cache[key] = self.dens(np.hstack([np.repeat(self.ctx.Xall[j][None], len(P), 0), P]))
            dt = time.perf_counter() - t0; STAGE["dens_ev"] += dt
            if kind == "ev":
                self.t_dens_ev += dt
        return self.cache[key]

    def pts_all(self, kind="ev"):
        """fill the cache for all evaluation covariates with one batched call"""
        miss = [j for j in range(self.ctx.nx) if (j, kind) not in self.cache]
        if not miss:
            return
        t0 = time.perf_counter()
        P = [self.ctx.evpts(j, kind) for j in miss]
        E = np.vstack([np.hstack([np.repeat(self.ctx.Xall[j][None], len(p), 0), p]) for j, p in zip(miss, P)])
        v = self.dens(E); o = 0
        for j, p in zip(miss, P):
            self.cache[(j, kind)] = v[o:o + len(p)]; o += len(p)
        dt = time.perf_counter() - t0; STAGE["dens_ev"] += dt
        if kind == "ev":
            self.t_dens_ev += dt


class EtaFit(Fit):
    """density eta2dens(fun(rows)) of a contrastive fit (NN or LP), floored at 1e-12 as in sim_eq.py"""
    def __init__(self, ctx, fun, name, chunk=200000, cap=0.995):
        super().__init__(ctx); self.fun, self.name, self.chunk, self.cap = fun, name, chunk, cap

    def dens(self, rows):
        out = np.empty(len(rows))
        for s0 in range(0, len(rows), self.chunk):
            A = rows[s0:s0 + self.chunk]
            out[s0:s0 + self.chunk] = np.maximum(eta2dens(self.fun(A), self.ctx.dY, A[:, DX:], self.cap), 1e-12)
        return out


class TrueFit(Fit):
    name = "true"
    def dens(self, rows):
        return self.ctx.model.tdensity(rows[:, :DX], rows[:, DX:])


class CKDE(Fit):
    """Nadaraya-Watson conditional KDE: product Epanechnikov kernel in x, Gaussian kernel in y (d_Y-dimensional);
    equals sim_contrastive.ckde for d_Y = 2."""
    name = "kde"

    def __init__(self, ctx, X, Y, hx, hy):
        super().__init__(ctx); self.X, self.Y, self.hx, self.hy = X, Y, hx, hy
        self.tree = cKDTree(X / hx); self.Y2 = (Y ** 2).sum(1)
        self.norm = (2 * np.pi * hy ** 2) ** (Y.shape[1] / 2)

    def _at(self, x0, Yq):
        nb = np.array(self.tree.query_ball_point(x0 / self.hx, r=1.0, p=np.inf), dtype=np.int64)
        if len(nb) == 0:
            return np.full(len(Yq), self.ctx.model.Q)
        w = np.prod(0.75 * (1 - ((self.X[nb] - x0) / self.hx) ** 2), axis=1)
        Yn = self.Y[nb]; out = np.empty(len(Yq)); ch = max(1, 4_000_000 // len(nb))
        for s0 in range(0, len(Yq), ch):
            yq = Yq[s0:s0 + ch]
            d2 = np.maximum((yq ** 2).sum(1)[:, None] + self.Y2[nb][None] - 2 * yq @ Yn.T, 0)
            out[s0:s0 + ch] = (np.exp(-d2 / (2 * self.hy ** 2)) @ w) / self.norm / max(w.sum(), 1e-12)
        return out

    def dens(self, rows):
        ux, inv = np.unique(rows[:, :DX], axis=0, return_inverse=True); inv = np.ravel(inv)
        out = np.empty(len(rows))
        order = np.argsort(inv, kind="stable"); bounds = np.searchsorted(inv[order], np.arange(len(ux) + 1))
        for g in range(len(ux)):
            r = order[bounds[g]:bounds[g + 1]]
            out[r] = self._at(ux[g], rows[r, DX:])
        return out


class CellFit(Fit):
    """Partition estimate: Gaussian KDE in y (bandwidth hy) of the responses whose covariates share the cube of side
    1/m containing x, divided by its mass on the box (closed form); a cube without fitting points gets the uniform
    density. Piecewise constant in x: proposal and auxiliary values are cached per cube."""
    name = "cell"

    def __init__(self, ctx, Xf, Yf, m, hy):
        super().__init__(ctx); self.m, self.hy = m, hy
        cid = self.cid(Xf); self.Ys = [Yf[cid == c] for c in range(m ** 3)]
        self.Z = np.array([np.mean(np.prod(ndtr((BOX - Ys) / hy) - ndtr((-BOX - Ys) / hy), axis=1)) if len(Ys) else 1.0
                           for Ys in self.Ys])
        self.norm = (2 * np.pi * hy ** 2) ** (Yf.shape[1] / 2)
        self.pcache, self.acache = {}, {}

    def cid(self, Xq):
        k = np.clip(np.floor(Xq * self.m).astype(int), 0, self.m - 1)
        return (k[:, 0] * self.m + k[:, 1]) * self.m + k[:, 2]

    def kde(self, c, Yq):
        Ys = self.Ys[c]
        if len(Ys) == 0:
            return np.full(len(Yq), self.ctx.model.Q)
        out = np.empty(len(Yq)); ch = max(1, 4_000_000 // len(Ys)); Y2 = (Ys ** 2).sum(1)
        for s0 in range(0, len(Yq), ch):
            yq = Yq[s0:s0 + ch]
            d2 = np.maximum((yq ** 2).sum(1)[:, None] + Y2[None] - 2 * yq @ Ys.T, 0)
            out[s0:s0 + ch] = np.exp(-d2 / (2 * self.hy ** 2)).mean(1) / self.norm
        return out / self.Z[c]

    def dens(self, rows):
        c = self.cid(rows[:, :DX]); out = np.empty(len(rows))
        for cc in np.unique(c):
            ii = np.flatnonzero(c == cc); out[ii] = self.kde(cc, rows[ii, DX:])
        return np.maximum(out, 1e-12)

    def prop_dens(self, T, m=None):
        c = self.cid(T)
        for cc in np.unique(c):
            if cc not in self.pcache:
                self.pcache[cc] = np.maximum(self.kde(cc, self.ctx.prop.U), 1e-12)
        return np.stack([self.pcache[cc] for cc in c])[:, :m]

    def aux_vals(self, T, idxs):
        out = []
        for cc, idx in zip(self.cid(T), idxs):
            if cc not in self.acache:
                self.acache[cc] = np.maximum(self.kde(cc, self.ctx.YB), 1e-12)
            out.append(self.acache[cc][idx])
        return out


# ====================================================================== the experiment
class Ctx:
    pass


def run(N, seed, scen, dY):
    global REF_OBJ
    T_START = time.perf_counter()
    N_REG = int(os.environ.get("N_REG", str(N // 6)))
    NVAL = int(os.environ.get("NVAL", str(N // 6)))
    NCAL = int(os.environ.get("NCAL", "1000"))
    M = int(os.environ.get("MC_M", {2: "16000", 3: "20000", 5: "40000"}[dY]))   # SD of the SNIS mass at u = 0.9 about 0.002
    M_CAL = int(os.environ.get("MC_M_CAL", str(M // 4)))  # proposal points for the HPD values of calibration and
                                                        # auxiliary points (SD about 0.004); evaluation covariates use M
    HP = tuple(float(v) for v in os.environ.get("MC_HP", "0.35").split(","))
    HP = HP[0] if len(HP) == 1 else HP
    L_TRUE = int(os.environ.get("L_TRUE", "20000")); L_MARG = int(os.environ.get("L_MARG", "5000"))
    CHECK_GRID = os.environ.get("CHECK_GRID") == "1" and dY == 2
    G = int(os.environ.get("GRID", "60"))
    ONLY = os.environ.get("ONLY")                       # optional comma-separated subset of method groups (tests)
    NO_LPUS = os.environ.get("NO_LPUS") == "1"          # skip the undersmoothed one-stage LP row (not reported)
    SWEEP = os.environ.get("RANK_SWEEP") == "1"         # tests: extra rank-calibration rows (ICI gamma = 1, each h)
    # Every row has "cov_x", the coverages at the evaluation covariates, which a first row "EVAL" lists (with their
    # regions), so that figures over the covariate can be drawn from the output alone.
    # EVAL_LINE=K (coverage curves): the evaluation covariates are K points on x_1 in [0.03, 0.97] with x_2 = x_3 = 0.5,
    # and only the one-stage rows of the full fits and the rank-calibration rows of the base fits are computed
    # (no -A, -RCP, -LOC, -US, RCP-PCP rows)
    LINE = int(os.environ.get("EVAL_LINE", "0"))
    # TAU_FILE=result file of the main experiment for this job: the one-stage rows HPD-NN, HPD-LP and CELL take their
    # thresholds from it instead of computing the HPD values of the calibration points (no CHCDS and PCP rows)
    TAU_SAVED = {}
    if os.environ.get("TAU_FILE"):
        saved = [r for r in map(json.loads, filter(lambda l: l.startswith("{"), open(os.environ["TAU_FILE"])))
                 if r["method"] in ("HPD-NN", "HPD-LP", "CELL")]
        assert len(saved) == 3 and all((r["seed"], r["dY"], r["M"], r["M_CAL"]) == (seed, dY, M, M_CAL) for r in saved)
        TAU_SAVED = {r["method"]: r["tau"] for r in saved}
    want = (lambda g: True) if not ONLY else (lambda g: g in ONLY.split(","))
    # d_Y >= 5: one more (wider) value at the top of the LP (h_X, h_Y), KDE h_Y and partition h_Y grids, which bind there
    G5 = ((3.0,), (0.95,), (0.45,), (0.65,)) if dY >= 5 else ((), (), (), ())
    base = {"N": N, "N_REG": N_REG, "NCAL": NCAL, "NVAL": NVAL, "dY": dY, "scen": scen, "seed": seed, "M": M, "M_CAL": M_CAL,
            "SDB": SDB, "REF": REF, "MC_HP": HP}

    ctx = Ctx(); ctx.dY = dY
    model = ctx.model = Model(scen, dY)
    rng = np.random.default_rng(seed)
    X, Y = model.sample(N, rng); Xc, Yc = model.sample(NCAL, rng)
    nA = N - N_REG
    XA, YA, XB, YB = X[:nA], Y[:nA], X[nA:], Y[nA:]
    REF_OBJ = Ref(YA) if REF == "gmix" else None        # reference from the responses of the first N - N_REG points
    ctx.XB, ctx.YB = XB, YB
    assert 0 < NVAL < nA
    VI = np.arange(nA - NVAL, nA)                       # tuning points: the last NVAL points of the first N - N_REG
    XV, YV = X[VI], Y[VI]
    tune_idx = lambda n: np.setdiff1d(np.arange(n), VI)  # tuning-fit part of a fitting sample of size n (X or XA)

    # ---------------- evaluation covariates (as sim_eq.py) and evaluation points
    with Timer("evaluation"):
        er = np.random.default_rng(4242)
        cand = er.uniform(0.02, 0.98, size=(20000, 3)); m_ = model.inR(cand)
        Xs = np.vstack([cand[m_][:10], cand[~m_][:10]]); Xm = model.sample(100, er)[0]
        if LINE:
            Xs = np.column_stack([np.linspace(0.03, 0.97, LINE), np.full(LINE, 0.5), np.full(LINE, 0.5)])
        NS, HARDX = len(Xs), model.inR(Xs)                  # set covariates (the first NS of Xall) and their region
        print(json.dumps({**base, "method": "EVAL", "line": LINE, "x": np.round(Xs, 5).tolist(),
                          "hard": [bool(h) for h in HARDX]}), flush=True)
        Xall = ctx.Xall = np.vstack([Xs, Xm]); nx = ctx.nx = len(Xall)
        ev = np.random.default_rng(4343 + seed)
        Uu = ev.uniform(-BOX, BOX, size=(L_TRUE, dY))                       # uniform half, shared by all x
        Lj = [L_TRUE if j < NS else L_MARG for j in range(nx)]
        EVP, PT, IW, TST, INS, LEBS = [], [], [], [], [], []
        for j in range(nx):
            Yt = model.draw_truncated(Xall[j], Lj[j], ev)
            P = np.vstack([Yt, Uu[:Lj[j]]]); EVP.append(P)
            pt = model.tdensity(np.repeat(Xall[j][None], len(P), 0), P); PT.append(pt)
            IW.append(1.0 / ((0.5 * pt + 0.5 * model.Q) * len(P)))           # Leb(A) ~ sum IW 1_A
            TST.append(np.quantile(pt[:Lj[j]], ALPHA)); INS.append(pt >= TST[-1]); LEBS.append(float(IW[-1] @ INS[-1]))
        if CHECK_GRID:
            g = np.linspace(-BOX, BOX, G + 1); g = (g[:-1] + g[1:]) / 2; cell = (g[1] - g[0]) ** 2
            Yg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2); ng = len(Yg)
            p_grid = model.density(np.repeat(Xall, ng, 0), np.tile(Yg, (nx, 1))).reshape(nx, ng)
            p_grid /= p_grid.sum(1, keepdims=True) * cell
    ctx.evpts = lambda j, kind: EVP[j] if kind == "ev" else Yg

    with Timer("proposal"):
        prop = ctx.prop = Proposal(YA, M, HP, seed, dY)   # responses of the first N - N_REG points only (not D_reg)
    wU = prop.wU

    # ---------------- evaluation
    def tau_of(s_cal):
        k = int(np.ceil((1 - ALPHA) * (NCAL + 1)))
        return np.inf if k > NCAL else float(np.sort(s_cal)[k - 1])

    def evaluate(name, s_cal, sfun, extra=None):
        """s_cal: calibration scores; sfun(j, kind) -> scores at the evaluation points (kind "ev") or grid ("grid")"""
        tau = TAU_SAVED[name] if name in TAU_SAVED else tau_of(s_cal)
        t0 = time.perf_counter(); S_EV = [sfun(j, "ev") for j in range(nx)]; t_ev = time.perf_counter() - t0
        with Timer("evaluation"):
            out = defaultdict(list)
            for j in range(nx):
                inC = S_EV[j] <= tau; L = Lj[j]; pt, iw = PT[j], IW[j]
                cov = float(inC[:L].mean()); lebC = float(iw @ inC)
                if j >= NS:
                    out["cM"].append(cov); out["vh"].append(lebC); out["vs"].append(LEBS[j]); continue
                out["cR" if HARDX[j] else "cO"].append(abs(cov - GAMMA))
                tc = np.quantile(pt[:L], min(max(1 - cov, 0.0), 1.0)); inCc = pt >= tc
                out["vol"].append(lebC / LEBS[j]); out["sym"].append(float(iw @ (inC ^ INS[j])) / LEBS[j])
                out["sh"].append(float(iw @ ((inC ^ inCc) * np.abs(pt - tc))) / tc / LEBS[j])
                out["shld"].append((lebC - float(iw @ inCc)) / LEBS[j])
                out["cov"].append(cov)
            res = {"marg": float(np.mean(out["cM"])), "cerrR": float(np.mean(out["cR"])), "cerrO": float(np.mean(out["cO"])),
                   "vol": float(np.mean(out["vol"])), "volM": float(np.sum(out["vh"]) / np.sum(out["vs"])),
                   "sym": float(np.mean(out["sym"])), "shape": float(np.mean(out["sh"])), "shape_ld": float(np.mean(out["shld"])),
                   "cmax": float(np.max(np.abs(np.array(out["cov"]) - GAMMA))), "tau": tau}
            res["cov_x"] = [round(c, 5) for c in out["cov"]]
        ex = dict(extra or {})
        if "time_s" in ex:
            ex["time_s"] = round(ex["time_s"] + t_ev, 2)
        print(json.dumps({**base, "method": name, **res, **ex}), flush=True)
        if CHECK_GRID:
            with Timer("evaluation"):
                og = defaultdict(list)
                for j in range(nx):
                    Ch = sfun(j, "grid") <= tau; pt = p_grid[j]; cov = float(pt[Ch].sum() * cell)
                    ot = np.argsort(-pt); cst = np.cumsum(pt[ot]) * cell
                    Cs_ = np.zeros(ng, bool); Cs_[ot[:np.searchsorted(cst, 1 - ALPHA) + 1]] = True
                    Cc = np.zeros(ng, bool); Cc[ot[:max(np.searchsorted(cst, cov) + 1, 1)]] = True
                    if j >= NS:
                        og["cM"].append(cov); og["vh"].append(Ch.sum()); og["vs"].append(Cs_.sum()); continue
                    og["cR" if HARDX[j] else "cO"].append(abs(cov - 0.9))
                    og["vol"].append(Ch.sum() / Cs_.sum()); og["sym"].append((Ch ^ Cs_).sum() / Cs_.sum())
                    kc = min(max(np.searchsorted(cst, cov) + 1, 1), ng); tc = pt[ot[kc - 1]]
                    og["sh"].append(np.abs(pt - tc)[Ch ^ Cc].sum() / tc / Cs_.sum())
                    og["shld"].append((Ch.sum() - Cc.sum()) / Cs_.sum())
                rg = {"marg": float(np.mean(og["cM"])), "cerrR": float(np.mean(og["cR"])), "cerrO": float(np.mean(og["cO"])),
                      "vol": float(np.mean(og["vol"])), "volM": float(np.sum(og["vh"]) / np.sum(og["vs"])),
                      "sym": float(np.mean(og["sym"])), "shape": float(np.mean(og["sh"])), "shape_ld": float(np.mean(og["shld"])),
                      "tau": tau}
            print(json.dumps({**base, "method": name + "@grid", **rg}), flush=True)

    # ---------------- one-stage HPD values by self-normalized importance sampling
    def snis(fit, T, own=None, keep_rows=False, chcds=False, pcp_rng=None, m=None):
        """For covariates T: HPD value of own densities (own: (nT,) estimated densities at the own points), and/or
        per-row sorted proposal densities with tail masses (keep_rows), CHCDS level and normalizer, PCP samples."""
        nT = len(T); out = {"hpd": np.empty(nT) if own is not None else None, "rows": [], "lev": np.empty(nT),
                            "Z": np.empty(nT), "pcp": []}
        m = M if m is None else min(m, M); wm = wU[:m]
        ch = max(1, 600000 // m)
        for s0 in range(0, nT, ch):
            pu = fit.prop_dens(T[s0:s0 + ch], m)
            w = pu * wm[None]; tot = w.sum(1)
            if own is not None:
                out["hpd"][s0:s0 + ch] = (w * (pu > own[s0:s0 + ch, None])).sum(1) / tot
            if keep_rows or chcds:
                o = np.argsort(pu, axis=1); ps = np.take_along_axis(pu, o, 1)
                tail = np.cumsum(np.take_along_axis(w, o, 1)[:, ::-1], 1)[:, ::-1] / tot[:, None]
                if keep_rows:
                    out["rows"] += [(ps[i], np.r_[tail[i], 0.0]) for i in range(len(pu))]
                if chcds:                                   # level of the super-level set with SNIS mass 1 - alpha
                    for i in range(len(pu)):
                        # descending cumulative mass reaches 1 - alpha at the largest index with tail >= 1 - alpha
                        k = max(int(np.searchsorted(-tail[i], -(1 - ALPHA), side="right")) - 1, 0)
                        out["lev"][s0 + i] = ps[i][k]
                    out["Z"][s0:s0 + ch] = (pu * wm[None]).mean(1)
            if pcp_rng is not None:
                for i in range(len(pu)):
                    out["pcp"].append(prop.U[pcp_rng.choice(m, size=50, p=w[i] / tot[i])])
        return out

    def hpd_lookup(row, v):
        ps, tail = row
        return tail[np.searchsorted(ps, v, side="right")]

    def one_stage(fit, chcds=False, pcp_rng=None, tag="hpd_mc"):
        """HPD-split scores of a fit: calibration scores, per-covariate eval rows; times."""
        rc = {"hpd": None}
        with Timer() as tc:
            with Timer(tag):
                if not TAU_SAVED:
                    own = fit.dens(np.hstack([Xc, Yc]))
                    rc = snis(fit, Xc, own=own, chcds=chcds, pcp_rng=pcp_rng, m=M_CAL)
        with Timer() as te:
            with Timer(tag):
                re = snis(fit, Xall, keep_rows=True, chcds=chcds, pcp_rng=pcp_rng)
        fit.pts_all("ev")
        s_cal = rc["hpd"]
        sfun = lambda j, kind: hpd_lookup(re["rows"][j], fit.pts(j, kind))
        return s_cal, sfun, rc, re, {"time_s": tc.dt + te.dt + fit.t_dens_ev, "cal_s": round(tc.dt, 2),
                                     "fit_s": round(fit.fit_s, 2)}

    # ---------------- regression helpers (as sim_eq.py; ll_reg and cv_bw at module level)
    knn_last = {}                                       # CV losses of the last knn_quantile_cv call
    def knn_quantile_cv(Xtr, Vtr, grid=(25, 50, 100, 200, 400, 600, 1000)):
        idx = np.random.default_rng(seed + 3000).permutation(len(Xtr)); folds = np.array_split(idx, 5); bestk = None
        knn_cv = {}
        for k in grid:
            if k >= 0.8 * len(Xtr): continue
            loss = 0.0
            for f in folds:
                tr = np.setdiff1d(idx, f)
                _, ii = cKDTree(Xtr[tr]).query(Xtr[f], k=k); q = np.quantile(Vtr[tr][ii], 1 - ALPHA, axis=1)
                u = Vtr[f] - q; loss += np.mean(np.maximum((1 - ALPHA) * u, -ALPHA * u))
            knn_cv[k] = round(float(loss), 7)
            if bestk is None or loss < bestk[0]:
                bestk = (loss, k)
        k = bestk[1]; tree = cKDTree(Xtr)
        knn_last.clear(); knn_last.update({"k_cv": {str(a): b for a, b in knn_cv.items()},
                                           "k_edge": grid_edge(list(knn_cv.values()), list(knn_cv))})
        return k, lambda Xq: np.quantile(Vtr[tree.query(Xq, k=k)[1]], 1 - ALPHA, axis=1)

    # ---------------- rank calibration (CV bandwidth and fixed bandwidth, sharing the evaluations of a(Y_i | x))
    def rank(fit):
        """Rank calibration: "-RANK" with coordinatewise regression bandwidths by frozen-target CV (frozen_cv_vec),
        "-RANKISO" with the isotropic bandwidth of the same CV losses (diagonal of the product grid), "-RANKFIX" with
        the rate bandwidth and "-RANKIND" with the isotropic bandwidth of ind_cv (references).
        Tests (RANK_SWEEP=1): isotropic frozen-target CV, ICI (each x, global, gamma = 1) and every grid value."""
        with Timer("rank") as tcv:
            a_own = fit.dens(np.hstack([XB, YB]))
            hx, fcv_loss_v, fcv_nskip_v = frozen_cv_vec(XB, a_own, fit.aux_vals, seed)
        with Timer("rank") as tind:
            hi, cv_loss = ind_cv(a_own, XB, seed)
        combos = list(itertools.product(H_AXES, repeat=DX)); iso = [i for i, c in enumerate(combos) if len(set(c)) == 1]
        hiso = np.array(combos[iso[int(np.argmin(fcv_loss_v[iso]))]], float) if not ISO_ONLY else hx   # isotropic frozen CV
        hf = h_fixed(len(XB))
        hs = sorted(H_GRID); ip = hs.index(H_PILOT)
        VAR = ["-RANK", "-RANKFIX", "-RANKIND", "-RANKISO"]; hvals = [hx, hf, hi, hiso]; NB = len(VAR)
        if SWEEP:
            with Timer("rank") as tfc:
                hfc, fc_loss, fc_nskip = frozen_cv(XB, a_own, fit.aux_vals, seed)
            with Timer("rank") as tici:
                chk = XB[np.random.default_rng(seed + 4000).choice(len(XB), size=min(300, len(XB)), replace=False)]
                h_chk = ici_bandwidths(XB, chk, fit.aux_vals)
                hv = np.sort(h_chk[np.isfinite(h_chk)])
                hg = float(hv[(len(hv) - 1) // 2]) if len(hv) else float(H_PILOT)
            VAR += ["-RANKFCV", "-RANKICI", "-RANKICIG", "-RANKICI1"] + [f"-RANKH{h}" for h in hs]
            hvals += [hfc, None, hg, None] + hs

        def tables(T):
            tabs, kx, hsel = tuple([] for _ in VAR), tuple([] for _ in VAR), []
            for s0 in range(0, len(T), 50):
                Tc = T[s0:s0 + 50]
                Wb = [ll_weights_batch(XB, Tc, h) for h in hvals[:NB]]
                Wg = np.stack([ll_weights_batch(XB, Tc, h) for h in hs], axis=1) if SWEEP else None
                idxs = []
                for i in range(len(Tc)):
                    nz = np.any([W[i] != 0 for W in Wb], axis=0)
                    if SWEEP:
                        nz = nz | np.any(Wg[i] != 0, axis=0)
                    idxs.append(np.flatnonzero(nz))
                for i, a in enumerate(fit.aux_vals(Tc, idxs)):
                    I = idxs[i]; ws = [W[i][I] for W in Wb]
                    if SWEEP:
                        G = Wg[i][:, I]
                        k = ici_point(a, G, ip); k = ip if k is None else k; hsel.append(hs[k])
                        k1 = ici_point(a, G, ip, gam=1.0); k1 = ip if k1 is None else k1
                        ws += [G[hs.index(hfc)], G[k], G[hs.index(hg)], G[k1]] + [G[m] for m in range(len(hs))]
                    for v, w in enumerate(ws):
                        kx[v].append(int((w != 0).sum())); tabs[v].append(rank_table(a, w))
            return tabs, kx, hsel

        with Timer("rank") as tc:
            tabs_c, kx_c, hsel_c = tables(Xc)
            own = fit.dens(np.hstack([Xc, Yc]))
            s_cal = [np.array([rank_eval(u, R, np.array([v]))[0] for (u, R), v in zip(tabs_c[k], own)]) for k in range(len(VAR))]
        with Timer("rank") as te:
            tabs_e, kx_e, hsel_e = tables(Xall)
        fit.pts_all("ev")
        out = []
        for v, (h, tag) in enumerate(zip(hvals, VAR)):
            kx = np.array(kx_c[v] + kx_e[v])
            sel_s = {"-RANK": tcv.dt, "-RANKISO": tcv.dt, "-RANKIND": tind.dt, "-RANKFCV": tfc.dt if SWEEP else 0.0,
                     "-RANKICIG": tici.dt if SWEEP else 0.0}.get(tag, 0.0)
            ex = {"time_s": tc.dt + te.dt + fit.t_dens_ev + sel_s,
                  "cal_s": round(tc.dt + sel_s, 2), "fit_s": round(fit.fit_s, 2),
                  "hx": [float(v) for v in h] if isinstance(h, np.ndarray) else h,
                  "kx_mean": float(kx.mean()), "kx_min": int(kx.min()), "nzero": int((kx == 0).sum())}
            if tag == "-RANK":
                ex.update({"cv_s": round(tcv.dt, 2), "cv_axes": list(H_AXES), "cv_edge": axes_edge(hx),
                           "fcv_nskip": fcv_nskip_v, "hx_fix": hf})
            if tag == "-RANKIND":
                ex.update({"cv_s": round(tind.dt, 2), "cv_grid": list(H_GRID), "cv_loss": [round(float(c), 7) for c in cv_loss],
                           "cv_edge": grid_edge(cv_loss, H_GRID)})
            if tag == "-RANKFCV":
                ex.update({"fcv_loss": [round(float(c), 7) for c in fc_loss], "fcv_nskip": fc_nskip,
                           "fcv_edge": grid_edge(fc_loss, H_GRID)})
            if tag == "-RANKICI":
                hloc = np.array(hsel_c + hsel_e)
                ex.update({"ici_h_dist": {str(h_): round(float(np.mean(hloc == h_)), 4) for h_ in hs},
                           "ici_h_mean": round(float(hloc.mean()), 4), "ici_gamma": ICI_GAMMA, "ici_lev": list(ICI_LEV)})
            if tag == "-RANKICIG":
                ex.update({"ici_chk_dist": {str(h_): round(float(np.mean(h_chk == h_)), 4) for h_ in hs},
                           "ici_s": round(tici.dt, 2), "ici_gamma": ICI_GAMMA})
            out.append((tag, s_cal[v], (lambda j, kind, tb=tabs_e[v]: rank_eval(*tb[j], fit.pts(j, kind))), ex))
        return out

    def own_cov(sB, s_cal_hpd, sfun_hpd):
        """"-LOC" (ablation): rank calibration with the HPD values of the regression points at their own covariates,
        i.e. the HPD-level recalibration of Dey et al. (2025, App. D.2) followed by split conformal. Local linear
        regression over D_reg of 1{s(X_i, Y_i) <= u}; score = nondecreasing envelope (clipped) at s(x, y); coordinatewise
        bandwidths by ind_cv_vec on these values (the CV of its own regression; same weights and axes as "-RANK")."""
        with Timer("rank") as t:
            h, loss = ind_cv_vec(-sB, XB, seed)

            def tabs(T):
                out = []
                for s0 in range(0, len(T), 200):
                    for w in ll_weights_batch(XB, T[s0:s0 + 200], h):
                        I = np.flatnonzero(w != 0); out.append(rank_table(-sB[I], w[I]))
                return out
            tc_, te_ = tabs(Xc), tabs(Xall)
            s_c = np.array([rank_eval(u, R, np.array([-v]))[0] for (u, R), v in zip(tc_, s_cal_hpd)])
        return s_c, (lambda j, kind: rank_eval(*te_[j], -sfun_hpd(j, kind))), \
            {"hx": [float(v) for v in h], "cv_axes": list(H_AXES), "cv_edge": axes_edge(h), "loc_s": round(t.dt, 2)}

    # ---------------- density fits
    def fit_density(Xf, Yf, tag):
        n = len(Xf); tr = tune_idx(n)
        Z, L = crows(Xf, Yf, np.random.default_rng(seed + (0 if tag == "full" else 77)))
        rows = lambda I: np.concatenate([I, n + (I[:, None] * K + np.arange(K)).ravel()])   # point i and its K references
        rtr = rows(tr); rv = np.random.default_rng(seed + 5).choice(rows(VI), size=min(3000, NVAL * (K + 1)), replace=False)
        Ztr, Ltr, Zvs, Lvs = Z[rtr], L[rtr], Z[rv], L[rv]
        with Timer("fit_nn") as t1:
            prn, n_ep = fit_net_tuned(Ztr, Ltr, Zvs, Lvs, Z, L, seed)
        from llfast_batch import local_linear_fast
        poly = list(range(DX + dY))
        llf = lambda A, sc: local_linear_fast(Z, L, A, sc, poly, ydim=dY)[0]
        lp_hx, lp_hy = LP_HX + G5[0], LP_HY + G5[1]
        with Timer("fit_lp") as t2:
            best = None
            lp_cv = {}
            for hx, hy in itertools.product(lp_hx, lp_hy):
                sc = np.array([hx] * DX + [hy] * dY)
                loss = bce(local_linear_fast(Ztr, Ltr, Zvs, sc, poly, ydim=dY)[0], Lvs); lp_cv[f"{hx},{hy}"] = round(float(loss), 6)
                if best is None or loss < best[0]:
                    best = (loss, hx, hy)
            sc_f = np.array([best[1]] * DX + [best[2]] * dY)                        # refit: all n points
        nn = EtaFit(ctx, prn, "NN", cap=0.995 if REF_OBJ is None else 0.9999); nn.fit_s = t1.dt
        lp = EtaFit(ctx, lambda A: llf(A, sc_f), "LP"); lp.fit_s = t2.dt
        # undersmoothed: bandwidths times n^{-(1/(2 beta + d_X) - 1/(2 beta + d_X + d_Y))}, beta = 2 (n^{-2/63} for d_Y = 2)
        us = n ** (-(1 / (4 + DX) - 1 / (4 + DX + dY)))
        lpus = EtaFit(ctx, lambda A: llf(A, sc_f * us), "LP-US"); lpus.fit_s = t2.dt
        info = {"lp_hx": best[1], "lp_hy": best[2], "lp_cv": lp_cv, "nn_epochs": n_ep,
                "lp_edge": [grid_edge([lp_cv[f"{h},{best[2]}"] for h in lp_hx], lp_hx),
                            grid_edge([lp_cv[f"{best[1]},{h}"] for h in lp_hy], lp_hy)]}
        log(f"[{time.perf_counter() - T_START:7.1f}s] fit {tag}: nn {t1.dt:.1f}s, lp {t2.dt:.1f}s, {info}")
        return nn, lp, lpus, info

    def fit_cell(Xf, Yf, ms=(2, 3, 4, 5), hys=(0.1, 0.15, 0.2, 0.3, 0.45) + G5[3]):
        with Timer("fit_cell") as t:
            best = None; tr = tune_idx(len(Xf)); cv = {}
            for m_, hy in itertools.product(ms, hys):
                f = CellFit(ctx, Xf[tr], Yf[tr], m_, hy)
                ll = np.mean(np.log(np.maximum(f.dens(np.hstack([XV, YV])), 1e-12))); cv[f"{m_},{hy}"] = round(float(ll), 6)
                if best is None or ll > best[0]:
                    best = (ll, m_, hy)
            f = CellFit(ctx, Xf, Yf, best[1], best[2])
        f.fit_s = t.dt
        return f, {"m": best[1], "hy": best[2], "cell_cv": cv,
                   "cell_edge": [grid_edge([-cv[f"{m_},{best[2]}"] for m_ in ms], ms),
                                 grid_edge([-cv[f"{best[1]},{h}"] for h in hys], hys)]}

    def ellipsoid(Xf, Yf):
        with Timer("fit_mean"):
            mnet = fit_mean_net(Xf, Yf, seed)
        res = Yf - mnet(Xf); tree = cKDTree(Xf)
        def prec_at(Xq, k=150):
            _, ii = tree.query(Xq, k=k); R = res[ii]; R = R - R.mean(1, keepdims=True)
            return np.linalg.inv(np.einsum("nki,nkj->nij", R, R) / (k - 1) + 1e-4 * np.eye(dY))
        def score_pts(Xq, Yq):
            d = Yq - mnet(Xq); return np.sqrt(np.einsum("ni,nij,nj->n", d, prec_at(Xq), d))
        me, Pe = mnet(Xall), prec_at(Xall)
        def sfun(j, kind):
            d = ctx.evpts(j, kind) - me[j]; return np.sqrt(np.einsum("ni,ij,nj->n", d, Pe[j], d))
        return mnet, score_pts, sfun

    def emit(name, s_cal, sfun, ex):
        ex = dict(ex); ex["time_s"] = ex.get("time_s", 0.0)
        evaluate(name, s_cal, sfun, ex)
        log(f"[{time.perf_counter() - T_START:7.1f}s] {name} done")

    rr = np.random.default_rng(seed + 7)                       # PCP draws (order: full cal, full eval, A cal, A eval, A aux)

    # ================= methods fitted on all N points
    nn_f, lp_f, lpus_f, info_f = fit_density(X, Y, "full")
    if want("nn"):
        s_cal, sfun, rc, re, ex = one_stage(nn_f, chcds=not TAU_SAVED, pcp_rng=None if TAU_SAVED else rr)
        emit("HPD-NN", s_cal, sfun, ex)
        if not TAU_SAVED:
            with Timer("baselines") as tb:
                own = nn_f.dens(np.hstack([Xc, Yc]))
                s_ch = (rc["lev"] - own) / rc["Z"]
            emit("CHCDS", s_ch, lambda j, kind: (re["lev"][j] - nn_f.pts(j, kind)) / re["Z"][j],
                 {"time_s": ex["time_s"] + tb.dt, "fit_s": round(nn_f.fit_s, 2)})
            with Timer("baselines") as tb:
                s_pcp = np.array([np.min(np.linalg.norm(rc["pcp"][i] - Yc[i], axis=1)) for i in range(NCAL)])
                trees = [cKDTree(p) for p in re["pcp"]]
            emit("PCP", s_pcp, lambda j, kind: trees[j].query(ctx.evpts(j, kind))[0],
                 {"time_s": ex["time_s"] + tb.dt, "fit_s": round(nn_f.fit_s, 2)})
    if want("lp"):
        s_cal, sfun, _, _, ex = one_stage(lp_f); emit("HPD-LP", s_cal, sfun, {**ex, **info_f})
        lp_f.cache.clear()
        if not LINE and not NO_LPUS:
            s_cal, sfun, _, _, ex = one_stage(lpus_f); emit("HPD-LP-US", s_cal, sfun, {**ex, **info_f})
            lpus_f.cache.clear()
    if want("kde"):
        with Timer("fit_kde") as tk:
            bestk = None; kde_cv = {}; KX, KY = (0.2, 0.35, 0.5, 0.7, 1.0), (0.05, 0.07, 0.1, 0.14, 0.2, 0.3) + G5[2]
            trk = tune_idx(N)
            for hx, hy in itertools.product(KX, KY):
                f = CKDE(ctx, X[trk], Y[trk], hx, hy)
                ll = np.mean(np.log(np.maximum(f.dens(np.hstack([XV[:800], YV[:800]])), 1e-8))); kde_cv[f"{hx},{hy}"] = round(float(ll), 6)
                if bestk is None or ll > bestk[0]:
                    bestk = (ll, hx, hy)
            kde = CKDE(ctx, X, Y, bestk[1], bestk[2])
        kde.fit_s = tk.dt
        s_cal, sfun, _, _, ex = one_stage(kde)
        emit("HPD-KDE", s_cal, sfun, {**ex, "hx": bestk[1], "hy": bestk[2], "kde_cv": kde_cv,
                                      "kde_edge": [grid_edge([-kde_cv[f"{h},{bestk[2]}"] for h in KX], KX),
                                                   grid_edge([-kde_cv[f"{bestk[1]},{h}"] for h in KY], KY)]})
        kde.cache.clear()
    if want("ell"):
        mnet_f, ell_pts_f, ell_sfun_f = ellipsoid(X, Y)
        with Timer("baselines") as tb:
            s_ell = ell_pts_f(Xc, Yc)
        emit("ELLIPSOID", s_ell, ell_sfun_f, {"time_s": tb.dt})
    if want("cell"):
        cell_f, inf_cf = fit_cell(X, Y)
        s_cal, sfun, _, _, ex = one_stage(cell_f); emit("CELL", s_cal, sfun, {**ex, **inf_cf})
        cell_f.cache.clear()
    del nn_f, lp_f, lpus_f

    # ================= methods with an auxiliary step: base on the first N - N_REG points, auxiliary on N_REG
    nn_a, lp_a, _, info_a = fit_density(XA, YA, "A")
    for name, fit, grp in (("HPD-NN", nn_a, "nn"), ("HPD-LP", lp_a, "lp")):
        if not want(grp):
            continue
        info = info_a if grp == "lp" else {}
        if LINE:
            for tag, s_cal, sfun, exr in rank(fit):
                emit(name + tag, s_cal, sfun, {**exr, **info})
            fit.cache.clear(); continue
        s_cal, sfun, rc, re, ex = one_stage(fit, pcp_rng=rr if grp == "nn" else None)
        emit(name + "-A", s_cal, sfun, {**ex, **info})              # diagnostic: one-stage on the same fit
        s1_cal, s1_sfun = s_cal, sfun
        # two-stage (quantile): RCP on the HPD score, conditional 0.9-quantile by k-NN on the auxiliary sample
        with Timer("rcp_aux_hpd") as ta:
            rB = snis(fit, XB, own=fit.dens(np.hstack([XB, YB])), pcp_rng=rr if grp == "nn" else None, m=M_CAL); sB = rB["hpd"]
        with Timer("baselines") as tq:
            kq, qfun = knn_quantile_cv(XB, sB)
            qc, qe = np.maximum(qfun(Xc), 1e-6), np.maximum(qfun(Xall), 1e-6)
        emit(name + "-RCP", s1_cal / qc, lambda j, kind, f=s1_sfun, qe=qe: f(j, kind) / qe[j],
             {"time_s": ex["time_s"] + ta.dt + tq.dt, "cal_s": round(ex["cal_s"] + ta.dt + tq.dt, 2), "k": int(kq),
              **knn_last, "fit_s": ex["fit_s"], **info})
        s_l, sfun_l, exl = own_cov(sB, s1_cal, s1_sfun)
        emit(name + "-LOC", s_l, sfun_l, {"time_s": ex["time_s"] + ta.dt + exl["loc_s"], **exl, "fit_s": ex["fit_s"], **info})
        for tag, s_cal, sfun, exr in rank(fit):
            emit(name + tag, s_cal, sfun, {**exr, **info})
        if grp == "nn":
            rc_nn_a, re_nn_a, pcpB = rc, re, rB["pcp"]
        fit.cache.clear()
    if want("cell"):
        cell_a, inf_ca = fit_cell(XA, YA)
    if want("cell") and LINE:
        for tag, s_cal, sfun, exr in rank(cell_a):
            emit("CELL" + tag, s_cal, sfun, {**exr, **inf_ca})
        cell_a.cache.clear()
    elif want("cell"):
        s_cal, sfun, _, _, ex = one_stage(cell_a); emit("CELL-A", s_cal, sfun, {**ex, **inf_ca})
        s1_cal, s1_sfun = s_cal, sfun
        with Timer("rcp_aux_hpd") as ta:
            sB = snis(cell_a, XB, own=cell_a.dens(np.hstack([XB, YB])), m=M_CAL)["hpd"]
        s_l, sfun_l, exl = own_cov(sB, s_cal, sfun)
        emit("CELL-LOC", s_l, sfun_l, {"time_s": ex["time_s"] + ta.dt + exl["loc_s"], **exl, "fit_s": ex["fit_s"], **inf_ca})
        for tag, s_cal, sfun, exr in rank(cell_a):
            emit("CELL" + tag, s_cal, sfun, {**exr, **inf_ca})
        # two-stage (quantile) as for NN and LP; emitted last, so the rows above are those of the runs without it
        with Timer("baselines") as tq:
            kq, qfun = knn_quantile_cv(XB, sB)
            qc, qe = np.maximum(qfun(Xc), 1e-6), np.maximum(qfun(Xall), 1e-6)
        emit("CELL-RCP", s1_cal / qc, lambda j, kind, f=s1_sfun, qe=qe: f(j, kind) / qe[j],
             {"time_s": ex["time_s"] + ta.dt + tq.dt, "cal_s": round(ex["cal_s"] + ta.dt + tq.dt, 2), "k": int(kq),
              **knn_last, "fit_s": ex["fit_s"], **inf_ca})
        cell_a.cache.clear()

    if want("ell"):
        mnet_a, ell_pts_a, ell_sfun_a = ellipsoid(XA, YA)
        with Timer("baselines") as tb:
            vB = ell_pts_a(XB, YB)
            kq, qfun = knn_quantile_cv(XB, vB)
            qc, qe = qfun(Xc), qfun(Xall); s_c = ell_pts_a(Xc, Yc) / qc
        emit("RCP-ELLIPSOID", s_c, lambda j, kind: ell_sfun_a(j, kind) / qe[j], {"time_s": tb.dt, "k": int(kq), **knn_last})
    if want("nn") and not LINE:
        with Timer("baselines") as tb:
            s_cal_a = np.array([np.min(np.linalg.norm(rc_nn_a["pcp"][i] - Yc[i], axis=1)) for i in range(NCAL)])
            trees_a = [cKDTree(p) for p in re_nn_a["pcp"]]
            mB = min(len(XB), 2000)                             # PCP scores of the first mB auxiliary points
            vB_pcp = np.array([np.min(np.linalg.norm(pcpB[i] - YB[i], axis=1)) for i in range(mB)])
            kq, qfun = knn_quantile_cv(XB[:mB], vB_pcp)
            qc, qe = qfun(Xc), qfun(Xall)
        emit("RCP-PCP", s_cal_a / qc, lambda j, kind: trees_a[j].query(ctx.evpts(j, kind))[0] / qe[j],
             {"time_s": tb.dt, "k": int(kq), **knn_last})
    if want("ell") and dY == 2:                             # OT-CP: d_Y = 2 only (its k-NN rank does not extrapolate in d_Y = 3)
        from scipy.optimize import linear_sum_assignment
        with Timer("baselines") as tb:
            mres = len(XB); rv = YB - mnet_a(XB)                          # all auxiliary points (1500)
            ro = np.random.default_rng(seed + 11)
            ang = ro.uniform(0, 2 * np.pi, mres); rad = np.sqrt(ro.uniform(size=mres))
            Ub = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)          # uniform on the unit disk
            ri, ci = linear_sum_assignment(((rv[:, None, :] - Ub[None]) ** 2).sum(-1)); Tm = np.empty_like(Ub); Tm[ri] = Ub[ci]
            rtree = cKDTree(rv)
            def ot_rank(r, k=5):
                _, ii = rtree.query(r, k=k)
                return np.linalg.norm(Tm[ii].mean(1), axis=-1)
            mc, me = mnet_a(Xc), mnet_a(Xall)
            s_ot = ot_rank(Yc - mc)
        emit("OT-CP", s_ot, lambda j, kind: ot_rank(ctx.evpts(j, kind) - me[j]), {"time_s": tb.dt})

    # ================= true-density references
    if want("oracle"):
        tr = TrueFit(ctx)
        s_cal, sfun, _, _, ex = one_stage(tr, tag="oracle")
        emit("ORACLE-COND", s_cal, sfun, ex)
        emit("ORACLE-MARG", -tr.dens(np.hstack([Xc, Yc])), lambda j, kind: -tr.pts(j, kind), {})

    log("# stage_times " + json.dumps({k: round(v, 1) for k, v in sorted(STAGE.items())}))
    log(f"# total {time.perf_counter() - T_START:.1f}s")


# ====================================================================== tests
def selftest():
    rng = np.random.default_rng(1)
    # 1. rank score vs brute force (ties, negative weights, envelope, clipping)
    worst = 0.0
    for rep in range(300):
        n = int(rng.integers(1, 40))
        a = np.round(rng.normal(size=n), int(rng.integers(0, 3)))          # rounding creates ties
        w = rng.normal(size=n) * rng.choice([0.05, 0.3, 1.0])
        if rep % 3 == 0:
            w = np.abs(w) / np.abs(w).sum() * rng.choice([0.8, 1.0, 1.3])  # nonnegative, total mass below/above 1
        t = np.r_[a, a + 1e-9, a - 1e-9, rng.normal(size=10), a.max() + 1, a.min() - 1]
        u, R = rank_table(a, w)
        q1, q2 = rank_eval(u, R, t), rank_bruteforce(a, w, t)
        worst = max(worst, np.abs(q1 - q2).max())
        o = np.argsort(t); assert np.all(np.diff(q1[o]) <= 1e-12), "Qbar not nonincreasing"
        Qhat = np.array([w[a >= tk].sum() for tk in t]); assert np.all(q1 >= np.clip(Qhat, 0, 1) - 1e-12)
        assert np.all((q1 >= 0) & (q1 <= 1))
    print(f"rank score vs brute force: max abs diff {worst:.2e} over 300 random cases")
    # explicit example: values 3,2,2,1 with weights 0.5,-0.2,0.4,0.6
    a = np.array([3., 2., 2., 1.]); w = np.array([0.5, -0.2, 0.4, 0.6]); u, R = rank_table(a, w)
    ex = rank_eval(u, R, np.array([4., 3., 2.5, 2., 1.5, 1., 0.]))
    print("example Qbar at t = 4,3,2.5,2,1.5,1,0:", ex, "(expect 0, .5, .5, .7, .7, 1, 1)")
    assert np.allclose(ex, [0, .5, .5, .7, .7, 1, 1])
    a = np.array([3., 2., 1.]); w = np.array([0.6, -0.4, 0.1]); u, R = rank_table(a, w)
    ex = rank_eval(u, R, np.array([3., 2., 1.]))
    print("envelope example (weights .6,-.4,.1):", ex, "(expect .6, .6, .6; raw Qhat = .6, .2, .3)")
    assert np.allclose(ex, [.6, .6, .6])
    # 1b. frozen-target CV loss vs brute force (levels by a fine scan of the pilot envelope, integral on a fine grid)
    worst_t, worst_L = 0.0, 0.0
    for rep in range(60):
        n = int(rng.integers(3, 16))
        a = np.round(rng.normal(size=n), 1)
        Wp = np.abs(rng.normal(size=n)); Wp *= rng.uniform(0.9, 1.2) / Wp.sum()
        W = np.vstack([rng.normal(size=n) * 0.3, rng.normal(size=n) * 0.1 + 1.0 / n, Wp])
        a_j = float(rng.choice(a)) if rep % 2 else float(rng.normal())
        L, lv = frozen_point_loss(a, W, a_j, 2)
        tt = np.linspace(a.min() - 1, a.max() + 1, 20001); Qp = rank_bruteforce(a, W[2], tt)
        tb = [tt[np.argmax(Qp <= c)] - (tt[1] - tt[0]) if np.any(Qp <= c) else -np.inf for c in CV_LEV]
        if tb[0] == -np.inf or tb[0] < a.min():
            tb[0] = a.min()
        if L is None:
            assert not tb[1] > tb[0] + 2 * (tt[1] - tt[0]), (tb, a, W[2])
            continue
        worst_t = max(worst_t, abs(lv[0] - tb[0]), abs(lv[1] - tb[1]))
        tm = np.linspace(lv[0], lv[1], 4001); tm = (tm[1:] + tm[:-1]) / 2
        Lb = np.array([np.mean(((a_j >= tm) - rank_bruteforce(a, W[k], tm)) ** 2) for k in range(3)])
        worst_L = max(worst_L, np.abs(Lb - L).max())
    print(f"frozen-target CV loss vs brute force: max level diff {worst_t:.1e} (scan step <= 3e-4), "
          f"max loss diff {worst_L:.1e} (integration grid 4000)")
    # end-to-end: frozen_cv against a direct loop over folds and held-out points (synthetic ranking function)
    XB = rng.uniform(size=(60, 3)); YB = rng.normal(size=(60, 2))
    afun = lambda x, Y: np.exp(-((Y - x[:2]) ** 2).sum(1))
    aux = lambda T, idxs: [afun(T[t], YB[i]) for t, i in enumerate(idxs)]
    a_own = np.array([afun(XB[j], YB[j:j + 1])[0] for j in range(60)])
    grid = (0.3, 0.55, 1.0, 1.5)
    hsel, loss, nskip = frozen_cv(XB, a_own, aux, 3, grid=grid)
    idx = np.random.default_rng(3 + 2000).permutation(60); tot = np.zeros(len(grid)); cnt = 0
    for f in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, f)
        for j in f:
            Wj = np.vstack([ll_weights_batch(XB[tr], XB[j:j + 1], h)[0] for h in grid])
            Lj, _ = frozen_point_loss(afun(XB[j], YB[tr]), Wj, a_own[j], grid.index(1.0))
            if Lj is not None:
                tot += Lj; cnt += 1
    print(f"frozen_cv vs direct loop: max loss diff {np.abs(tot / cnt - loss).max():.1e}; selected {hsel}, skipped {nskip}")
    assert np.allclose(tot / cnt, loss)
    # 2. local linear weights: reproduce linear functions, sum to one
    XB = rng.uniform(size=(2000, 3)); T = rng.uniform(size=(20, 3))
    Wt = ll_weights_batch(XB, T, 0.3)
    lin = XB @ np.array([0.3, -1.0, 2.0]) + 0.7
    print("ll weights: max |sum w - 1| %.1e, max |linear reproduction error| %.1e, min weight %.3f" % (
        np.abs(Wt.sum(1) - 1).max(), np.abs(Wt @ lin - (T @ np.array([0.3, -1.0, 2.0]) + 0.7)).max(), Wt.min()))
    # 3. models: box mass in closed form and normalization of the truncated density
    for scen in ("gauss", "smooth", "hetero"):
        print(scen)
        for dY in (2, 3):
            mo = Model(scen, dY); x = np.random.default_rng(5).uniform(size=(4, 3))
            ZZ = mo.box_mass(x)
            # box mass by Monte Carlo, and the normalization of the truncated density
            rr = np.random.default_rng(6); fr = []
            for i in range(4):
                Yd = mo.draw_y(np.repeat(x[i:i + 1], 200000, 0), rr); fr.append(np.mean(np.all(np.abs(Yd) <= BOX, 1)))
            Uq = rr.uniform(-BOX, BOX, size=(400000, dY))
            integ = [np.mean(mo.tdensity(np.repeat(x[i:i + 1], len(Uq), 0), Uq)) * (2 * BOX) ** dY for i in range(4)]
            print(f"   dY={dY}: P_x(box) closed form {np.round(ZZ, 5)} vs MC {np.round(fr, 5)}; "
                  f"int of truncated density {np.round(integ, 3)}")
    # 4. llfast_batch.local_linear_fast with ydim = 3 against the direct implementation
    from llfast_batch import local_linear_fast
    from llfast import local_linear_sd
    for dY in (2, 3):
        n = 3000; Xf = rng.uniform(size=(n, 3)); Yf = 0.5 * rng.standard_normal((n, dY))
        Z, L = crows(Xf, Yf, rng); sc = np.array([0.3] * 3 + [0.4] * dY)
        E = np.vstack([np.hstack([np.repeat(rng.uniform(size=(3, 3)), 500, 0), rng.uniform(-2, 2, (1500, dY))]),
                       np.hstack([rng.uniform(size=(500, 3)), rng.uniform(-2, 2, (500, dY))])])
        a1 = local_linear_fast(Z, L, E, sc, list(range(3 + dY)), ydim=dY)[0]; a2 = local_linear_sd(Z, L, E, sc, list(range(3 + dY)))[0]
        print(f"local_linear_fast(ydim={dY}) vs llfast.local_linear_sd: max abs diff {np.abs(a1 - a2).max():.1e}")
    # 5. proposal density integrates to one
    for dY in (2, 3):
        mo = Model("hetero", dY); _, Yf = mo.sample(5000, np.random.default_rng(2))
        pr = Proposal(Yf, 20000, 0.25, 0, dY)
        Uq = np.random.default_rng(8).uniform(-BOX, BOX, size=(200000, dY))
        print(f"proposal dY={dY}: int q over box (MC) = {np.mean(pr.density(Uq)) * (2 * BOX) ** dY:.4f}; "
              f"E_q[1/q]/vol = {np.mean(pr.wU) / (2 * BOX) ** dY:.4f}")


def mccheck(scen, dY, M, hp, reps=40):
    """SD over independent proposal sets of the SNIS estimate of the true mass of the 0.9-HPD set, at the 20
    evaluation covariates (true density in place of the estimate)."""
    mo = Model(scen, dY)
    _, Yf = mo.sample(9000, np.random.default_rng(0))
    er = np.random.default_rng(4242)
    cand = er.uniform(0.02, 0.98, size=(20000, 3)); m_ = mo.inR(cand)
    Xs = np.vstack([cand[m_][:10], cand[~m_][:10]])
    ev = np.random.default_rng(1)
    lev = []
    for j in range(20):
        Yt = mo.draw_truncated(Xs[j], 200000, ev)
        lev.append(np.quantile(mo.tdensity(np.repeat(Xs[j][None], len(Yt), 0), Yt), ALPHA))
    est = np.empty((reps, 20))
    for r in range(reps):
        pr = Proposal(Yf, M, hp, 100 + r, dY)
        for j in range(20):
            pu = mo.tdensity(np.repeat(Xs[j][None], M, 0), pr.U); w = pu * pr.wU
            est[r, j] = w[pu >= lev[j]].sum() / w.sum()
    sd = est.std(0, ddof=1); bias = est.mean(0) - GAMMA
    print(json.dumps({"scen": scen, "dY": dY, "M": M, "hp": hp, "sd_mean": float(sd.mean()), "sd_max": float(sd.max()),
                      "sd_hard": float(sd[:10].mean()), "sd_easy": float(sd[10:].mean()),
                      "absbias_max": float(np.abs(bias).max())}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "mccheck":
        mccheck(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    else:
        run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
