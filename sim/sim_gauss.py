"""Unimodal Gaussian model ("gauss") for Table 2: Y | x ~ N((m(x), 0), s(x)^2 I_2), with
m(x) = 0.6 Phi(idx_1(x)) - 0.3 and s(x) = 0.30 + 0.30 Phi(idx_2(x)), where idx(x) = sqrt(12) W^T (x - 1/2) are the two
linear indices of the smooth model. The level sets are circles with |grad_y p| bounded away from zero on them, so
Assumption 3.1(ii) holds. Hard region: {x_1 > 0.5}, as in the smooth model."""
import numpy as np
from scipy.stats import norm
from sim_contrastive import BOX


def params(x, W):
    t = np.sqrt(12) * (x - 0.5) @ W
    return 0.6 * norm.cdf(t[:, 0]) - 0.3, 0.30 + 0.30 * norm.cdf(t[:, 1])


def sample(n, W, rng):
    X = rng.uniform(size=(n, 3))
    m, s = params(X, W)
    Y = np.stack([m, np.zeros(n)], 1) + s[:, None] * rng.standard_normal((n, 2))
    return X, np.clip(Y, -BOX + 1e-6, BOX - 1e-6)


def density(x, y, W):
    m, s = params(x, W)
    return np.exp(-((y[:, 0] - m) ** 2 + y[:, 1] ** 2) / (2 * s**2)) / (2 * np.pi * s**2)
