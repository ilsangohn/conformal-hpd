"""UCI Bike Sharing data (hourly) for the bike-sharing application (realdata_rank.py).

Response  Y = (log(1 + casual + U1), log(1 + registered + U2)), U ~ U(0,1) (dequantization), affinely
mapped into the box [-BOX, BOX]^2.  Covariates X = (hour/23, temp, hum) in [0,1]^3.  The working-day
indicator is deliberately omitted, so that the conditional distribution of Y given X mixes commuting and leisure
patterns and is often bimodal. Evaluation groups: 5 hour blocks x 3 temperature terciles.
"""
import os
import numpy as np

from sim_contrastive import BOX

K, ALPHA = 5, 0.1
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "hour.csv")


def load(seed):
    raw = np.genfromtxt(DATA, delimiter=",", names=True)
    rng = np.random.default_rng(seed)
    y = np.c_[np.log1p(raw["casual"] + rng.uniform(size=len(raw))), np.log1p(raw["registered"] + rng.uniform(size=len(raw)))]
    x = np.c_[raw["hr"] / 23.0, raw["temp"], raw["hum"]]
    grp_hour = np.digitize(raw["hr"], [6, 10, 16, 20])                    # 0-5, 6-9, 10-15, 16-19, 20-23
    t1, t2 = np.quantile(raw["temp"], [1 / 3, 2 / 3])
    grp = grp_hour * 3 + np.digitize(raw["temp"], [t1, t2])
    # affine map of each response coordinate into the box (fixed across splits)
    lo, hi = np.quantile(y, 0.001, axis=0), np.quantile(y, 0.999, axis=0)
    scale = (hi - lo) / (2 * (BOX - 0.25)); centre = (hi + lo) / 2
    yb = np.clip((y - centre) / scale, -BOX + 1e-6, BOX - 1e-6)
    return x, yb, grp, scale
