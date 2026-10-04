# -*- coding: utf-8 -*-
"""SlidingWindowRidge (anchored dual) machine-exactness selftest.

Checks the anchored dual refit against the direct primal solve of the
calibration-centred weighted ridge objective it claims to solve exactly:

    min_b ||y_w - Xw b||^2 + w ||y_a - Xa b||^2 + alpha ||b||^2

where every X and y is centred by the FROZEN calibration means.
Semantics under test: anchor = FULL centred calibration set, window
starts EMPTY and fills from the stream. Then:

  - fit() with w=1 is the intercept ridge on all of (X, y) exactly;
  - after k streamed (x, y) with a refit at each refit_every boundary,
    beta solves the weighted ridge on the centred union;
  - the first streamed prediction is the calibration ridge prediction.
"""
import numpy as np

from ghost_lattice.core import SlidingWindowRidge

rng = np.random.default_rng(0)
alpha = 10.0
n, d = 800, 60
X = rng.normal(size=(n, d)) + 3.0          # non-zero operating point
true_beta = rng.normal(size=d)
y = X @ true_beta + 2.0 + 0.1 * rng.normal(size=n)
xbar, ybar = X.mean(axis=0), y.mean()      # calibration stats

print("=== [1] w=1 fit(): intercept ridge on ALL of (X, y) ===")
from sklearn.linear_model import Ridge
sk = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
swr = SlidingWindowRidge(alpha=alpha, window=500, refit_every=25,
                         anchor_weight=1.0).fit(X, y)
print(f"beta vs sklearn      = {np.max(np.abs(swr.beta - sk.coef_)):.3e}")
print(f"intercept vs sklearn = "
      f"{abs((ybar - xbar @ swr.beta) - sk.intercept_):.3e}")

print("=== [2] after 40 streamed updates: centred weighted ridge ===")
stream_X = rng.normal(size=(40, d)) + 3.0
stream_y = stream_X @ true_beta + 2.0 + 0.1 * rng.normal(size=40)
preds = []
for i in range(40):
    preds.append(swr.predict(stream_X[i]))
    swr.update(stream_X[i], stream_y[i])
Xc = X - xbar
yc = y - ybar
Xs = stream_X[:25] - xbar
ys = stream_y[:25] - ybar
A = np.vstack([Xs, Xc])                     # window weight 1, anchor w=1
t = np.concatenate([ys, yc])
beta_ref = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ t)
print(f"beta vs direct primal= {np.max(np.abs(swr.beta - beta_ref)):.3e}")

print("=== [3] w=0 legacy pure-window mode (centred tail ridge) ===")
swr0 = SlidingWindowRidge(alpha=alpha, window=500, refit_every=25,
                          anchor_weight=0.0).fit(X, y)
Xw0 = (X[-500:] - xbar)
yw0 = y[-500:] - ybar
K = Xw0 @ Xw0.T + alpha * np.eye(500)
c = np.linalg.solve(K, yw0)
beta_pure = Xw0.T @ c
print(f"w=0 max|dual-primal| = {np.max(np.abs(swr0.beta - beta_pure)):.3e}")

print("=== [4] first-streamed-prediction continuity (w=1) ===")
swr2 = SlidingWindowRidge(alpha=alpha, window=500, refit_every=25,
                          anchor_weight=1.0).fit(X, y)
p1 = swr2.predict(stream_X[0])
sk1 = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
print(f"first pred diff      = {abs(p1 - float(sk1.predict(stream_X[0:1])[0])):.3e}")

print("SELFTEST_DONE")
