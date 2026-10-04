"""Ghost-Lattice QNN core components.

Implements the full pipeline: LNN pre-gate, 29-qubit Ising reservoir
(tanh-oscillator surrogate), Volterra polynomial forge, convex readouts
(ridge / sliding-window ridge / GPU RLS variants), and the two genetic
algorithms (Chaos Algorithm, Spectral Genesis).

Author: Heylel Yaka 
License: CC BY-NC 4.0
"""

import ctypes
import glob
import os
import random
import sys

# --- OpenMP runtime unification (Windows/conda) ----------------------------
# torch and MKL-based scikit-learn each bundle their own OpenMP runtime;
# loading both crashes the process (segfault at import or teardown).
# Preloading the single MKL copy makes every later load resolve to the same
# module. Must happen before numpy/torch/sklearn are imported.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for _dll in glob.glob(os.path.join(sys.prefix, "Library", "bin",
                                       "libiomp5md.dll")):
        try:
            ctypes.CDLL(_dll)
        except OSError:
            pass

# --- heavy imports below ---------------------------------------------------
# Order matters on some Windows/conda setups: importing scikit-learn (MKL)
# BEFORE torch avoids a fatal clash between the two bundled OpenMP runtimes.

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_QUBITS = 29        # Ising reservoir size
SPATIAL_R = 3        # independent parallel reservoirs
TEMPORAL_V = 5       # temporal multiplexing taps
POLY_DEGREE = 2      # cross-pair order in the forge (cubes/4th powers added)

POP_SIZE_CHAOS = 8
GENERATIONS_CHAOS = 5
POP_SIZE_GENESIS = 8
GENERATIONS_GENESIS = 5
GA_TRAIN_SIZE = 500  # samples used for GA fitness evaluation
K_PER_BLOCK = 5000   # features kept per reservoir block for RLS


# ---------------------------------------------------------------------------
# Liquid Time-Constant (LNN) pre-gate
# ---------------------------------------------------------------------------

class TauNet(nn.Module):
    """Tiny network emitting a positive time-constant tau from recent input."""

    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return torch.abs(self.net(x)) + 0.5


def lnn_gate(inputs, tau_net=None):
    """Filter a (T,) or (T, D) signal through the adaptive LNN gate.

    Discrete backward-Euler-style update:
        alpha = tau / (tau + 1)
        x[t]  = alpha * x[t-1] + (1 - alpha) * u[t]
    with tau produced by the TauNet from a 4-step look-back window.
    """
    if tau_net is None:
        return inputs.copy()
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    inputs = inputs.astype(np.float32)
    T, D = inputs.shape
    x = np.zeros_like(inputs)
    x[0] = inputs[0] * 0.5
    dev = next(tau_net.parameters()).device
    for t in range(1, T):
        if t >= 3:
            past = inputs[t - 3:t + 1].flatten()
        else:
            past = np.zeros(4 * D, dtype=np.float32)
            past[(3 - t) * D:] = inputs[:t + 1].flatten()
        tau = tau_net(torch.tensor(past, device=dev)
                      .unsqueeze(0)).item()
        alpha = tau / (tau + 1.0)
        x[t] = alpha * x[t - 1] + (1 - alpha) * inputs[t]
    return x


# ---------------------------------------------------------------------------
# Ising reservoir (tanh-oscillator mean-field surrogate)
# ---------------------------------------------------------------------------

def init_reservoir(seed, input_dim=1):
    """Symmetric couplings J, local fields h, input projection W_in."""
    rng = np.random.RandomState(seed)
    J = rng.randn(N_QUBITS, N_QUBITS).astype(np.float32) * 0.1
    J = (J + J.T) / 2
    h = rng.randn(N_QUBITS).astype(np.float32) * 0.1
    W_in = rng.randn(input_dim, N_QUBITS).astype(np.float32) * 0.1
    return J, h, W_in


def simulate_reservoir(inputs, J, h, W_in):
    """Drive the reservoir and return temporally multiplexed features."""
    if inputs.ndim == 1:
        inputs = inputs[:, None].astype(np.float32)
    else:
        inputs = inputs.astype(np.float32)
    T, _ = inputs.shape
    state = np.zeros((T, N_QUBITS), dtype=np.float32)
    for t in range(1, T):
        state[t] = np.tanh(inputs[t] @ W_in + state[t - 1] @ J + h)
    taps = [state[max(0, T - 1 - v * (T // TEMPORAL_V))] for v in range(TEMPORAL_V)]
    return np.concatenate(taps).astype(np.float32)


# ---------------------------------------------------------------------------
# Polynomial forge (truncated Volterra series)
# ---------------------------------------------------------------------------

def polynomial_forge(raw_vector):
    """Expand reservoir taps: cross-pairs/squares + cubes + 4th powers."""
    raw_vector = raw_vector.astype(np.float32)
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    feats = poly.fit_transform(raw_vector.reshape(1, -1)).flatten().astype(np.float32)
    return np.concatenate([feats, raw_vector ** 3, raw_vector ** 4])


def per_reservoir_features(inputs, tau_net=None, J=None, h=None, W_in=None,
                           input_dim=1):
    """Filter, then run each of the R spatial reservoirs and forge features."""
    filtered = lnn_gate(inputs, tau_net)
    feats = []
    for r in range(SPATIAL_R):
        Jr, hr, W_inr = ((J, h, W_in) if J is not None
                         else init_reservoir(42 + r, input_dim))
        raw = simulate_reservoir(filtered, Jr, hr, W_inr)
        feats.append(polynomial_forge(raw))
    return feats


def combined_features(inputs, tau_net=None, J=None, h=None, W_in=None,
                      input_dim=1):
    """Concatenate all R reservoir blocks into one feature vector."""
    return np.concatenate(
        per_reservoir_features(inputs, tau_net, J, h, W_in, input_dim))


# ---------------------------------------------------------------------------
# Batched fast path (identical math, all windows in parallel)
# ---------------------------------------------------------------------------

def batch_lnn_gate(inputs, tau_net=None):
    """LNN gate for a batch of windows at once. inputs: (B, T) or (B, T, D).

    Same recurrence and the same TauNet as lnn_gate; the per-window
    time constants are computed in one batched forward pass per step.
    """
    if tau_net is None:
        return inputs.copy()
    if inputs.ndim == 2:
        inputs = inputs[:, :, None]
    inputs = inputs.astype(np.float32)
    B, T, D = inputs.shape
    dev = next(tau_net.parameters()).device
    pad = np.zeros((B, 3, D), dtype=np.float32)
    upad = np.concatenate([pad, inputs], axis=1)   # upad[:, t:t+4] = past
    x = np.zeros_like(inputs)
    x[:, 0] = inputs[:, 0] * 0.5
    for t in range(1, T):
        past = torch.tensor(upad[:, t:t + 4].reshape(B, 4 * D),
                            dtype=torch.float32, device=dev)
        tau = tau_net(past).squeeze(-1)             # (B,)
        alpha = (tau / (tau + 1.0)).detach().cpu().numpy()[:, None]
        x[:, t] = alpha * x[:, t - 1] + (1 - alpha) * inputs[:, t]
    return x


def batch_reservoir(filtered, J, h, W_in):
    """Ising surrogate for a batch: filtered (B, T, D) -> taps (B, N*V).

    Same recurrence as simulate_reservoir, batched over windows.
    """
    B, T, D = filtered.shape
    Jt = torch.tensor(J, dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)
    Wt = torch.tensor(np.asarray(W_in).reshape(D, N_QUBITS),
                      dtype=torch.float32)
    states = np.zeros((B, T, N_QUBITS), dtype=np.float32)
    state = torch.zeros(B, N_QUBITS, dtype=torch.float32)
    filt = torch.tensor(filtered, dtype=torch.float32)
    for t in range(1, T):
        u = filt[:, t] @ Wt                        # (B, N)
        state = torch.tanh(u + state @ Jt + ht)
        states[:, t] = state.numpy()
    taps = [states[:, max(0, T - 1 - v * (T // TEMPORAL_V)), :]
            for v in range(TEMPORAL_V)]
    return np.concatenate(taps, axis=1).astype(np.float32)


def batch_polynomial_forge(raw):
    """Volterra forge for a batch: raw (B, n_taps) -> (B, ~11k)."""
    n_taps = raw.shape[1]
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    poly.fit(np.zeros((1, n_taps)))
    feats = poly.transform(raw).astype(np.float32)
    return np.concatenate([feats, raw ** 3, raw ** 4], axis=1)


def batch_per_reservoir_features(inputs_batch, tau_net=None, J=None,
                                 h=None, W_in=None, input_dim=1):
    """Per-reservoir forge blocks for a batch; returns R arrays (B, ~11k)."""
    if inputs_batch.ndim == 2:
        inputs_batch = inputs_batch[:, :, None]
    filtered = batch_lnn_gate(inputs_batch, tau_net)
    blocks = []
    for r in range(SPATIAL_R):
        Jr, hr, W_inr = ((J, h, W_in) if J is not None
                         else init_reservoir(42 + r,
                                             input_dim=inputs_batch
                                             .shape[2]))
        taps = batch_reservoir(filtered, Jr, hr, W_inr)
        blocks.append(batch_polynomial_forge(taps))
    return blocks


def batch_combined_features(inputs_batch, tau_net=None, J=None, h=None,
                            W_in=None, input_dim=1):
    """Full forge for a batch: (B, T[, D]) -> (B, R*~11k)."""
    return np.concatenate(
        batch_per_reservoir_features(inputs_batch, tau_net, J, h, W_in,
                                     input_dim), axis=1)


def features_for_set(X, tau_net=None, J=None, h=None, W_in=None,
                     input_dim=1, batched=True):
    """Feature matrix for a window set, batched when possible.

    Mathematically identical to the per-window loop path (see
    test in run_mackey_glass --verify-batch); 100-1000x faster.
    """
    if batched and isinstance(X, np.ndarray) and X.ndim in (2, 3):
        return batch_combined_features(X, tau_net, J, h, W_in, input_dim)
    return np.array([combined_features(x, tau_net, J, h, W_in, input_dim)
                     for x in X])


# ---------------------------------------------------------------------------
# Echo State Network baseline
# ---------------------------------------------------------------------------

class EchoStateNetwork:
    """Classical ESN baseline: random sparse reservoir + ridge readout."""

    def __init__(self, n_reservoir=200, spectral_radius=0.9,
                 input_scaling=0.1, ridge_alpha=10.0):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha

    def _states(self, X):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for i in range(n_samples):
            state = np.tanh(self.W_in @ X_flat[i] + self.W @ state)
            states[i] = state
        return states

    def fit(self, X, y):
        n_features = X.reshape(X.shape[0], -1).shape[1]
        self.W_in = (np.random.randn(self.n_reservoir, n_features)
                     * self.input_scaling)
        W = np.random.randn(self.n_reservoir, self.n_reservoir) * 0.1
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < 0.1
        W *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / radius) * self.spectral_radius
        self.readout = Ridge(alpha=self.ridge_alpha)
        self.readout.fit(self._states(X), y)
        return self

    def predict(self, X):
        return self.readout.predict(self._states(X))


# ---------------------------------------------------------------------------
# Chaos Algorithm (tau-net evolution)
# ---------------------------------------------------------------------------

def get_params(net):
    return np.concatenate([p.data.cpu().numpy().flatten()
                           for p in net.parameters()])


def set_params(net, flat):
    idx = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(flat[idx:idx + numel].reshape(p.shape),
                              device=DEVICE)
        idx += numel


def chaos_algorithm(X_train, y_train, X_val, y_val, tau_net,
                    input_dim, pop_size, generations):
    """Evolve TauNet weights: tournament elites, uniform crossover, mutation."""
    pop = [get_params(tau_net)
           + np.random.randn(get_params(tau_net).shape[0]).astype(np.float32) * 0.01
           for _ in range(pop_size)]
    for gen in range(generations):
        fitness = []
        for params in pop:
            set_params(tau_net, params)
            feats = features_for_set(X_train, tau_net,
                                     input_dim=input_dim)
            feats = StandardScaler().fit_transform(feats)
            ridge = Ridge(alpha=1.0).fit(feats, y_train)
            mse = mean_squared_error(y_val, ridge.predict(X_val))
            fitness.append(mse)
        elite_idx = np.argsort(fitness)[:2]
        new_pop = [pop[i] for i in elite_idx]
        while len(new_pop) < pop_size:
            p1 = pop[random.choice(elite_idx)]
            p2 = pop[random.choice(elite_idx)]
            child = (np.where(np.random.rand(len(p1)) < 0.5, p1, p2)
                     + np.random.randn(len(p1)).astype(np.float32) * 0.02)
            new_pop.append(child)
        pop = new_pop
        print(f"  Chaos Gen {gen + 1}: best MSE = {min(fitness):.6f}")
    set_params(tau_net, pop[0])
    return tau_net


# ---------------------------------------------------------------------------
# Spectral Genesis (reservoir evolution)
# ---------------------------------------------------------------------------

def qr_crossover(J_a, J_b, rng=None):
    """QR recombination: swap orthogonal bases between two parents.

    Both parents are symmetric; Q from the second parent spans an
    orthogonal mixing basis that is recombined with the first parent's
    spectrum, preserving symmetry and rank structure.
    """
    rng = rng or np.random
    Ua, sa, _ = np.linalg.svd(J_a)
    Qb, _ = np.linalg.qr(J_b)
    child = Ua @ np.diag(sa) @ Qb.T
    child = (child + child.T) / 2
    return child.astype(np.float32)


def spectral_genesis(X_train, y_train, X_val, y_val, tau_net,
                     J_init, h_init, W_in, input_dim, pop_size, generations,
                     use_qr_crossover=True):
    """Evolve J, h: SVD singular-value mutation, QR recombination,
    symmetry restoration, and a spectral-radius filter rejecting rho(J) >= 1."""
    J_pop = [J_init + np.random.randn(*J_init.shape).astype(np.float32) * 0.01
             for _ in range(pop_size)]
    h_pop = [h_init + np.random.randn(*h_init.shape).astype(np.float32) * 0.01
             for _ in range(pop_size)]
    for gen in range(generations):
        fitness = []
        for J, h in zip(J_pop, h_pop):
            sr = np.max(np.abs(np.linalg.eigvals(J)))
            if sr >= 1.0:
                J = J / (sr + 0.1)
            feats = features_for_set(X_train, tau_net, J, h, W_in,
                                     input_dim=input_dim)
            feats = StandardScaler().fit_transform(feats)
            ridge = Ridge(alpha=1.0).fit(feats, y_train)
            mse = mean_squared_error(y_val, ridge.predict(X_val))
            fitness.append(mse)
        elite_idx = np.argsort(fitness)[:2]
        new_J = [J_pop[i] for i in elite_idx]
        new_h = [h_pop[i] for i in elite_idx]
        while len(new_J) < pop_size:
            p1_J = J_pop[random.choice(elite_idx)]
            p2_J = J_pop[random.choice(elite_idx)]
            p1_h = h_pop[random.choice(elite_idx)]
            p2_h = h_pop[random.choice(elite_idx)]
            child_h = np.where(np.random.rand(*p1_h.shape) < 0.5, p1_h, p2_h)
            if use_qr_crossover:
                child_J = qr_crossover(p1_J, p2_J)
            else:
                child_J = np.where(np.random.rand(*p1_J.shape) < 0.5,
                                   p1_J, p2_J)
            U, s, Vt = np.linalg.svd(child_J)
            child_J = U @ np.diag(s * (1 + np.random.randn(len(s))
                                       .astype(np.float32) * 0.05)) @ Vt
            child_J = (child_J + child_J.T) / 2
            new_J.append(child_J.astype(np.float32))
            new_h.append(child_h.astype(np.float32))
        J_pop, h_pop = new_J, new_h
        print(f"  Genesis Gen {gen + 1}: best MSE = {min(fitness):.6f}")
    return J_pop[0].astype(np.float32), h_pop[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Online readouts
# ---------------------------------------------------------------------------

class SlidingWindowRidge:
    """Periodically re-solve ridge on the last M samples.

    Retains convex optimality while tracking non-stationary streams; a more
    robust alternative to RLS in the high-dimensional forge space.
    """

    def __init__(self, alpha=10.0, window=500):
        self.alpha = alpha
        self.window = window
        self.recent_X = []
        self.recent_y = []
        self.model = Ridge(alpha=alpha)

    def update(self, x, y):
        self.recent_X.append(np.asarray(x, dtype=np.float64).ravel())
        self.recent_y.append(float(y))
        if len(self.recent_X) > self.window:
            self.recent_X.pop(0)
            self.recent_y.pop(0)
        X = np.vstack(self.recent_X)
        yy = np.hstack(self.recent_y)
        self.model.fit(X, yy)

    def predict(self, x):
        return self.model.predict(
            np.asarray(x, dtype=np.float64).reshape(1, -1))[0]


class GPUBlockDiagonalRLS:
    """Stabilised block-diagonal RLS: one P per reservoir block on GPU."""

    def __init__(self, feat_dims_per_block, alpha=10.0, device="cuda"):
        self.num_blocks = len(feat_dims_per_block)
        self.alpha = alpha
        self.device = device
        self.P = [torch.eye(d, dtype=torch.float64, device=device) / alpha
                  for d in feat_dims_per_block]
        self.beta = [torch.zeros(d, dtype=torch.float64, device=device)
                     for d in feat_dims_per_block]
        self.offsets = torch.tensor(
            [0] + list(feat_dims_per_block[:-1]), device=device).cumsum(dim=0)

    def predict(self, x_tensor):
        pred = 0.0
        for b in range(self.num_blocks):
            start = self.offsets[b].item()
            end = start + len(self.beta[b])
            pred += torch.dot(self.beta[b], x_tensor[start:end])
        return pred

    def update(self, x_tensor, y_tensor):
        error = y_tensor - self.predict(x_tensor)
        for b in range(self.num_blocks):
            start = self.offsets[b].item()
            end = start + len(self.beta[b])
            xb = x_tensor[start:end]
            Px = self.P[b] @ xb
            denom = torch.clamp(1.0 + torch.dot(xb, Px), min=1.0)
            gain = torch.clamp(Px / denom, min=-1.0, max=1.0)
            self.P[b] = self.P[b] - torch.outer(gain, Px)
            self.beta[b] += gain * error


class FullRLS:
    """Full-matrix RLS (Woodbury update) on a single P matrix."""

    def __init__(self, feature_dim, alpha=10.0, device="cuda"):
        self.alpha = alpha
        self.device = device
        self.P = torch.eye(feature_dim, dtype=torch.float64,
                           device=device) / alpha
        self.beta = torch.zeros(feature_dim, dtype=torch.float64,
                                device=device)

    def predict(self, x_tensor):
        return torch.dot(self.beta, x_tensor)

    def update(self, x_tensor, y_tensor):
        error = y_tensor - self.predict(x_tensor)
        Px = self.P @ x_tensor
        denom = torch.clamp(1.0 + torch.dot(x_tensor, Px), min=1.0)
        gain = torch.clamp(Px / denom, min=-1.0, max=1.0)
        self.P = self.P - torch.outer(gain, Px)
        self.beta += gain * error


def select_blocks(train_blocks_raw, test_blocks_raw, y_train, k=K_PER_BLOCK):
    """Standardise each reservoir block and keep its top-k f_regression features."""
    train_sel, test_sel, selectors = [], [], []
    for X_tr, X_te in zip(train_blocks_raw, test_blocks_raw):
        sc = StandardScaler()
        tr = sc.fit_transform(X_tr)
        te = sc.transform(X_te)
        sel = SelectKBest(f_regression, k=min(k, tr.shape[1]))
        train_sel.append(sel.fit_transform(tr, y_train))
        test_sel.append(sel.transform(te))
        selectors.append(sel)
    return selectors, train_sel, test_sel


def ridge_warm_start(X, y, alpha=10.0):
    """Closed-form ridge solution for warm-starting RLS.

    Returns (P, beta) with P = (X'X + alpha I)^-1, beta = P X'y.
    """
    XtX = X.T @ X
    P = np.linalg.inv(XtX + alpha * np.eye(X.shape[1]))
    beta = P @ X.T @ y
    return P, beta


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
