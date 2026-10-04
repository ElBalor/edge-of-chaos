#!/usr/bin/env python3
"""Ghost-Lattice QNN - Mackey-Glass tau=17 benchmark.

Victus-optimised run: ESN / GRU / linear baselines, closed-form ridge on
the complete forge, and the online sliding-window dual ridge (M=500).

Author: Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0
"""

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# torch AFTER scikit-learn: avoids the Windows/conda OpenMP double-load
# crash (MKL vs torch runtimes) on this machine class.
import torch
import torch.nn as nn

from ghost_lattice.core import (
    DEVICE,
    GA_TRAIN_SIZE,
    GENERATIONS_CHAOS,
    GENERATIONS_GENESIS,
    POP_SIZE_CHAOS,
    POP_SIZE_GENESIS,
    EchoStateNetwork,
    SlidingWindowRidge,
    TauNet,
    chaos_algorithm,
    combined_features,
    init_reservoir,
    rmse,
    spectral_genesis,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MG_TAU = 17
MG_BETA = 0.2
MG_GAMMA = 0.1
MG_N = 10
SEQUENCE_LENGTH = 5000
WINDOW_SIZE = 30
FORECAST_HORIZON = 6      # can the shii to 12 for the harder variant
GA_VAL_SIZE = 200


# ---------------------------------------------------------------------------
# Mackey-Glass generator
# ---------------------------------------------------------------------------

def mackey_glass(n, beta=MG_BETA, gamma=MG_GAMMA, tau=MG_TAU, n_pow=MG_N):
    x = np.zeros(n)
    x[0] = 1.2
    for t in range(tau, n - 1):
        x[t + 1] = (x[t]
                    + beta * x[t - tau] / (1 + x[t - tau] ** n_pow)
                    - gamma * x[t])
    return x


def make_windows(series, window, horizon):
    X = [series[i:i + window]
         for i in range(len(series) - window - horizon)]
    y = [series[i + window + horizon]
         for i in range(len(series) - window - horizon)]
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# ---------------------------------------------------------------------------
# GRU baseline
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Ghost-Lattice QNN - Mackey-Glass benchmark")
    ap.add_argument("--horizon", type=int, default=FORECAST_HORIZON,
                    help="forecast horizon (6 or 12)")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="ridge regularisation for the final readout")
    ap.add_argument("--quick", action="store_true",
                    help="tiny subsets + 1 GA generation (smoke test)")
    args = ap.parse_args()

    seq_len = SEQUENCE_LENGTH
    ga_train_size = GA_TRAIN_SIZE
    ga_val_size = GA_VAL_SIZE
    pop_c, gen_c = POP_SIZE_CHAOS, GENERATIONS_CHAOS
    pop_g, gen_g = POP_SIZE_GENESIS, GENERATIONS_GENESIS
    if args.quick:
        seq_len, ga_train_size, ga_val_size = 1600, 60, 30
        pop_c = gen_c = pop_g = gen_g = 2
        gen_c = gen_g = 1

    print(f"Device: {DEVICE}")
    print("Generating Mackey-Glass series...")
    series = mackey_glass(seq_len + args.window + args.horizon + MG_TAU)
    series = series[MG_TAU:]
    series = (series - series.mean()) / series.std()

    X, y = make_windows(series, args.window, args.horizon)
    print(f"Total windows: {X.shape[0]}, input shape: {X.shape[1]}")

    if args.quick:
        n_tr, n_va, n_te = 150, 60, 60
    else:
        n_tr, n_va, n_te = 2000, 500, 500
    train_X, train_y = X[:n_tr], y[:n_tr]
    val_X, val_y = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    test_X, test_y = (X[n_tr + n_va:n_tr + n_va + n_te],
                      y[n_tr + n_va:n_tr + n_va + n_te])
    ga_train_X, ga_train_y = train_X[:ga_train_size], train_y[:ga_train_size]
    ga_val_X, ga_val_y = val_X[:ga_val_size], val_y[:ga_val_size]

    # ----- Baselines -----
    print("\n=== Baselines ===")
    lr = LinearRegression().fit(train_X[:, :10], train_y)
    lr_rmse = rmse(test_y, lr.predict(test_X[:, :10]))
    print(f"Linear AR(10) RMSE: {lr_rmse:.6f}")

    gru = GRUModel().to(DEVICE)
    opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
    tX = torch.tensor(train_X, device=DEVICE).float()
    ty = torch.tensor(train_y, device=DEVICE).float().unsqueeze(1)
    for _ in range(50):
        opt.zero_grad()
        nn.MSELoss()(gru(tX), ty).backward()
        opt.step()
    gru.eval()
    with torch.no_grad():
        pred = gru(torch.tensor(test_X, device=DEVICE).float())
        gru_rmse = rmse(test_y, pred.cpu().numpy().flatten())
    print(f"GRU (32 units, 50 epochs) RMSE: {gru_rmse:.6f}")

    esn = EchoStateNetwork(n_reservoir=200, spectral_radius=0.9,
                           input_scaling=0.1, ridge_alpha=10.0)
    esn.fit(train_X, train_y)
    esn_rmse = rmse(test_y, esn.predict(test_X))
    print(f"ESN (200 neurons) RMSE: {esn_rmse:.6f}")

    # ----- Ghost-Lattice -----
    print("\n=== Ghost-Lattice QNN  ===")
    val_feats = np.array([combined_features(x) for x in ga_val_X])

    print("\n--- Chaos Algorithm ---")
    tau_net = TauNet(input_dim=4).to(DEVICE)
    tau_net_best = chaos_algorithm(ga_train_X, ga_train_y, val_feats,
                                   ga_val_y, tau_net, input_dim=1,
                                   pop_size=pop_c, generations=gen_c)

    print("\n--- Spectral Genesis ---")
    J_init, h_init, W_in = init_reservoir(42, input_dim=1)
    J_best, h_best = spectral_genesis(ga_train_X, ga_train_y, val_feats,
                                      ga_val_y, tau_net_best, J_init,
                                      h_init, W_in, input_dim=1,
                                      pop_size=pop_g, generations=gen_g)

    print("\n--- Full Ridge Regression ---")
    train_feats = np.array(
        [combined_features(x, tau_net_best, J_best, h_best, W_in)
         for x in tqdm(train_X)])
    test_feats = np.array(
        [combined_features(x, tau_net_best, J_best, h_best, W_in)
         for x in test_X])
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats = scaler.transform(test_feats)
    ridge = Ridge(alpha=args.alpha).fit(train_feats, train_y)
    ridge_rmse = rmse(test_y, ridge.predict(test_feats))
    print(f"Ridge RMSE (full forge, alpha={args.alpha}): {ridge_rmse:.6f}")

    swr_test_rmse = None
    if not args.quick:
        print("\n--- Online ridge (sliding-window dual, M=500, anchored) ---")
        swr = SlidingWindowRidge(alpha=args.alpha, window=500,
                                 refit_every=25).fit(train_feats, train_y)
        swr_preds = []
        for i in tqdm(range(len(test_feats)), desc="Online ridge"):
            swr_preds.append(swr.predict(test_feats[i]))
            swr.update(test_feats[i], test_y[i])   # label known after use
        swr_test_rmse = rmse(test_y, swr_preds)
        print(f"Online ridge test RMSE (convex, sliding-window): "
              f"{swr_test_rmse:.6f}")

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS - Mackey-Glass tau=17, "
          f"{args.horizon}-step forecast")
    print("=" * 60)
    print(f"{'Linear AR(10)':<25} {lr_rmse:>10.6f}")
    print(f"{'GRU (32 units)':<25} {gru_rmse:>10.6f}")
    print(f"{'ESN (200 neurons)':<25} {esn_rmse:>10.6f}")
    print(f"{'Ghost-Lattice Ridge':<25} {ridge_rmse:>10.6f}")
    if swr_test_rmse is not None:
        print(f"{'Ghost-Lattice Online ridge':<25} {swr_test_rmse:>10.6f}")
    print("=" * 60)
    print("\nGhost-Lattice QNN - Victus run complete.")


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    # torch + sklearn both load OpenMP runtimes on Windows; DLL unload at
    # interpreter shutdown can segfault after all work is done. Terminate
    # the process directly (bypasses C-runtime DLL detach) for a clean exit.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), rc)
    os._exit(rc)
