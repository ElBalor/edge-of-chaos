#!/usr/bin/env python3
"""Exact TFIM vs tanh-surrogate reservoir validation.

The paper claims the mean-field tanh surrogate "captures the essential
spectral properties" of the quantum Ising reservoir. This harness turns
that claim into measurements at qubit counts where BOTH simulations can
run exactly (N = 10..16):

  1. Trajectory correlation - per-qubit Pearson correlation between the
     exact quantum <Z_i>(t) taps and the surrogate spins s_i(t) over the
     same input stream.
  2. Tap-space geometry - correlation between the reservoir Gram
     matrices (how similar the feature spaces the readouts see are).
  3. Downstream parity - identical ridge readouts trained on each
     reservoir's taps to predict a delayed Mackey-Glass value; the
     NMSE ratio is the reservoir-computing equivalence test.

  4. Trotter self-consistency - exact-vs-exact correlation between two
     step sizes (dt = 0.1 vs 0.05) and between Trotter orders 1 and 2,
     demonstrating the quantum simulation itself is converged.

Run:  python validate_quantum.py            (N = 10, 12, 14, 16)
      python validate_quantum.py --quick    (N = 10, 12 only)

Author: Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0
"""

import argparse
import os
import sys

import numpy as np
import sklearn.linear_model  # sklearn BEFORE torch: OpenMP fix
from sklearn.linear_model import Ridge

# torch AFTER sklearn (import-order fix applied inside ghost_lattice too)
import torch

from ghost_lattice.quantum import TFIMReservoir, surrogate_spin_trajectories


def mackey_stream(n, tau=17, beta=0.2, gamma=0.1):
    """A real chaotic driving stream (Mackey-Glass), normalised."""
    x = np.zeros(n + tau)
    x[0] = 1.2
    for t in range(tau, n + tau - 1):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau] ** 10) \
            - gamma * x[t]
    s = x[tau:]
    return (s - s.mean()) / s.std()


def gram_correlation(A, B):
    """Correlation between the Gram matrices of two tap sets."""
    Ga = A @ A.T
    Gb = B @ B.T
    ga, gb = Ga.ravel(), Gb.ravel()
    ga = (ga - ga.mean()) / (ga.std() + 1e-12)
    gb = (gb - gb.mean()) / (gb.std() + 1e-12)
    return float((ga * gb).mean())


def taps_for_surrogate(J, h, W_in, stream, n_steps, read_taps):
    traj = surrogate_spin_trajectories(J, h, W_in, stream, n_steps)
    return traj[read_taps, :]


def downstream_nmse(taps_train, y_train, taps_test, y_test):
    """NMSE of a ridge readout on reservoir taps (frozen gate: none)."""
    mu, sd = taps_train.mean(0), taps_train.std(0) + 1e-9
    Xtr = (taps_train - mu) / sd
    Xte = (taps_test - mu) / sd
    model = Ridge(alpha=1.0).fit(Xtr, y_train)
    pred = model.predict(Xte)
    return float(np.mean((pred - y_test) ** 2) / np.var(y_test))


def run_size(N, n_steps=60, seed=11):
    rng = np.random.RandomState(seed)
    J = (rng.randn(N, N) * 0.1)
    J = (J + J.T) / 2
    h = rng.randn(N) * 0.1
    W_in = rng.randn(N) * 0.1

    stream = mackey_stream(n_steps + 1)[:n_steps]
    # replay the same stream through both reservoirs
    read_taps = sorted(set(int(k) for k in
                           np.linspace(n_steps // 3, n_steps - 1, 5)))

    # ----- exact quantum -----
    q = TFIMReservoir(n_qubits=N, J=J, h=h, W_in=W_in, dt=0.1,
                      order=2).prepare()
    mags_q = q.evolve(stream, n_steps, read_taps=read_taps)

    # ----- surrogate -----
    mags_s = taps_for_surrogate(J, h, W_in, stream, n_steps, read_taps)

    # 1. per-qubit trajectory correlation
    cors = []
    for i in range(N):
        a, b = mags_q[:, i], mags_s[:, i]
        if a.std() > 1e-9 and b.std() > 1e-9:
            cors.append(np.corrcoef(a, b)[0, 1])
        else:
            cors.append(0.0)
    cors = np.array(cors)

    # 2. tap-space geometry
    geo = gram_correlation(mags_q, mags_s)

    # 3. downstream parity on delayed Mackey-Glass prediction
    n_tr = 40
    long_stream = mackey_stream(600)
    feats_q, feats_s, targets = [], [], []
    win, horizon = 30, 3
    for start in range(0, 480, 6):
        seg = long_stream[start:start + win]
        y = long_stream[start + win + horizon]
        fq = TFIMReservoir(n_qubits=N, J=J, h=h, W_in=W_in, dt=0.1,
                           order=2).prepare().evolve(seg, win)
        fs = taps_for_surrogate(J, h, W_in, seg, win,
                                sorted(set(int(k) for k in
                                           np.linspace(win // 3, win - 1,
                                                       5))))
        feats_q.append(fq.ravel())
        feats_s.append(fs.ravel())
        targets.append(y)
    feats_q = np.array(feats_q)
    feats_s = np.array(feats_s)
    targets = np.array(targets)
    nmse_q = downstream_nmse(feats_q[:n_tr], targets[:n_tr],
                             feats_q[n_tr:], targets[n_tr:])
    nmse_s = downstream_nmse(feats_s[:n_tr], targets[:n_tr],
                             feats_s[n_tr:], targets[n_tr:])

    return {"N": N, "corr_mean": float(cors.mean()),
            "corr_std": float(cors.std()), "geometry": geo,
            "nmse_quantum": nmse_q, "nmse_surrogate": nmse_s}


def trotter_convergence(N=12, seed=11):
    """Exact-vs-exact: dt and order convergence of the simulator."""
    rng = np.random.RandomState(seed)
    J = (rng.randn(N, N) * 0.1); J = (J + J.T) / 2
    h = rng.randn(N) * 0.1
    W_in = rng.randn(N) * 0.1
    stream = mackey_stream(61)[:60]
    taps = sorted(set(int(k) for k in np.linspace(20, 59, 5)))

    m_coarse = TFIMReservoir(n_qubits=N, J=J, h=h, W_in=W_in, dt=0.1,
                             order=2).prepare().evolve(stream, 60, taps)
    m_fine = TFIMReservoir(n_qubits=N, J=J, h=h, W_in=W_in, dt=0.05,
                           order=2).prepare().evolve(stream, 60, taps)
    m_o1 = TFIMReservoir(n_qubits=N, J=J, h=h, W_in=W_in, dt=0.05,
                         order=1).prepare().evolve(stream, 60, taps)

    def corr(A, B):
        vals = [np.corrcoef(A[:, i], B[:, i])[0, 1]
                for i in range(N)
                if A[:, i].std() > 1e-9 and B[:, i].std() > 1e-9]
        return float(np.mean(vals))

    return {"dt_convergence": corr(m_coarse, m_fine),
            "order_convergence": corr(m_fine, m_o1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sizes = [10, 12] if args.quick else [10, 12, 14, 16]

    print("=" * 74)
    print("Ghost-Lattice quantum reservoir validation - exact TFIM vs "
      "tanh surrogate")
    print("=" * 74)
    print(f"{'N':>3} {'corr(mean+-std)':>18} {'geometry':>9} "
          f"{'NMSE quantum':>13} {'NMSE surrogate':>15}")
    for N in sizes:
        r = run_size(N)
        print(f"{r['N']:>3} {r['corr_mean']:>9.3f}+-{r['corr_std']:.3f} "
              f"{r['geometry']:>9.3f} {r['nmse_quantum']:>13.4f} "
              f"{r['nmse_surrogate']:>15.4f}")

    tc = trotter_convergence()
    print("\nTrotter self-consistency (N=12):")
    print(f"  dt=0.1 vs dt=0.05 correlation:    "
          f"{tc['dt_convergence']:.4f}")
    print(f"  order-1 vs order-2 correlation:   "
          f"{tc['order_convergence']:.4f}")
    print("\nInterpretation:")
    print("  corr  = per-qubit agreement of quantum <Z_i>(t) with "
          "surrogate s_i(t)")
    print("  geom  = similarity of the tap-space geometry (Gram "
          "correlation)")
    print("  NMSE  = downstream forecasting parity of identical ridge "
          "readouts")
    print("\nVALIDATION_DONE")


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
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.TerminateProcess(k.GetCurrentProcess(), rc)
    os._exit(rc)
