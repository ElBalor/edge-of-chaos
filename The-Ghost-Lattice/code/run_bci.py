#!/usr/bin/env python3
"""Ghost-Lattice QNN - BCI motor-imagery classification (BCI IV 2a style).

4-class motor imagery from multichannel EEG: bandpower features per
channel per band -> Rheon gate -> Ising reservoirs -> forge -> ridge
classifier. Gate and reservoir are evolved by the Chaos Algorithm and
Spectral Genesis with classification accuracy as fitness, with per-reservoir
feature selection keeping 500 features per reservoir (paper Section 5.2).

Run with --demo for a synthetic end-to-end selftest without the dataset.

Author: Heylel Yaka 
License: CC BY-NC 4.0
"""

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# torch AFTER scikit-learn: avoids the Windows/conda OpenMP double-load
# crash (MKL vs torch runtimes) on this machine class.
import torch

from ghost_lattice.core import (
    DEVICE,
    POP_SIZE_CHAOS,
    POP_SIZE_GENESIS,
    GENERATIONS_CHAOS,
    GENERATIONS_GENESIS,
    SPATIAL_R,
    RheonTauNet,
    chaos_algorithm,
    combined_features,
    init_reservoir,
    per_reservoir_features,
    spectral_genesis,
)

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 30)]  # theta..beta
INPUT_DIM = 1   # the gate/reservoir see the bandpower vector as
                # univariate pseudo-time, as in the forecasting runners


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def bandpower_features(epoch, fs=250.0):
    """Per-channel band-power vector for one EEG epoch.

    epoch: (n_channels, n_samples). Welch-style periodogram via FFT
    aggregated over the canonical frequency bands.
    """
    epoch = np.asarray(epoch, dtype=np.float64)
    n_ch, n_s = epoch.shape
    win = np.hanning(n_s)
    spec = np.abs(np.fft.rfft(epoch * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_s, d=1.0 / fs)
    feats = []
    for lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(spec[:, mask].mean(axis=1))
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic data (selftest mode)
# ---------------------------------------------------------------------------

def synthetic_epochs(n_per_class=60, n_ch=22, fs=250.0, t_s=2.5, seed=0):
    """Synthetic 4-class motor-imagery-like dataset.

    Each class is defined by a distinct band-power signature over the
    motor channels (event-related synchronisation/desynchronisation),
    buried in pink-ish noise -- hard enough that chance is 25% and a
    working pipeline must land well above it.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(int(t_s * fs)) / fs
    signatures = [
        {0: 1.8, 1: 0.6},             # class 0: strong theta sync
        {1: 1.6, 2: 1.2},             # class 1: alpha dominant
        {2: 0.4, 3: 1.9},             # class 2: beta ERD pattern
        {0: 0.5, 2: 1.7, 4: 0.9},     # class 3: mixed
    ]
    X, y = [], []
    for label, sig in enumerate(signatures):
        for _ in range(n_per_class):
            epoch = 3e-6 * rng.randn(n_ch, len(t))
            for ch in (0, 1, 2):
                for bi, amp in sig.items():
                    lo, hi = BANDS[bi]
                    f0 = (lo + hi) / 2
                    phase = rng.rand() * 2 * np.pi
                    epoch[ch] += (amp * 4e-6
                                  * np.sin(2 * np.pi * f0 * t + phase))
            X.append(epoch)
            y.append(label)
    return X, np.array(y)


def load_bci2a_gdf(data_dir):
    """Load real BCI Competition IV 2a .gdf files if present.

    Expects train files (e.g. A01T.gdf) in ``data_dir``. Returns
    (epochs, labels) with epochs (n_trials, 22, samples), labels 0..3.
    Requires the optional ``mne`` dependency.
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, "A*T.gdf")))
    if not files:
        return None, None
    import mne  # optional dependency
    epochs, labels = [], []
    for path in files:
        raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
        events, _ = mne.events_from_annotations(raw, verbose="ERROR")
        picks = [i for i in range(22)]
        ep = mne.Epochs(raw, events, event_id=[7, 8, 9, 10],
                        tmin=0.5, tmax=3.0, picks=picks,
                        baseline=None, verbose="ERROR")
        epochs.extend(ep.get_data())
        labels.extend(ep.events[:, 2] - 7)
    return epochs, np.array(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Ghost-Lattice QNN - BCI IV 2a motor imagery")
    ap.add_argument("--data-dir", default=None,
                    help="directory with BCI IV 2a .gdf files "
                         "(requires mne); omit for synthetic selftest")
    ap.add_argument("--k-per-res", type=int, default=500,
                    help="selected features kept per reservoir")
    ap.add_argument("--generations", type=int, default=None,
                    help="override GA generations (paper: 15)")
    ap.add_argument("--demo", action="store_true",
                    help="force synthetic selftest mode")
    ap.add_argument("--quick", action="store_true",
                    help="tiny dataset + 1 GA generation (smoke test)")
    args = ap.parse_args()

    if args.quick:
        gens = 1
    if args.data_dir and not args.demo:
        epochs, labels = load_bci2a_gdf(args.data_dir)
        if epochs is None:
            print(f"No A*T.gdf files found in {args.data_dir}; "
                  f"falling back to synthetic selftest.")
            epochs, labels = synthetic_epochs()
            mode = "synthetic"
        else:
            mode = "2a"
    else:
        epochs, labels = synthetic_epochs(
            n_per_class=15 if args.quick else 60)
        mode = "synthetic"

    fs = 250.0
    X = np.array([bandpower_features(ep, fs) for ep in tqdm(
        epochs, desc="Bandpower features")])
    y = labels
    n_ch = epochs[0].shape[0]
    n_feats_per_ch = len(BANDS)
    print(f"Epochs: {X.shape[0]}, features/ch: {n_feats_per_ch}, "
          f"channels: {n_ch}")

    # split: stratified-ish by class, last 25% of each class to test
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_test = len(X) // 4
    X_tr, y_tr = X[n_test:], y[n_test:]
    X_te, y_te = X[:n_test], y[:n_test]
    n_ga = min(400, len(X_tr))
    X_ga, y_ga = X_tr[:n_ga], y_tr[:n_ga]

    print(f"Device: {DEVICE}")
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}  GA subset: {n_ga}")
    print(f"Chance level: {100.0 / len(np.unique(y)):.0f}%")

    # GA fitness via a quick ridge classifier on combined features
    n_val = 20 if args.quick else 40
    n_ga_fit = 40 if args.quick else 200
    val_feats = np.array([combined_features(x) for x in X_ga[:n_val]])

    gens = args.generations if args.generations else GENERATIONS_CHAOS
    print("\n--- Chaos Algorithm ---")
    tau_net = RheonTauNet(input_dim=4).to(DEVICE)
    tau_net_best = chaos_algorithm(
        X_ga[:n_ga_fit], y_ga[:n_ga_fit], val_feats, y_ga[:n_val], tau_net,
        input_dim=INPUT_DIM, pop_size=2 if args.quick else POP_SIZE_CHAOS,
        generations=gens)

    print("\n--- Spectral Genesis ---")
    J_init, h_init, W_in = init_reservoir(42, input_dim=INPUT_DIM)
    J_best, h_best = spectral_genesis(
        X_ga[:n_ga_fit], y_ga[:n_ga_fit], val_feats, y_ga[:n_val],
        tau_net_best, J_init, h_init, W_in, input_dim=INPUT_DIM,
        pop_size=2 if args.quick else POP_SIZE_GENESIS,
        generations=args.generations if args.generations
        else (1 if args.quick else GENERATIONS_GENESIS))

    print("\n--- Feature selection + ridge classifier ---")
    train_blocks = [
        np.array([per_reservoir_features(x, tau_net_best, J_best, h_best,
                                         W_in,
                                         input_dim=INPUT_DIM)[r]
                  for x in tqdm(X_tr, desc=f"Train Res {r + 1}")])
        for r in range(SPATIAL_R)]
    test_blocks = [
        np.array([per_reservoir_features(x, tau_net_best, J_best, h_best,
                                         W_in,
                                         input_dim=INPUT_DIM)[r]
                  for x in tqdm(X_te, desc=f"Test Res {r + 1}")])
        for r in range(SPATIAL_R)]

    from sklearn.feature_selection import SelectKBest, f_classif
    sel_tr, sel_te = [], []
    k = min(args.k_per_res, train_blocks[0].shape[1])
    for tr_b, te_b in zip(train_blocks, test_blocks):
        sc = StandardScaler()
        tr_s = sc.fit_transform(tr_b)
        te_s = sc.transform(te_b)
        sel = SelectKBest(f_classif, k=k)
        sel_tr.append(sel.fit_transform(tr_s, y_tr))
        sel_te.append(sel.transform(te_s))
    X_tr_sel = np.hstack(sel_tr)
    X_te_sel = np.hstack(sel_te)
    print(f"Selected feature space: {X_tr_sel.shape[1]} "
          f"({args.k_per_res}/reservoir)")

    clf = RidgeClassifier(alpha=10.0).fit(X_tr_sel, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te_sel))
    print(f"\nGhost-Lattice accuracy: {100 * acc:.2f}% "
          f"(chance 25%)")

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS - BCI motor imagery ({mode})")
    print("=" * 60)
    print(f"{'Ghost-Lattice (ridge classifier)':<35} {100 * acc:>9.2f}%")
    print("=" * 60)
    print("\nGhost-Lattice QNN - BCI run complete.")

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
        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), rc)
    os._exit(rc)
