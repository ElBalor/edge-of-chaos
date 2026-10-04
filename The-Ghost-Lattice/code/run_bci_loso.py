#!/usr/bin/env python3
"""Ghost-Lattice QNN - BCI Competition IV 2a, leave-one-subject-out.

Proper LOSO protocol for the paper's BCI claim: train on 8 subjects,
test on the held-out 9th, rotate. Bandpower features feed the frozen
Ghost-Lattice pipeline (LNN gate + Ising reservoirs + forge + ridge
classifier); per-reservoir feature selection is fitted on training
subjects only.

Dataset (BCI Competition IV 2a, 9 subjects, 22-channel EEG):
  1. Download from http://www.bbci.de/competition/iv/#dataset2a
     (registration required) - the .gdf files A01T..A09T (+ E sets).
  2. Place them in a directory, e.g.  data/bci2a/
  3. pip install mne   (optional dependency, loaders use it)
  4. Run:
       python run_bci_loso.py --data-dir data/bci2a
     or, to smoke-test the full protocol on synthetic multi-subject
     data without the dataset:
       python run_bci_loso.py --demo

Synthetic demo mode generates 4 "subjects" whose band-power signatures
share a subject-independent structure plus per-subject offsets - hard
enough that chance (25%) is far below what a working cross-subject
pipeline scores.

Author: Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0
"""

import argparse
import os
import sys

import numpy as np
import sklearn.linear_model  # sklearn BEFORE torch: OpenMP fix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch

from ghost_lattice.core import (
    DEVICE,
    SPATIAL_R,
    TauNet,
    chaos_algorithm,
    combined_features,
    init_reservoir,
    per_reservoir_features,
    spectral_genesis,
)

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 30)]  # theta..beta


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def bandpower_features(epoch, fs=250.0):
    """Per-channel band-power vector for one EEG epoch."""
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
# Data loading
# ---------------------------------------------------------------------------

def load_bci2a_subjects(data_dir):
    """Load all A*T.gdf training files; returns list of (X, y) per subject.

    X: (n_trials, n_channels, n_samples) epochs; y: labels 0..3.
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, "A*T.gdf")))
    if not files:
        return None
    import mne  # optional dependency
    subjects = []
    for path in files:
        raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
        events, _ = mne.events_from_annotations(raw, verbose="ERROR")
        ep = mne.Epochs(raw, events, event_id=[7, 8, 9, 10],
                        tmin=0.5, tmax=3.0, picks=list(range(22)),
                        baseline=None, verbose="ERROR")
        subjects.append((ep.get_data(), ep.events[:, 2] - 7))
    return subjects


def synthetic_subjects(n_subjects=4, n_per_class=40, n_ch=22, fs=250.0,
                       t_s=2.5, seed=0):
    """Synthetic multi-subject MI-like data with shared + per-subject
    band-power structure (for protocol smoke-testing without the real
    dataset)."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(t_s * fs)) / fs
    shared = {0: 1.6, 2: 1.5}      # class signature: theta + beta (motor)
    class_dirs = [{1: 0.9}, {3: 1.1}, {1: -0.4, 3: 0.5}, {2: 0.8}]
    subjects = []
    for s in range(n_subjects):
        offset = {b: rng.uniform(-0.5, 0.5) for b in range(len(BANDS))}
        X, y = [], []
        for label in range(4):
            for _ in range(n_per_class):
                epoch = 3e-6 * rng.randn(n_ch, len(t))
                for ch in (0, 1, 2):
                    for bi, amp in {**shared, **class_dirs[label],
                                    **offset}.items():
                        lo, hi = BANDS[bi]
                        f0 = (lo + hi) / 2
                        phase = rng.rand() * 2 * np.pi
                        epoch[ch] += (amp * 4e-6
                                      * np.sin(2 * np.pi * f0 * t + phase))
                X.append(epoch)
                y.append(label)
        subjects.append((X, np.array(y)))
    return subjects


# ---------------------------------------------------------------------------
# Ghost-Lattice pipeline (frozen once per run)
# ---------------------------------------------------------------------------

def evolve_pipeline(X_train, y_train, quick=False, gens=None):
    """Evolve gate + reservoir on training-subject features."""
    pop = 2 if quick else 8
    if gens is None:
        gens = 1 if quick else 5
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X_train))
    sub = idx[:min(150, len(idx))]
    Xg, yg = X_train[sub], y_train[sub]
    val_feats = np.array([combined_features(x) for x in Xg[:40]])

    print("  Chaos Algorithm...")
    tau_net = TauNet(input_dim=4).to(DEVICE)
    tau_net = chaos_algorithm(Xg, yg, val_feats, yg[:40], tau_net,
                              input_dim=1, pop_size=pop, generations=gens)
    print("  Spectral Genesis...")
    J, h, W_in = init_reservoir(42)
    J, h = spectral_genesis(Xg, yg, val_feats, yg[:40], tau_net, J, h,
                            W_in, input_dim=1, pop_size=pop,
                            generations=gens)
    return tau_net, J, h, W_in


def features(X, tau_net, J, h, W_in, desc="feats"):
    blocks = [np.array([per_reservoir_features(x, tau_net, J, h, W_in)[r]
                        for x in tqdm(X, desc=desc, leave=False)])
              for r in range(SPATIAL_R)]
    return blocks


def loso(subjects, k_per_res=500, quick=False, gens=None):
    """Leave-one-subject-out cross-validation."""
    accs = []
    for test_i in range(len(subjects)):
        train = [s for i, s in enumerate(subjects) if i != test_i]
        X_tr_ep = np.concatenate([s[0] for s in train])
        y_tr = np.concatenate([s[1] for s in train])
        X_te_ep, y_te = subjects[test_i]

        X_tr = np.array([bandpower_features(e) for e in X_tr_ep])
        X_te_ep = X_te_ep[:144] if len(X_te_ep) > 144 else X_te_ep
        X_te = np.array([bandpower_features(e) for e in X_te_ep])

        print(f"Subject {test_i + 1}/{len(subjects)}: "
              f"train={len(X_tr)} test={len(X_te)}")

        tau_net, J, h, W_in = evolve_pipeline(X_tr, y_tr, quick, gens)

        tr_blocks = features(X_tr, tau_net, J, h, W_in,
                             desc=f"train S{test_i + 1}")
        te_blocks = features(X_te, tau_net, J, h, W_in,
                             desc=f"test  S{test_i + 1}")

        sel_tr, sel_te = [], []
        k = min(k_per_res, tr_blocks[0].shape[1])
        for tr_b, te_b in zip(tr_blocks, te_blocks):
            sc = StandardScaler()
            tr_s = sc.fit_transform(tr_b)
            te_s = sc.transform(te_b)
            sel = SelectKBest(f_classif, k=k)
            sel_tr.append(sel.fit_transform(tr_s, y_tr))
            sel_te.append(sel.transform(te_s))
        Xtr = np.hstack(sel_tr)
        Xte = np.hstack(sel_te)

        clf = RidgeClassifier(alpha=10.0).fit(Xtr, y_tr)
        acc = accuracy_score(y_te, clf.predict(Xte))
        accs.append(acc)
        print(f"  -> accuracy {100 * acc:.2f}%")

    return accs


def main():
    ap = argparse.ArgumentParser(
        description="Ghost-Lattice QNN - BCI IV 2a LOSO evaluation")
    ap.add_argument("--data-dir", default=None,
                    help="directory with A*T.gdf files (needs mne)")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic multi-subject protocol selftest")
    ap.add_argument("--k-per-res", type=int, default=500)
    ap.add_argument("--generations", type=int, default=None,
                    help="GA generations (paper protocol: 15)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny GA + subset of trials (smoke test)")
    args = ap.parse_args()

    if args.data_dir and not args.demo:
        subjects = load_bci2a_subjects(args.data_dir)
        if subjects is None:
            print(f"No A*T.gdf files in {args.data_dir}; "
                  f"use --demo for the synthetic protocol selftest.")
            return 1
        mode = "2a"
    else:
        subjects = synthetic_subjects(n_subjects=4,
                                      n_per_class=8 if args.quick else 40)
        mode = "synthetic"

    print(f"Mode: {mode} | subjects: {len(subjects)} | "
          f"device: {DEVICE}")
    accs = loso(subjects, k_per_res=args.k_per_res, quick=args.quick,
                gens=args.generations)
    print("\n" + "=" * 60)
    print(f"LOSO RESULTS ({mode}) - mean over "
          f"{len(accs)} held-out subjects")
    print("=" * 60)
    print(f"per-subject: {[round(100 * a, 1) for a in accs]}")
    print(f"mean accuracy: {100 * np.mean(accs):.2f}%  (chance 25%)")
    print("BCI LOSO complete.")
    return 0


if __name__ == "__main__":
    rc = 0
    try:
        rc = main() or 0
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
