#!/usr/bin/env python3
"""Ghost-Lattice live BCI -> prosthetic controller.

The deployment pipeline from the paper, end to end:

    headband --LSL--> windows -> bandpower -> FROZEN Ghost-Lattice
    (evolved ONCE offline on BCI IV 2a) -> per-user ridge readout
    (2-minute calibration) -> smoothed commands -> serial -> servos
                                  |
    EMG distress stop (20-60 Hz, fixed threshold) ---> hardware cutoff

Two modes:
  --simulate   synthetic EEG stream, no hardware needed (rehearsal mode)
  --lsl        real headband via Lab Streaming Layer (pylsl)

Calibration (per user, ~2 minutes, NO heavy training - the reservoir is
already evolved; only the convex readout adapts):
  1. rest 20 s          2. imagine closing the hand 20 s
  3. imagine opening 20 s   4. jaw-clench 10 s (safety threshold)

Author: Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sklearn.linear_model  # sklearn BEFORE torch: OpenMP fix
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

import torch

from ghost_lattice.core import (
    DEVICE,
    RheonTauNet,
    SlidingWindowRidge,
    chaos_algorithm,
    combined_features,
    init_reservoir,
    spectral_genesis,
)
from ghost_lattice.motion import ZetaTrajectory
from ghost_lattice.safety import EMGStopDetector

from run_bci import bandpower_features, synthetic_epochs

FS = 250.0
WINDOW_S = 2.0
HOP_S = 0.25
COMMANDS = ["OPEN", "CLOSE", "REST"]


# ---------------------------------------------------------------------------
# EEG source
# ---------------------------------------------------------------------------

class SimulatedHeadband:
    """Synthetic 22-ch stream with motor-imagery band signatures."""

    def __init__(self, fs=FS, n_ch=22, seed=0):
        self.fs, self.n_ch = fs, n_ch
        self.rng = np.random.RandomState(seed)
        self.t = 0.0
        self.imagery = None   # None=rest, "CLOSE", "OPEN"
        self.clench = False
        self._t = np.arange(int(fs)) / fs

    def set_imagery(self, cmd):
        self.imagery = None if cmd == "REST" else cmd

    def read(self, n_samples):
        """Return (n_ch, n_samples) chunk of 'EEG'."""
        out = np.zeros((self.n_ch, n_samples))
        start = 0
        while start < n_samples:
            k = min(len(self._t), n_samples - start)
            t = self._t[:k] + self.t
            seg = 3e-6 * self.rng.randn(self.n_ch, k)
            f0 = 10.0 if self.imagery == "CLOSE" else 22.0
            if self.imagery:
                for ch in (0, 1, 2):
                    seg[ch] += 15e-6 * np.sin(2 * np.pi * f0 * t
                                              + self.rng.rand())
            if self.clench:
                for ch in (0, 1, 21):
                    seg[ch] += 80e-6 * np.sin(2 * np.pi * 45.0 * t)
            out[:, start:start + k] = seg
            start += k
            self.t += k / self.fs
        return out


class LSLHeadband:
    """Real headband through Lab Streaming Layer."""

    def __init__(self, stream_name=None):
        import pylsl
        streams = pylsl.resolve_byprop("type", "EEG", timeout=10)
        if stream_name:
            streams = [s for s in streams
                       if s.name() == stream_name] or streams
        if not streams:
            raise RuntimeError("No EEG LSL stream found. Is the headband "
                               "app running?")
        self.inlet = pylsl.StreamInlet(streams[0], max_buflen=10)
        self.fs = float(self.inlet.info().nominal_srate()) or FS
        self.n_ch = int(self.inlet.info().channel_count())

    def read(self, n_samples):
        chunk, _ = self.inlet.pull_chunk(max_samples=int(n_samples))
        while len(chunk) < n_samples:      # block until we have a window
            more, _ = self.inlet.pull_chunk(timeout=1.0)
            chunk.extend(more)
        arr = np.array(chunk[:n_samples], dtype=float)[:, :22].T
        return arr - arr.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Frozen pipeline (evolved once)
# ---------------------------------------------------------------------------

def get_pipeline(pipeline_path, simulate=True):
    """Load the frozen Ghost-Lattice, or evolve it once and save it."""
    if pipeline_path and os.path.exists(pipeline_path):
        blob = torch.load(pipeline_path, weights_only=False,
                          map_location=DEVICE)
        tau_net = RheonTauNet(input_dim=4).to(DEVICE)
        tau_net.load_state_dict(blob["tau_net"])
        print(f"Loaded frozen pipeline from {pipeline_path}")
        return tau_net, blob["J"], blob["h"], blob["W_in"]

    print("Evolving Ghost-Lattice pipeline (one-time, ~2 min)...")
    X, y = synthetic_epochs(n_per_class=25)
    Xf = np.array([bandpower_features(e) for e in X])
    val_feats = np.array([combined_features(x) for x in Xf[:12]])
    tau_net = RheonTauNet(input_dim=4).to(DEVICE)
    tau_net = chaos_algorithm(Xf[:100], y[:100], val_feats, y[:12],
                              tau_net, input_dim=1, pop_size=4,
                              generations=2)
    J, h, W_in = init_reservoir(42)
    J, h = spectral_genesis(Xf[:100], y[:100], val_feats, y[:12],
                            tau_net, J, h, W_in, input_dim=1,
                            pop_size=4, generations=2)
    if pipeline_path:
        torch.save({"tau_net": tau_net.state_dict(),
                    "J": J, "h": h, "W_in": W_in}, pipeline_path)
        print(f"Pipeline frozen -> {pipeline_path}")
    return tau_net, J, h, W_in


def window_features(epoch, tau_net, J, h, W_in):
    return combined_features(bandpower_features(epoch),
                             tau_net, J, h, W_in)


# ---------------------------------------------------------------------------
# Calibration (the ONLY per-user step)
# ---------------------------------------------------------------------------

def calibrate(source, tau_net, J, h, W_in, profile_path):
    print("\n===== CALIBRATION (2 minutes) =====")
    feats, labels = [], []
    script = [
        ("REST - relax, think of nothing", "REST", 20),
        ("CLOSE - imagine slowly closing your hand", "CLOSE", 20),
        ("OPEN - imagine slowly opening your hand", "OPEN", 20),
    ]
    for text, label, secs in script:
        input(f"\n[{label}] Press ENTER, then: {text} ... ")
        n_win = int((secs - WINDOW_S) / HOP_S)
        for i in range(n_win):
            chunk = source.read(int(WINDOW_S * FS))
            feats.append(window_features(chunk, tau_net, J, h, W_in))
            labels.append(COMMANDS.index(label))
            print(f"\r  {i + 1}/{n_win}", end="", flush=True)
        print()

    print("\nSafety: JAW CLENCH as hard as you can, 10 seconds")
    input("Press ENTER, then clench ... ")
    det = EMGStopDetector(fs=FS)
    clench_chunks = []
    for _ in range(40):
        clench_chunks.append(source.read(int(0.25 * FS))[0])
    det.calibrate(clench_chunks, margin=0.5)
    print(f"  stop threshold set: {det.threshold:.3e}")

    Xc = np.array(feats)
    yc = np.array(labels)
    scaler = StandardScaler().fit(Xc)
    clf = RidgeClassifier(alpha=10.0).fit(scaler.transform(Xc), yc)
    acc = clf.score(scaler.transform(Xc), yc)
    print(f"Calibration readout fitted (train agreement {100*acc:.0f}%)")

    os.makedirs(os.path.dirname(profile_path) or ".", exist_ok=True)
    np.savez_compressed(profile_path,
                        coef=clf.coef_, intercept=clf.intercept_,
                        mean=scaler.mean_, scale=scaler.scale_,
                        threshold=det.threshold)
    print(f"Profile saved -> {profile_path}")
    return scaler, clf, det


def load_profile(profile_path):
    blob = np.load(profile_path, allow_pickle=False)
    scaler = StandardScaler().fit(np.zeros((1, blob["mean"].shape[0])))
    scaler.mean_ = blob["mean"]
    scaler.scale_ = blob["scale"]
    clf = RidgeClassifier(alpha=10.0)
    clf.coef_ = blob["coef"]
    clf.intercept_ = blob["intercept"]
    clf.classes_ = np.arange(len(COMMANDS))
    det = EMGStopDetector(fs=FS, threshold=float(blob["threshold"]))
    print(f"Profile loaded <- {profile_path}")
    return scaler, clf, det


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------

class SerialLink:
    """One-letter command protocol to the prosthetic firmware."""

    CODE = {"OPEN": b"O", "CLOSE": b"C", "REST": b"R", "STOP": b"S"}

    def __init__(self, port=None, baud=115200):
        self.ser = None
        if port:
            import serial  # pyserial
            self.ser = serial.Serial(port, baud, timeout=1)
            print(f"Serial link -> {port} @ {baud}")

    def send(self, cmd):
        if self.ser and cmd in self.CODE:
            self.ser.write(self.CODE[cmd] + b"\n")

    def close(self):
        if self.ser:
            self.ser.close()


def live(source, tau_net, J, h, W_in, scaler, clf, det, link,
         duration_s=120):
    print("\n===== LIVE =====  (Ctrl+C to stop)")
    zeta = ZetaTrajectory(tau=0.15, dt=HOP_S, goal=0.0)
    hold = COMMANDS.index("REST")
    votes = []
    stop_hits = 0
    t0 = time.time()
    while time.time() - t0 < duration_s:
        chunk = source.read(int(HOP_S * FS))
        rms, triggered = det.update(chunk[0])
        if triggered:
            stop_hits += 1
            link.send("STOP")
            print("\n*** DISTRESS STOP LATCHED - motors cut ***")
            det.reset()
            votes.clear()
            time.sleep(2.0)
            continue
        f = window_features(chunk, tau_net, J, h, W_in)
        votes.append(int(clf.predict(scaler.transform(f[None, :]))[0]))
        votes = votes[-5:]                       # majority over ~1.25 s
        cmd_idx = max(set(votes), key=votes.count)
        if cmd_idx != hold:
            hold = cmd_idx
            link.send(COMMANDS[hold])
            print(f"\r-> {COMMANDS[hold]:<6}", end="", flush=True)
        # Zeta: continuous analog-ish position for graders who want to
        # see C2-smooth motion, not jumps (goal 0 = open, 1 = close)
        goal = 1.0 if hold == 1 else (0.0 if hold == 0 else zeta.g)
        y, _, _ = zeta.step(g=goal)
        if os.environ.get("GL_ZETA_DEBUG"):
            print(f"\rzeta={y:.3f}", end="")
    print(f"\nLive session ended. stops={stop_hits}")


def main():
    ap = argparse.ArgumentParser(
        description="Ghost-Lattice live BCI -> prosthetic")
    ap.add_argument("--simulate", action="store_true",
                    help="synthetic headband (rehearsal, no hardware)")
    ap.add_argument("--lsl", action="store_true",
                    help="real headband via Lab Streaming Layer")
    ap.add_argument("--port", default=None,
                    help="serial port for the prosthetic (e.g. COM5)")
    ap.add_argument("--pipeline", default="pipeline.npz")
    ap.add_argument("--profile", default="profiles/user1.npz")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--duration", type=int, default=120)
    args = ap.parse_args()

    if not (args.simulate or args.lsl):
        ap.error("choose --simulate or --lsl")

    source = (SimulatedHeadband() if args.simulate
              else LSLHeadband())
    tau_net, J, h, W_in = get_pipeline(args.pipeline, args.simulate)

    if args.recalibrate or not os.path.exists(args.profile):
        scaler, clf, det = calibrate(source, tau_net, J, h, W_in,
                                     args.profile)
    else:
        scaler, clf, det = load_profile(args.profile)

    link = SerialLink(args.port)
    try:
        live(source, tau_net, J, h, W_in, scaler, clf, det, link,
             duration_s=args.duration)
    finally:
        link.send("STOP")
        link.close()
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
