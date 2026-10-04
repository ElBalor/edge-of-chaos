"""EMG distress-stop detector (paper section "Distress Stop for Telekinesis").

Safety layer for the BCI wheelchair: jaw-clench / masseter EMG artifact
picked up by frontal/temporal electrodes, detected with a 20-60 Hz bandpass,
rolling RMS over a short window, and a FIXED per-user threshold.

Deliberately not learned: a fixed-threshold detector has a failure mode you
can reason about and test exhaustively in five minutes. The software flag
sits IN PARALLEL with the decode pipeline and drives a hardware relay /
MOSFET cutoff whose default fault state is "no power to motors". A physical
panic button is modelled here as a second, independent path to the same
cutoff (HardwareStopSimulator).

Author: Heylel Yaka 
License: CC BY-NC 4.0
"""

from collections import deque

import numpy as np
from scipy.signal import butter, lfilter


def _bandpass(fs, lo=20.0, hi=60.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return b, a


class EMGStopDetector:
    """Streaming EMG-artifact detector -> distress stop flag."""

    def __init__(self, fs=250.0, band=(20.0, 60.0), window_s=0.25,
                 threshold=None, margin=0.5, cooldown_s=1.0):
        self.fs = float(fs)
        self.b, self.a = _bandpass(fs, band[0], band[1])
        self.win_samples = max(1, int(round(window_s * fs)))
        self.buffer = deque(maxlen=self.win_samples)
        self.threshold = threshold
        self.margin = float(margin)
        self.cooldown = int(round(cooldown_s * fs))
        self._cooldown_left = 0
        self.latched = False

    # -- calibration --------------------------------------------------------

    def calibrate(self, clench_samples, margin=None):
        """Set the threshold from recorded clench segments.

        ``clench_samples``: iterable of 1D arrays (raw voltage chunks)
        recorded while the user clenches hard for ~2 s. The threshold is
        placed at ``margin`` (default 0.5) times the mean clench RMS --
        comfortably below the artifact and far above resting RMS.
        """
        rms_vals = [self._rms(np.asarray(chunk, dtype=float))
                    for chunk in clench_samples]
        mean_clench = float(np.mean(rms_vals))
        self.margin = self.margin if margin is None else float(margin)
        self.threshold = mean_clench * self.margin
        return self.threshold

    # -- streaming ----------------------------------------------------------

    def update(self, samples):
        """Feed the newest raw samples; returns (rms, triggered)."""
        x = np.asarray(samples, dtype=float).ravel()
        filtered = lfilter(self.b, self.a, x)
        for v in filtered:
            self.buffer.append(float(v))
        rms = float(np.sqrt(np.mean(np.square(self.buffer)))) \
            if len(self.buffer) else 0.0
        if self._cooldown_left > 0:
            self._cooldown_left -= len(x)
            return rms, False
        triggered = (self.threshold is not None and rms >= self.threshold)
        if triggered:
            self.latched = True
            self._cooldown_left = self.cooldown
        return rms, triggered

    def reset(self):
        """Clear the latch after the hazard is resolved."""
        self.latched = False
        self._cooldown_left = 0
        self.buffer.clear()

    # -- helpers ------------------------------------------------------------

    def _rms(self, x):
        filtered = lfilter(self.b, self.a, x)
        return float(np.sqrt(np.mean(np.square(filtered))))

    def report(self):
        return {"threshold": self.threshold, "margin": self.margin,
                "window_samples": self.win_samples, "latched": self.latched}


class HardwareStopSimulator:
    """Model of the relay/MOSFET cutoff stage.

    Fault state is 'no power': motors are enabled only while the detector
    is clear AND the panic button is unpressed. The software detector and
    the physical button are independent inputs to the same cutoff.
    """

    def __init__(self, detector):
        self.detector = detector
        self.panic_pressed = False

    def press_panic(self):
        self.panic_pressed = True

    def release_panic(self):
        self.panic_pressed = False

    @property
    def motor_power_enabled(self):
        return (not self.panic_pressed) and (not self.detector.latched)


def selftest(fs=250.0, seed=0):
    """End-to-end bench test on synthetic EEG + clench artifact.

    8 s of 10 Hz resting EEG-like noise, then a 0.5 s 45 Hz high-amplitude
    masseter burst (the jaw clench). Returns True if the stop latches
    exactly during the burst and motors cut power.
    """
    rng = np.random.RandomState(seed)
    det = EMGStopDetector(fs=fs, band=(20.0, 60.0), window_s=0.25,
                          margin=0.5)
    t = np.arange(int(8.0 * fs)) / fs

    resting = 5e-6 * np.sin(2 * np.pi * 10.0 * t) \
        + 2e-6 * rng.randn(len(t))
    burst = 80e-6 * np.sin(2 * np.pi * 45.0 * t)
    burst_mask = (t >= 4.0) & (t < 4.5)
    signal = resting + np.where(burst_mask, burst, 0.0)

    # Calibrate on three clench segments (the burst itself, as in setup).
    det.calibrate([signal[burst_mask], signal[burst_mask],
                   signal[burst_mask]], margin=0.5)

    hw = HardwareStopSimulator(det)
    chunk = int(0.05 * fs)
    trigger_times = []
    for start in range(0, len(signal), chunk):
        _, trig = det.update(signal[start:start + chunk])
        if trig:
            trigger_times.append(t[start])
        if not hw.motor_power_enabled:
            break

    fired = len(trigger_times) > 0 and trigger_times[0] >= 3.5
    if not fired:
        det.reset()
        return False
    det.reset()
    return True
