"""Validator-style local checks for the pinned TSPS prediction contract."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghost_lattice.core import RheonTauNet, init_reservoir, polynomial_forge, rheon_gate, simulate_reservoir  # noqa: E402
from tsps_integration.ghost_lattice_tsps import TSPSGhostLatticeModel  # noqa: E402


def _model() -> TSPSGhostLatticeModel:
    rng = np.random.default_rng(123)
    torch.manual_seed(123)
    gate = RheonTauNet(input_dim=4)
    reservoirs = [init_reservoir(100 + r, input_dim=1) for r in range(3)]
    x = np.sin(np.linspace(0, 30, 500, dtype=np.float32))
    filtered = rheon_gate(x, gate)
    phi = np.concatenate([polynomial_forge(simulate_reservoir(filtered, J, h, W)) for J, h, W in reservoirs])
    return TSPSGhostLatticeModel(
        gate, reservoirs, np.zeros((1, 1)), np.ones((1, 1)),
        rng.normal(0, 1e-4, size=(phi.size, 10)), 0.0,
    )


def main() -> None:
    model = _model()
    window = np.sin(np.linspace(0, 30, 500, dtype=np.float32))[:, None]
    start = time.perf_counter()
    prediction = model.predict(window)
    elapsed = time.perf_counter() - start
    assert prediction.shape == (100, 1)
    assert np.isfinite(prediction).all()

    hotkey = "5TestHotkey"
    committed = model.hash_predictions(hotkey, prediction[:, 0])
    revealed = model.hash_predictions(hotkey, prediction[:, 0].tolist())
    assert committed == revealed

    altered = prediction[:, 0].copy()
    altered[0] = np.nextafter(altered[0], np.inf)
    assert model.hash_predictions(hotkey, altered) != committed

    rejected = False
    try:
        model.predict(np.full((500, 1), np.nan, dtype=np.float32))
    except ValueError:
        rejected = True
    assert rejected
    print(f"TSPS protocol smoke PASS: shape={prediction.shape} hash=reveal-ok latency={elapsed:.3f}s")


if __name__ == "__main__":
    main()
