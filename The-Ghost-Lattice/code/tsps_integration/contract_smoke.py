"""Fast local checks for the TSPS adapter boundary.

This uses a tiny deterministic artifact generated in memory. It validates the
wire contract only; it does not claim benchmark quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghost_lattice.core import (  # noqa: E402
    RheonTauNet,
    init_reservoir,
    polynomial_forge,
    rheon_gate,
    simulate_reservoir,
)
from tsps_integration.ghost_lattice_tsps import TSPSGhostLatticeModel  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    gate = RheonTauNet(input_dim=4)
    reservoirs = [init_reservoir(42 + r, input_dim=1) for r in range(3)]
    window = np.sin(np.linspace(0, 12, 500, dtype=np.float32))[:, None]
    # The production artifact has a trained readout. Here we only size-check
    # the full forge and attach a deterministic ten-output projection.
    filtered = rheon_gate(window[:, 0], gate)
    phi = np.concatenate([
        polynomial_forge(simulate_reservoir(filtered, J, h, W))
        for J, h, W in reservoirs
    ])
    weights = rng.normal(0, 1e-4, size=(phi.size, 10)).astype(np.float64)
    model = TSPSGhostLatticeModel(gate, reservoirs, np.zeros((1, 1)), np.ones((1, 1)), weights, 0.0)
    pred = model.predict(window)
    assert pred.shape == (100, 1)
    assert np.isfinite(pred).all()
    h1 = model.hash_predictions("test-hotkey", pred[:, 0])
    h2 = model.hash_predictions("test-hotkey", pred[:, 0])
    assert h1 == h2 and len(h1) == 64
    print(f"TSPS contract smoke PASS: forge={phi.size} output={pred.shape} hash={h1}")


if __name__ == "__main__":
    main()
