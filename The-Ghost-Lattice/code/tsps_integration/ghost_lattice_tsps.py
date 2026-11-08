"""Ghost-Lattice model boundary for Taoshi TSPS.

The Ghost-Lattice architecture is kept intact. This module only translates
the TSPS 500-sample BTC/USD window into the existing reservoir/forge/readout
pipeline and translates its ten sparse outputs into TSPS's 100-step output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ghost_lattice.core import (
    SPATIAL_R,
    RheonTauNet,
    polynomial_forge,
    rheon_gate,
    simulate_reservoir,
)


TSPS_SAMPLE_COUNT = 500
TSPS_PREDICTION_COUNT = 10
TSPS_PREDICTION_LENGTH = 100
TSPS_UPSTREAM_REVISION = "5660ff464526daaf2fe2d63e6d0a5ad1416071f5"


class TSPSGhostLatticeModel:
    """Read-only TSPS predictor backed by a Ghost-Lattice artifact.

    `predict` follows the interface used by TSPS's `BaseMiningModel`: it
    accepts a feature matrix whose final 500 rows are the current window and
    returns a `(100, 1)` array of TSPS-scaled close predictions. The upstream
    TSPS `get_predictions` path performs the official feature unscaling after
    this model boundary.
    """

    sample_count = TSPS_SAMPLE_COUNT
    prediction_count = TSPS_PREDICTION_COUNT
    prediction_length = TSPS_PREDICTION_LENGTH

    def __init__(
        self,
        tau_net: RheonTauNet,
        reservoirs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        mu: np.ndarray,
        sd: np.ndarray,
        weights: np.ndarray,
        y_mean: float | np.ndarray,
        device: str = "cpu",
    ) -> None:
        if len(reservoirs) != SPATIAL_R:
            raise ValueError(f"expected {SPATIAL_R} reservoirs")
        self.tau_net = tau_net.eval()
        self.reservoirs = reservoirs
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.y_mean = np.asarray(y_mean, dtype=np.float64).reshape(-1)
        self.device = device
        if self.mu.ndim != 2 or self.mu.shape[0] != 1:
            raise ValueError("TSPS adapter expects one input channel")
        if self.sd.shape != self.mu.shape:
            raise ValueError("mu/sd shape mismatch")
        if self.y_mean.size not in (1, TSPS_PREDICTION_COUNT):
            raise ValueError("y_mean must be scalar or contain ten sparse outputs")
        self.sd = self.sd.copy()
        self.sd[self.sd < 1e-6] = 1.0

    @classmethod
    def load(cls, artifact_dir: str | Path, device: str = "cpu") -> "TSPSGhostLatticeModel":
        root = Path(artifact_dir)
        meta = json.loads((root / "model.json").read_text(encoding="utf-8"))
        if meta.get("upstream_revision") != TSPS_UPSTREAM_REVISION:
            raise ValueError("artifact was not trained against the pinned TSPS revision")
        z = np.load(root / "model.npz")
        tau_net = RheonTauNet(input_dim=4).to(device)
        tau_net.load_state_dict(torch.load(root / "gate.pt", map_location=device))
        reservoirs = [(z[f"J{r}"], z[f"h{r}"], z[f"W{r}"]) for r in range(SPATIAL_R)]
        return cls(
            tau_net=tau_net,
            reservoirs=reservoirs,
            mu=z["mu"],
            sd=z["sd"],
            weights=z["weights"],
            y_mean=z["y_mean"],
            device=device,
        )

    def _features(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] < self.sample_count or x.shape[1] != 1:
            raise ValueError(f"expected at least ({self.sample_count}, 1) input")
        x = x[-self.sample_count:]
        if not np.isfinite(x).all():
            raise ValueError("TSPS input contains non-finite values")
        # TSPS feature scaling is performed before the model boundary. The
        # artifact's mu/sd are retained for artifacts trained with this layer.
        x = ((x - self.mu.reshape(1, -1)) / self.sd.reshape(1, -1)).astype(np.float32)
        filtered = rheon_gate(x[:, 0], self.tau_net)
        blocks = [
            polynomial_forge(simulate_reservoir(filtered, J, h, W))
            for J, h, W in self.reservoirs
        ]
        return np.concatenate(blocks).astype(np.float32)

    def predict_sparse(self, model_input: np.ndarray) -> np.ndarray:
        phi = self._features(model_input)
        if phi.shape[0] != self.weights.shape[0]:
            raise ValueError("artifact feature dimension does not match input")
        sparse = phi.astype(np.float64) @ self.weights
        sparse = np.asarray(sparse, dtype=np.float64).reshape(-1)
        if self.y_mean.size == 1:
            sparse += self.y_mean[0]
        else:
            sparse += self.y_mean
        if sparse.size != TSPS_PREDICTION_COUNT:
            raise ValueError("readout must produce ten sparse predictions")
        return sparse

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        # TSPS expects ten sparse predictions expanded to one hundred points.
        sparse = self.predict_sparse(model_input)
        sparse_idx = np.linspace(0, TSPS_PREDICTION_LENGTH - 1, TSPS_PREDICTION_COUNT)
        full = np.interp(np.arange(TSPS_PREDICTION_LENGTH), sparse_idx, sparse)
        return full.reshape(-1, 1).astype(np.float32)

    @staticmethod
    def hash_predictions(hotkey_address: str, predictions: Iterable[float]) -> str:
        payload = hotkey_address + "-" + str(np.asarray(list(predictions)).tolist())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
