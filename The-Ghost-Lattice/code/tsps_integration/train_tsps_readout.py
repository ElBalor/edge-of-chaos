"""Fit the TSPS-specific Ghost-Lattice convex readout.

Input is an `.npz` containing:

    windows: (N, 500, 1) BTC/USD close windows
    targets: (N, 10) sparse future-close targets

The upstream TSPS data generator remains the source of truth. This script
only calibrates the Ghost-Lattice readout for that data contract; it does not
backpropagate through the gate or reservoirs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_lattice.core import (
    SPATIAL_R,
    RheonTauNet,
    batch_polynomial_forge,
    batch_rheon_gate,
    batch_reservoir,
    init_reservoir,
)
from tsps_integration.ghost_lattice_tsps import TSPS_UPSTREAM_REVISION


def _features(windows: np.ndarray, gate: RheonTauNet, reservoirs) -> np.ndarray:
    filtered = batch_rheon_gate(windows, gate)
    blocks = []
    for J, h, W in reservoirs:
        taps = batch_reservoir(filtered, J, h, W)
        blocks.append(batch_polynomial_forge(taps))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="NPZ with windows and targets")
    ap.add_argument("--output", required=True, help="artifact directory")
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--source-metadata",
        help="Optional JSON emitted by export_tsps_windows.py; copied into artifact metadata",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    z = np.load(args.data)
    windows = np.asarray(z["windows"], dtype=np.float32)
    targets = np.asarray(z["targets"], dtype=np.float64)
    if windows.ndim != 3 or windows.shape[1:] != (500, 1):
        raise ValueError(f"windows must have shape (N, 500, 1), got {windows.shape}")
    if targets.ndim != 2 or targets.shape != (len(windows), 10):
        raise ValueError(f"targets must have shape (N, 10), got {targets.shape}")
    if not np.isfinite(windows).all() or not np.isfinite(targets).all():
        raise ValueError("non-finite TSPS training data")

    split = max(1, int(len(windows) * 0.8))
    mu = windows[:split].mean(axis=(0, 1), keepdims=True).reshape(1, 1)
    sd = windows[:split].std(axis=(0, 1), keepdims=True).reshape(1, 1)
    sd[sd < 1e-6] = 1.0
    x = ((windows - mu) / sd).astype(np.float32)

    torch.manual_seed(args.seed)
    gate = RheonTauNet(input_dim=4)
    reservoirs = [init_reservoir(args.seed + r, input_dim=1) for r in range(SPATIAL_R)]
    phi = _features(x, gate, reservoirs)
    y_mean = targets[:split].mean(axis=0)
    readout = Ridge(alpha=args.alpha, fit_intercept=False)
    readout.fit(phi[:split], targets[:split] - y_mean)

    pred = phi[split:] @ readout.coef_.T + y_mean if split < len(phi) else np.empty((0, 10))
    rmse = float(np.sqrt(np.mean((pred - targets[split:]) ** 2))) if len(pred) else None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    source_metadata = None
    if args.source_metadata:
        source_path = Path(args.source_metadata)
        source_metadata = {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
            "metadata": json.loads(source_path.read_text(encoding="utf-8")),
        }
    torch.save(gate.state_dict(), out / "gate.pt")
    np.savez(
        out / "model.npz",
        mu=mu,
        sd=sd,
        weights=readout.coef_.T,
        y_mean=y_mean,
        alpha=np.float64(args.alpha),
        **{f"J{r}": J for r, (J, _, _) in enumerate(reservoirs)},
        **{f"h{r}": h for r, (_, h, _) in enumerate(reservoirs)},
        **{f"W{r}": W for r, (_, _, W) in enumerate(reservoirs)},
    )
    (out / "model.json").write_text(json.dumps({
        "model": "ghost-lattice-tsps",
        "upstream_revision": TSPS_UPSTREAM_REVISION,
        "seed": args.seed,
        "input_shape": [500, 1],
        "sparse_output_shape": [10],
        "expanded_output_shape": [100],
        "train_rows": split,
        "validation_rows": len(windows) - split,
        "alpha": args.alpha,
        "validation_rmse_sparse": rmse,
        "source_metadata": source_metadata,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(out), "forge_dim": int(phi.shape[1]), "validation_rmse_sparse": rmse}, indent=2))


if __name__ == "__main__":
    main()
