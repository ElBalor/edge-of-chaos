"""Export TSPS historical storage into Ghost-Lattice training windows.

Run this inside the pinned TSPS repository environment. It uses TSPS's own
feature collector and scaler, and exports only the BTC/USD close channel for
the first Ghost-Lattice TSPS readout. The validator still performs the
official feature-source conversion and unscaling around the miner model.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from features import FeatureCollector, FeatureID
from feature_sources import BinaryFileFeatureStorage
from streams.btcusd_5m import (
    INTERVAL_MS,
    SAMPLE_COUNT,
    PREDICTION_COUNT,
    PREDICTION_LENGTH,
    historical_feature_ids,
    model_feature_scaler,
    prediction_feature_ids,
    spontaneous_feature_sources,
)
from time_util import time_span_ms


def sparse_indexes(prediction_length: int, prediction_count: int) -> list[int]:
    if prediction_count > prediction_length:
        raise ValueError("prediction_count cannot exceed prediction_length")
    if prediction_count == 1:
        return [0]
    stride = int(prediction_length / (prediction_count - 1))
    return [i * stride for i in range(prediction_count - 1)] + [prediction_length - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historical", required=True, help="TSPS .taosfs historical file")
    ap.add_argument("--output", required=True, help="output NPZ")
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    storage = BinaryFileFeatureStorage(
        filename=args.historical,
        mode="r",
        feature_ids=historical_feature_ids,
    )
    # Historical rows are already complete TSPS feature rows.  Do not attach
    # live spontaneous sources here: they perform network fetches for every
    # window and can stall an offline/reproducible export.  The official
    # scaler is still applied unchanged below.
    collector = FeatureCollector([storage])
    start_ms = storage.get_start_time_ms()
    total = storage.get_sample_count()
    need = SAMPLE_COUNT + PREDICTION_LENGTH
    if total < need:
        raise ValueError(f"historical file has {total} samples; need at least {need}")

    # Read the historical file once.  Calling FeatureCollector for every
    # overlapping window repeatedly seeks and decodes the same rows and makes
    # a 67k-row official file needlessly slow.
    all_samples = storage.get_feature_samples(start_ms, INTERVAL_MS, total)

    # Prediction target is the TSPS close feature. Input is deliberately one
    # channel so the Ghost-Lattice Rheon scalar gate remains unchanged.
    close_id = prediction_feature_ids[0]
    indexes = sparse_indexes(PREDICTION_LENGTH, PREDICTION_COUNT)
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    max_start = total - need
    for row, offset in enumerate(range(0, max_start + 1, args.stride)):
        if row >= args.max_samples:
            break
        # TSPS fits its FeatureScaler on each request's lookback window.  Fit
        # a fresh copy per exported request so offline training has exactly
        # the same scaling/unscaling semantics as the official miner.
        raw_window = {
            feature_id: np.asarray(values[offset:offset + need])
            for feature_id, values in all_samples.items()
        }
        request_scaler = copy.deepcopy(model_feature_scaler)
        scaled = request_scaler.scale_feature_samples(raw_window)
        close = np.asarray(scaled[close_id], dtype=np.float32)
        x = close[:SAMPLE_COUNT].reshape(SAMPLE_COUNT, 1)
        future = close[SAMPLE_COUNT:need]
        windows.append(x.reshape(SAMPLE_COUNT, 1))
        targets.append(future[indexes])

    if not windows:
        raise RuntimeError("no TSPS windows were exported")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, windows=np.stack(windows), targets=np.stack(targets))
    out.with_suffix(".json").write_text(json.dumps({
        "source": str(args.historical),
        "samples": len(windows),
        "windows_shape": list(np.stack(windows).shape),
        "targets_shape": list(np.stack(targets).shape),
        "interval_ms": INTERVAL_MS,
        "sample_count": SAMPLE_COUNT,
        "prediction_count": PREDICTION_COUNT,
        "prediction_length": PREDICTION_LENGTH,
        "stride": args.stride,
        "scaled_by": "TSPS model_feature_scaler",
    }, indent=2), encoding="utf-8")
    storage.close()
    print(f"exported {len(windows)} TSPS windows -> {out}")


if __name__ == "__main__":
    main()
