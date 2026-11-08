"""Evaluate a trained TSPS artifact on the held-out official-history slice."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsps_integration.ghost_lattice_tsps import TSPSGhostLatticeModel

def weighted_rmse(pred, actual, k=0.1):
    w = np.exp(-k * np.arange(len(pred)))
    return float(np.sqrt(np.sum(w * (pred - actual) ** 2) / np.sum(w)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-eval', type=int, default=1000)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    z = np.load(args.data)
    windows, sparse_targets = z['windows'], z['targets']
    split = max(1, int(len(windows) * 0.8))
    end = min(len(windows), split + args.max_eval)
    model = TSPSGhostLatticeModel.load(args.artifact)
    sparse_idx = np.linspace(0, 99, 10)
    scores = []
    sparse_rmse = []
    latencies = []
    for window, target in zip(windows[split:end], sparse_targets[split:end]):
        t0 = time.perf_counter()
        pred_sparse = model.predict_sparse(window)
        latencies.append(time.perf_counter() - t0)
        sparse_rmse.append(float(np.sqrt(np.mean((pred_sparse - target) ** 2))))
        pred = np.interp(np.arange(100), sparse_idx, pred_sparse)
        actual = np.interp(np.arange(100), sparse_idx, target)
        scores.append(weighted_rmse(pred, actual))
    result = {
        'artifact': str(Path(args.artifact).resolve()),
        'data': str(Path(args.data).resolve()),
        'official_history_holdout_rows': int(end - split),
        'official_scoring': 'TSPS Scoring.calculate_weighted_rmse, k=0.1',
        'weighted_rmse_mean': float(np.mean(scores)),
        'weighted_rmse_median': float(np.median(scores)),
        'sparse_rmse_mean': float(np.mean(sparse_rmse)),
        'latency_seconds_mean': float(np.mean(latencies)),
        'latency_seconds_p95': float(np.percentile(latencies, 95)),
        'latency_seconds_max': float(np.max(latencies)),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
