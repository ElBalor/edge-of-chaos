"""Emit a reproducibility record for a trained TSPS adapter artifact.

This deliberately separates protocol/latency evidence from an official score.
An official score is copied in only when the pinned upstream evaluator has
already produced a JSON file; the script never invents one.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from tsps_integration.ghost_lattice_tsps import TSPSGhostLatticeModel, TSPS_UPSTREAM_REVISION


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tsps-repo", type=Path)
    ap.add_argument("--official-score-json", type=Path)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")

    model = TSPSGhostLatticeModel.load(args.artifact)
    window = np.sin(np.linspace(0, 20, 500, dtype=np.float32))[:, None]
    times: list[float] = []
    prediction = None
    for _ in range(args.runs):
        start = time.perf_counter()
        prediction = model.predict(window)
        times.append(time.perf_counter() - start)
    assert prediction is not None
    payload = {
        "kind": "tsps_adapter_evidence",
        "official_score_status": "not_provided",
        "tsps_upstream_revision": TSPS_UPSTREAM_REVISION,
        "tsps_repo_head": _git_head(args.tsps_repo) if args.tsps_repo else None,
        "artifact": str(Path(args.artifact).resolve()),
        "prediction_shape": list(prediction.shape),
        "finite_prediction": bool(np.isfinite(prediction).all()),
        "runs": args.runs,
        "latency_seconds": times,
        "latency_mean_seconds": float(np.mean(times)),
        "latency_p95_seconds": float(np.percentile(times, 95)),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if args.official_score_json:
        score_path = args.official_score_json.resolve()
        payload["official_score_status"] = "copied_from_pinned_evaluator"
        payload["official_score_file"] = str(score_path)
        payload["official_score"] = json.loads(score_path.read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
