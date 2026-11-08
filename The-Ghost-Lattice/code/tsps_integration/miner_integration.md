# Attaching the adapter to the pinned TSPS miner

The adapter is intentionally separate from Ghost-Lattice core. In the pinned
TSPS checkout, the miner integration has three required bindings:

1. Load `TSPSGhostLatticeModel.load(ARTIFACT_DIR)` when selecting the model.
2. Construct the model `FeatureCollector` with only
   `prediction_feature_ids` (BTC/USD close), while retaining TSPS's official
   `model_feature_scaler` for that feature.
3. Keep TSPS's existing `get_predictions`, `live_hash_f`, and `live_f`
   handlers. The adapter's `predict` returns the scaled `(100, 1)` tensor;
   TSPS performs its normal unscale and hash/reveal flow.

Do not replace the handlers with a new protocol. The upstream handlers are
the authority for `LiveForwardHash` and `LiveForward`, including the 60-second
reveal window.

Before connecting a wallet, verify the integration with:

```bash
python code/tsps_integration/contract_smoke.py
python code/tsps_integration/protocol_smoke.py
python runnable/miner_testing.py
```

The benchmark report must include the pinned upstream commit, artifact
metadata, prediction shape, hash/reveal result, per-request latency, and the
official scoring output. A successful smoke test is not a performance claim.

## Official evaluator wiring

In a copy of the pinned `runnable/miner_testing.py`, leave its
`FeatureCollector`, `model_feature_scaler`, `get_predictions`, scoring, and
latency loop unchanged. Replace only the construction of the test model list
with:

```python
from tsps_integration.ghost_lattice_tsps import TSPSGhostLatticeModel

ghost = TSPSGhostLatticeModel.load(
    "/content/drive/MyDrive/Edge-Of-Chaos/The-Ghost-Lattice/"
    "code/tsps_integration/artifacts/tsps-official"
)
# Keep the official sources/scaler, but expose only BTC/USD close to this
# scalar Ghost-Lattice adapter. The upstream evaluator still owns unscaling.
ghost_feature_source = FeatureCollector(
    sources=model_feature_sources,
    feature_ids=prediction_feature_ids,
)
stream_mining_solutions = [{
    "id": "ghost-lattice-tsps",
    "model": ghost,
    "feature_source": ghost_feature_source,
    "feature_scaler": new_model_feature_scaler,
}]
```

The upstream `get_predictions` function supplies the official scaled
500-sample window, calls `ghost.predict`, and performs the official unscaling
and feature conversion. Do not call `predict_sparse` from the evaluator and do
not add a second interpolation or rescaling step. Save evaluator stdout, the
pinned commit, `model.json`, and a separate latency JSON together.
