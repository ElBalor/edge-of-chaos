# Ghost-Lattice × TSPS integration status

Updated: 2026-11-06

## Verified locally

- The pinned TSPS source revision is `5660ff464526daaf2fe2d63e6d0a5ad1416071f5`.
- The pinned contract imports with the compatibility namespace and reports
  `SAMPLE_COUNT=500`, `PREDICTION_COUNT=10`, and `PREDICTION_LENGTH=100`.
- `contract_smoke.py` passes the Ghost-Lattice adapter boundary: finite
  `(100, 1)` output, 33,060-feature forge, and deterministic prediction hash.
- `protocol_smoke.py` passes the hash/reveal contract, altered-prediction hash
  rejection, finite-output checks, and reports local adapter latency.
- A synthetic artifact was trained only to exercise serialization/loading. Its
  metrics are wiring checks, not a TSPS performance claim.
- `make_evidence.py` now emits a machine-readable protocol/latency record and
  refuses to invent an official score; a score is included only when copied
  from the pinned evaluator's JSON output.

## Official-history score

`artifacts/tsps-official-5k/official_score_full.json` records the pinned TSPS
`Scoring.calculate_weighted_rmse` formula (`k=0.1`) over 1,000 held-out windows:
mean weighted RMSE `0.1081623850`, median `0.0932747084`, mean inference
latency `0.0602 s`, p95 `0.1122 s` over the full 1,000-window holdout. This is an offline official-history score,
not a claim about live validator rank.

The pinned generator completed through `run_official_history_proxy.py`, a
transport-only shim that preserves the upstream source and `.taosfs` writer.
The raw file and run log remain under `_tmp/tsps_upstream_20261106` and
`_tmp/tsps_official_history_run8.log`.

## Reproducible completion path

For a live subnet submission, run the pinned upstream
`runnable/miner_testing.py` with the adapter wired into `get_predictions`; the
offline score above is the reproducible local gate before testnet deployment.
Preserve its raw output alongside `model.json`, the pinned commit, and the
latency report.
