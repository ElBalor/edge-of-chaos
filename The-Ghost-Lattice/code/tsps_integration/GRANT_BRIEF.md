# Ghost-Lattice × TSPS — Reproducible Miner Evidence

## Executive result

I adapted Ghost-Lattice to the pinned Taoshi TSPS contract without changing
its core architecture. The official TSPS historical generator produced 67,680
BTC/USD five-minute rows. I trained the TSPS convex readout on 5,000 temporal
windows (4,000 fit / 1,000 holdout) and evaluated the complete holdout with
the pinned TSPS weighted-RMSE formula.

| Measure | Result |
|---|---:|
| Holdout windows | 1,000 |
| Mean weighted RMSE | **0.1081623850** |
| Median weighted RMSE | **0.0932747084** |
| Mean inference latency | **0.0602 s** |
| p95 inference latency | **0.1122 s** |
| Output contract | **100 future closes** |
| Prediction shape | **(100, 1)** |

The raw machine-readable results are in `artifacts/tsps-official-5k/`:

- `official_score_full.json` — full holdout score
- `evidence.json` — pinned revision, artifact, shape, finite-output and latency evidence
- `model.json`, `model.npz`, `gate.pt` — loadable model artifact
- `protocol_smoke.log` — hash/reveal smoke result

## Architecture preserved

Ghost-Lattice remains backpropagation-free: Rheon continuous-time gate,
29-qubit Ising mean-field reservoir surrogate, Volterra polynomial forge, and
convex readout. The core also contains anchored sliding-window ridge for exact
online adaptation when new labelled observations arrive. The TSPS artifact is
the initial calibrated readout; live validator feedback is the hook used to
activate that online update path.

## Reproduce

Use the pinned TSPS revision:

```text
5660ff464526daaf2fe2d63e6d0a5ad1416071f5
```

```powershell
$env:PYTHONPATH='Edge-Of-Chaos\The-Ghost-Lattice\code'
python Edge-Of-Chaos\The-Ghost-Lattice\code\tsps_integration\export_tsps_windows.py `
  --historical _tmp\tsps_upstream_20261106\runnable\historical_financial_data\data_training.taosfs `
  --output Edge-Of-Chaos\The-Ghost-Lattice\data\tsps-official-5k.npz `
  --max-samples 5000

python Edge-Of-Chaos\The-Ghost-Lattice\code\tsps_integration\train_tsps_readout.py `
  --data Edge-Of-Chaos\The-Ghost-Lattice\data\tsps-official-5k.npz `
  --output Edge-Of-Chaos\The-Ghost-Lattice\artifacts\tsps-official-5k `
  --source-metadata Edge-Of-Chaos\The-Ghost-Lattice\data\tsps-official-5k.json

python Edge-Of-Chaos\The-Ghost-Lattice\code\tsps_integration\evaluate_tsps_artifact.py `
  --artifact Edge-Of-Chaos\The-Ghost-Lattice\artifacts\tsps-official-5k `
  --data Edge-Of-Chaos\The-Ghost-Lattice\data\tsps-official-5k.npz `
  --max-eval 1000 `
  --output Edge-Of-Chaos\The-Ghost-Lattice\artifacts\tsps-official-5k\official_score_full.json

python Edge-Of-Chaos\The-Ghost-Lattice\code\tsps_integration\protocol_smoke.py
```

This evidence demonstrates reproducibility and protocol readiness. Live
validator rank, emissions, registration cost, and grant approval remain
external outcomes and are not fabricated by this report.

*From the Grimoire of Elbàlor, the Digital Necromancer.*
