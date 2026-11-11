# tsps-official-5k — what is here and how to read it

Official-history evaluation of the Ghost-Lattice TSPS miner against the pinned
contract revision `5660ff464526daaf2fe2d63e6d0a5ad1416071f5`.

## The headline number

`official_score_full.json`, produced by the pinned upstream
`Scoring.calculate_weighted_rmse` at `k=0.1` over 1,000 held-out windows:

| Metric | Value |
|---|---|
| Mean weighted RMSE | `0.10816238500768612` |
| Median weighted RMSE | `0.09327470841174158` |
| Sparse RMSE (k=0.1) | `0.24412631717736794` |
| Latency mean / p95 / max | `0.0602 s` / `0.1122 s` / `0.3556 s` |

This is an **offline** holdout score. It is not live validator rank, not
emissions, and not a claim of state of the art.

## Three numbers that will look inconsistent, and why they are not

A reviewer comparing these files will find figures that do not match. Each one
measures a different thing:

| Where | Value | What it actually is |
|---|---|---|
| `official_score_full.json` | sparse RMSE `0.2441` | Official evaluator, 1,000-window temporal holdout, `k=0.1` |
| `model.json` → `validation_rmse_sparse` | `0.2793` | Training-time validation split (4,000 train / 1,000 validation), a **different and harsher** split than the official holdout |
| `evidence.json` → `latency_mean_seconds` | `0.1311 s` (p95 `0.2051 s`) | Three adapter smoke runs including serialization overhead, not the 1,000-window scoring loop |
| `official_score_full.json` → latency | `0.0602 s` (p95 `0.1122 s`) | Per-window inference inside the official scoring loop |

`0.2793` is the honest training-split number; `0.2441` is the official holdout
number. They are reported separately on purpose rather than quoting the better
one alone.

## Files that are published, and files that are not

Published here so the score can be inspected:

- `official_score_full.json` — the official evaluator output
- `official_score.json` — the abbreviated copy of the same run
- `evidence.json` — machine-readable protocol and latency record
- `protocol_smoke.log` — hash/reveal contract smoke output
- `model.json` — model config and training-split metrics
- `release_manifest.json` — SHA-256 for every artifact, plus the `safety` block

Deliberately **not** in this public repo, because they are trained weights on a
competitive subnet:

- `gate.pt`
- `model.npz`
- `ghost-lattice-tsps-release.zip` (bundle containing both)

Their SHA-256 values are published in `release_manifest.json`, so a file
received privately can be verified byte-for-byte against this record. Send a
request to `heylelyaka@gmail.com` and the weights and the exact command that
produced `official_score_full.json` go out directly.

## A note on the paths inside the JSON files

`official_score_full.json`, `evidence.json` and `model.json` contain absolute
paths from the machine the run happened on (`C:\Users\...\The-Ghost-Lattice\...`).
They were left byte-identical on purpose: their SHA-256 values are published and
pinned in `release_manifest.json`, and editing the files to tidy the paths would
break that hash chain. The paths are runner metadata, not results.

## Reproducing

`../tsps_integration/GRANT_BRIEF.md` has the exact commands, including the
pinned-revision checkout, the historical-data regeneration, and the readout
training step. Expect 10–15 minutes on a normal laptop.
