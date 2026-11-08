# Ghost-Lattice × TSPS integration

This directory is an adapter layer. It does not modify `ghost_lattice.core`.

Pinned upstream TSPS revision:

```text
5660ff464526daaf2fe2d63e6d0a5ad1416071f5
```

The pinned TSPS contract is:

- mainnet netuid: `8`
- testnet netuid: `3`
- input window: `500` samples at five-minute cadence
- sparse output: `10` values
- expanded output: `100` future closes
- target: BTC/USD close
- protocol: `LiveForwardHash`, followed by `LiveForward`

`ghost_lattice_tsps.py` implements the model interface expected by the TSPS
miner (`predict`, `sample_count`, `prediction_count`, `prediction_length`).
The model artifact is trained specifically on TSPS BTC/USD windows. The
Numerai artifact in `code/live_state` is intentionally not reused: it has a
different feature schema and is not a valid TSPS checkpoint.

## Verified official-history result

The current reproducible artifact is `artifacts/tsps-official-5k`. Its full
1,000-window temporal holdout scored mean weighted RMSE `0.1081623850` and
median `0.0932747084` using the pinned TSPS scoring formula. Mean inference
latency was `0.0602 s` and p95 was `0.1122 s`. See `GRANT_BRIEF.md` for the
complete evidence table and reproduction commands.

## Artifact format

The adapter expects a directory containing:

```text
gate.pt
model.npz
model.json
```

`model.npz` stores `mu`, `sd`, `weights`, `y_mean`, and `J0..J2`, `h0..h2`,
`W0..W2`. `model.json` records the data schema, upstream revision, seed, and
training split. Pass the exporter JSON with `--source-metadata` so the artifact
also records the source-file hash and temporal export metadata. The artifact
must be produced by the TSPS-specific training script before it is used by a
miner.

## Local contract smoke test

From the Ghost-Lattice code directory:

```powershell
python tsps_integration/contract_smoke.py
```

This checks window shape, finite predictions, ten-value sparse output,
100-value expansion, and deterministic prediction hashing. It is not a
substitute for the upstream TSPS benchmark.
