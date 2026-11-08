# Terminal operations

The release gate is non-destructive and wallet-safe:

```powershell
$env:PYTHONPATH='Edge-Of-Chaos\The-Ghost-Lattice\code'
python Edge-Of-Chaos\The-Ghost-Lattice\code\tsps_integration\run_release_checks.py `
  --root Edge-Of-Chaos\The-Ghost-Lattice `
  --artifact Edge-Of-Chaos\The-Ghost-Lattice\artifacts\tsps-official-5k `
  --score Edge-Of-Chaos\The-Ghost-Lattice\artifacts\tsps-official-5k\official_score_full.json `
  --output Edge-Of-Chaos\The-Ghost-Lattice\artifacts\tsps-official-5k\release_manifest.json
```

It verifies the artifact hashes and records that no wallet, registration, or
transfer was attempted. The live sequence is deliberately separate:

1. Install the pinned TSPS dependencies and Bittensor on a Linux host.
2. Run the adapter through TSPS testnet (`netuid 3`).
3. Observe validator responses and emissions before any mainnet action.
4. Only after explicit confirmation, register on mainnet (`netuid 8`).
5. Transfer only confirmed rewards; never place seed phrases in scripts or logs.

The current offline score is evidence for funding review, not a promise of
emissions. Live ranking depends on validator scoring and network state.
