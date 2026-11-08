# Dev-funding post — Ghost-Lattice TSPS

**Project:** Ghost-Lattice TSPS Miner  
**Builder:** Elbalor  
**Contact:** heylelyaka@gmail.com  
**Request:** small registration/deployment support for TSPS testnet netuid 3

I built a backpropagation-free Ghost-Lattice predictor for the pinned TSPS
BTC/USD 5-minute contract. The core uses the Rheon adaptive gate, Ising
reservoir surrogate, Volterra polynomial forge, and convex readout.

## Reproducible result

- Pinned TSPS revision: `5660ff464526daaf2fe2d63e6d0a5ad1416071f5`
- Official-history temporal holdout: 1,000 windows
- Mean weighted RMSE: `0.1081623850`
- Median weighted RMSE: `0.0932747084`
- Mean inference latency: `60 ms`
- p95 latency: `112 ms`
- Output contract: 100 future BTC/USD closes
- Hash/reveal protocol checks: passed

Artifact SHA-256 (matches `release_manifest.json`, byte-for-byte):

```text
gate.pt               16fbeba1ecc00bf50326b2b0c283ad1eea18f7f609680d2822d5acd45f74b89a
model.npz             cee6ced4166ed3c14ebe7a5701dcca36444e4a4847b3c0da3b626b2442a5e145
model.json            9114f001542ead66e6f16b61cf5759ebb58b4e785f78094e22c758e2c2c5f882
official_score_full    a663af2176caf2307dc24cda29b959d2c9d6bee048e48cf47fff6b13473d443d
```

Evidence and model artifacts:

`Edge-Of-Chaos/The-Ghost-Lattice/artifacts/tsps-official-5k/`

Public SS58 address: `REPLACE_WITH_PUBLIC_SS58_ADDRESS`

## What this is and is not

This is an **offline** score computed with the pinned TSPS
`Scoring.calculate_weighted_rmse` (`k=0.1`) over a 1,000-window temporal
holdout. It is not a claim about live validator rank, emissions, or
mainnet performance. I have not registered on any subnet yet, and no wallet
or on-chain action is part of this package.

I do not know how this model ranks against other miners, because I have no
access to their scores. I am reporting what I can measure, and nothing else.

## Reproduction

A validator can verify the score independently. Two things to know up front:

1. A hash proves file identity; it does not reproduce the number. Reproduction
   needs `gate.pt`, `model.npz`, the 5,000-window dataset, the pinned TSPS
   checkout, and a working Python environment.
2. Budget roughly ten to fifteen minutes for a clean run, not two. The
   commands are in `GRANT_BRIEF.md`.

The trained weights are my own model artifact. I will share them directly
with any validator who wants to run the evaluation, rather than publishing
them in a public repository.

## Request

I am requesting a small registration/deployment grant to run the miner on
testnet netuid 3, publish live validator results, and then decide whether
mainnet participation is justified.

No seed phrase, private key, or wallet password is included, and none will
be sent to anyone in this or any other channel.
