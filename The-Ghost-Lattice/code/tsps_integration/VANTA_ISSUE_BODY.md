## What this is

I built a BTC/USD predictor for the SN8 5-minute contract that uses no gradient
descent anywhere: an adaptive leaky-integrator gate, a 29-node Ising reservoir
with `rho(J) >= 1` enforced, a Volterra polynomial forge, and a convex ridge
readout solved in the dual so the ~33k-dimensional basis never enters the
linear solve. Fitting is closed-form plus genetic search.

Everything below is public and clickable, no request needed:

- Code and adapter: https://github.com/ElBalor/edge-of-chaos/tree/main/The-Ghost-Lattice/code/tsps_integration
- Evidence artifacts and hashes: https://github.com/ElBalor/edge-of-chaos/tree/main/The-Ghost-Lattice/artifacts/tsps-official-5k
- Full write-up: https://github.com/ElBalor/edge-of-chaos/blob/main/The-Ghost-Lattice/code/tsps_integration/DEV_FUNDING_POST.md

The artifacts directory opens with a README that reconciles every figure,
including the two numbers in there that look contradictory but are not. If a
number in this issue disagrees with the files, the files win and I want to
know about it.

## Measured, with the artifacts to check it

1,000-window temporal holdout, `Scoring.calculate_weighted_rmse`, `k=0.1`,
against the contract pinned at revision `5660ff464526daaf2fe2d63e6d0a5ad1416071f5`.

| Metric | Value |
|---|---|
| Mean weighted RMSE | 0.1081623850 |
| Median weighted RMSE | 0.0932747084 |
| Sparse RMSE (k=0.1) | 0.2441263172 |
| Latency mean / p95 / max | 60.2 ms / 112.2 ms / 355.6 ms |

Artifact hashes (also in `release_manifest.json`, byte-for-byte):

```text
gate.pt               16fbeba1ecc00bf50326b2b0c283ad1eea18f7f609680d2822d5acd45f74b89a
model.npz              cee6ced4166ed3c14ebe7a5701dcca36444e4a4847b3c0da3b626b2442a5e145
model.json             9114f001542ead66e6f16b61cf5759ebb58b4e785f78094e22c758e2c2c5f882
official_score_full    a663af2176caf2307dc24cda29b959d2c9d6bee048e48cf47fff6b13473d443d
```

Reproduction takes ~10–15 minutes on a laptop. Weights are not published in a
public repo — competitive subnet — but I will send them to any maintainer or
validator who wants to regenerate the score.

## The question that decides whether this is useful to you

I pinned `taoshidev/time-series-prediction-subnet` (last push 2024-03-22),
because that is the contract with the weighted-RMSE scoring I built against.
Reading `docs/miner.md` here, SN8 now scores miners on orders and position
leverage — per-pair caps, portfolio leverage tiers, drawdown and top-25
requirements — and there is no weighted-RMSE path in this repo.

So, concretely:

1. Does the prediction contract still feed SN8 scoring, or has it been fully
   replaced by the order/leverage model?
2. If it is replaced, is a predictor like this still useful as an
   order-emission layer on top of the current miner interface — i.e. does
   anyone want a signal source that is not itself a trading policy?
3. What is the current mainnet **netuid 8** registration burn, and is the
   300 Theta minimum collateral requirement unavoidable for a new miner?

I would rather ask than ship against a dead contract.

## The funding request

I am asking for **0.5 TAO** to cover registration, and I will confirm the live
burn before spending any of it:

| Item | Amount |
|---|---|
| Testnet netuid 3 registration burn | dynamic, checked live |
| Mainnet netuid 8 registration burn | dynamic, checked live |
| Buffer for burn fluctuation | remainder |

`docs/miner.md` states the mainnet registration fee is under 1 TAO, so 0.5 may
not cover both registrations. If it does not, I will post the confirmed numbers
and register on testnet only rather than over-ask. Testnet pays nothing; mainnet
is the only path to actually earning, which is why both are in the request.

I am not asking for dev time, hardware, or API costs — the model already runs
on my machine.

Coldkeypub (public, safe to post):

```text
5FYep5qG4oDaSJLCywgjmSfTcm8sSmLM48dadoWBXNeBGWru
```

A transfer would be a standard coldkey-to-coldkey transfer from the sender's own
wallet. I will report the exact amount spent and what is left. No mnemonic,
private key, or wallet password exists in any repo or channel, and I will not
send one to anyone.

## What I am not claiming

- `0.1082` is an **offline holdout** number. It is not live validator rank, not
  emissions, and not a SOTA claim. I have no visibility into other miners'
  scores and will not pretend to rank against them.
- The forge resolves **10 anchor values** interpolated to the required 100
  closes. Points between anchors are linear interpolants, not independent
  predictions. The architecture is strongest under sparse observation and that
  is the regime where I expect it to hold.
- Offline holdout windows come from recorded history. They do not capture live
  latency, missed submissions, or adversarial scoring.
- I have not registered on any subnet. No coldkey has touched a subnet and no
  transfer has been attempted; `release_manifest.json` records this under
  `safety`.
- A SHA-256 proves file identity, not the score. Reproduction needs the files
  and the commands above.

I am happy to open a PR against this repo for anything that belongs in-tree —
an order-emission adapter, docs, or tests — once you tell me which of the two
scoring paths is current.
