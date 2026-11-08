# Paste-ready for `#dev-funding`

Discord caps a message at 2,000 characters. The block below is ~1,650, so it
posts in one message with room to spare. Copy everything between the rules.

---

**Ghost-Lattice TSPS Miner — testnet deployment grant request**

Builder: Elbalor · heylelyaka@gmail.com
Full post: https://github.com/ElBalor/edge-of-chaos/blob/main/The-Ghost-Lattice/code/tsps_integration/DEV_FUNDING_POST.md

A backpropagation-free BTC/USD predictor for the pinned TSPS 5-min contract:
Rheon adaptive gate → Ising reservoir → Volterra forge → convex readout.
There are no gradients anywhere in the stack.

**Measured offline** (1,000-window temporal holdout, pinned TSPS `5660ff4`, k=0.1):
• mean weighted RMSE 0.1082 · median 0.0933
• sparse RMSE 0.2441
• latency 60 ms mean / 112 ms p95
• contract: 100 closes · hash/reveal checks passed

**Artifact SHA-256** (matches `release_manifest.json` byte-for-byte):
`gate.pt` 16fbeba1…f74b89a
`model.npz` cee6ced4…2a5e145
`model.json` 9114f001…2c5f882
`official_score_full` a663af21…d443d

**What I'm NOT claiming:** that number is an offline holdout score. It is not
live validator rank, not emissions, and not a SOTA claim — I have no access to
other miners' scores, so I won't pretend to rank against them. I have not
registered on any subnet yet.

**Ask: 0.1 TAO** — testnet netuid 3 registration burn, confirmed live before
registering. Mainnet: 0.00 requested. No dev time, hardware, or API costs.

Weights go directly to any validator who wants to reproduce the score.
No seed phrase, private key, or wallet password will be posted here.

Full post also answers reproduction steps, weights access, current
registration status, and what happens if the live score comes back worse
than the holdout — so you shouldn't need to ask me anything to start.

---

## Notes before you post

- Fill in nothing. The public post link already works; verify it opens first.
- Do not add a wallet address until the wallet exists. A placeholder in a
  funding channel reads as unfinished and gets ignored.
- Post once. Reposting the same request daily is the fastest way to be muted
  in a grant channel — reviewers see it and it costs you credibility.
- Expect days, not minutes. A grant is a person reading this and approving a
  payment; there is no endpoint, command, or bot that issues one instantly.
