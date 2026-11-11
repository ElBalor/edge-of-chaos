# Paste-ready for `#dev-funding`

Discord caps a message at 2,000 characters. The block below is ~1,850, so it
posts in one message. Copy everything between the rules.

---

**Ghost-Lattice TSPS Miner — testnet + mainnet registration grant request**

Builder: Elbalor · heylelyaka@gmail.com
Full post: github.com/ElBalor/edge-of-chaos/blob/main/The-Ghost-Lattice/code/tsps_integration/DEV_FUNDING_POST.md

Backpropagation-free BTC/USD predictor for the pinned TSPS 5-min contract:
Rheon gate → Ising reservoir → Volterra forge → convex readout. No gradients
anywhere in the stack.

**Measured offline** (1,000-window holdout, pinned TSPS `5660ff4`, k=0.1):
mean weighted RMSE 0.1082 · median 0.0933 · sparse RMSE 0.2441
latency 60 ms mean / 112 ms p95 · 100 closes · hash/reveal checks passed

**Verify yourself, no request needed:**
artifacts + hashes: github.com/ElBalor/edge-of-chaos/tree/main/The-Ghost-Lattice/artifacts/tsps-official-5k
code + adapter: github.com/ElBalor/edge-of-chaos/tree/main/The-Ghost-Lattice/code/tsps_integration
SHA-256 for every artifact is in `release_manifest.json`, and the artifacts
README reconciles the figures that look contradictory but are not.

**Not claiming:** that is an offline holdout score — not live rank, not
emissions, not SOTA. I cannot see other miners' scores and won't pretend to
rank against them. Not registered on any subnet yet.

**Ask: 0.5 TAO** — both registration burns: testnet netuid 3 first for a live
run, then mainnet netuid 8 so the miner can actually compete for emissions.
Both burns checked live before registering. No dev time, hardware, or API costs.

SS58 (coldkeypub, public): `5FYep5qG4oDaSJLCywgjmSfTcm8sSmLM48dadoWBXNeBGWru`

Weights go directly to any validator who wants to reproduce the score.
No seed phrase, private key, or wallet password will be posted here.

Full post answers reproduction steps, weights access, current registration
status, and what happens if the live score comes back worse than the holdout —
so you shouldn't need to ask me anything to start.

---

## Notes before you post

- The SS58 in the block above is the real, already-created coldkeypub — no
  placeholder to fill in. It is public by design; never post anything but this.
- Post once. Reposting the same request daily is the fastest way to be muted
  in a grant channel — reviewers see it and it costs you credibility.
- Expect days, not minutes. A grant is a person reading this and approving a
  payment; there is no endpoint, command, or bot that issues one instantly.
