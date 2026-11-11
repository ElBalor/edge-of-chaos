# Dev-funding request: The Ghost-Lattice TSPS Miner (Quantum Neural Network AI)

**Project:** Ghost-Lattice TSPS miner — testnet netuid 3, then mainnet netuid 8
**Builder:** Elbalor — `heylelyaka@gmail.com`
**Ask:** **0.5 TAO** to register on testnet netuid 3 **and** mainnet netuid 8, so the miner can actually compete for emissions instead of sitting as an offline benchmark. Line-item breakdown below.
**Status:** artifacts built and hash-verified, offline score measured and reproducible.
**Not registered on any subnet yet.** No wallet funds have been received or sent.

## 1. What it is

A backpropagation-free BTC/USD predictor for the pinned TSPS 5-minute
contract. Four stages, no gradients anywhere:

1. **Rheon gate** — adaptive leaky integrator, `dx/dt = (I - x)/τ` solved by
   backward Euler, so it is unconditionally stable for `τ > 0`.
2. **Ising reservoir** — 29-node nonlinear reservoir, `ρ(J) ≥ 1` enforced.
3. **Volterra polynomial forge** — degree-2 plus cubes plus 4th powers,
   ~33,060 features across three reservoirs.
4. **Convex readout** — ridge regression solved in the dual, so the
   33,060-dimensional basis never enters the linear solve.

All fitting is closed-form or genetic search (Chaos Algorithm on the gate,
Spectral Genesis on `J` and `h`). Nothing is trained by gradient descent.

## 2. Measured result

Pinned TSPS revision `5660ff464526daaf2fe2d63e6d0a5ad1416071f5`.
1,000-window temporal holdout, `Scoring.calculate_weighted_rmse`, `k=0.1`.

| Metric | Value |
|---|---|
| Mean weighted RMSE | `0.1081623850` |
| Median weighted RMSE | `0.0932747084` |
| Sparse RMSE (k=0.1) | `0.2441263172` |
| Latency, mean | `60.2 ms` |
| Latency, p95 | `112.2 ms` |
| Latency, max | `355.6 ms` |
| Contract output | 100 future closes |
| Hash/reveal protocol checks | passed |


Verify directly, no request needed:

- Evidence artifacts and hashes: https://github.com/ElBalor/edge-of-chaos/tree/main/The-Ghost-Lattice/artifacts/tsps-official-5k
- Code and adapter: https://github.com/ElBalor/edge-of-chaos/tree/main/The-Ghost-Lattice/code/tsps_integration

The artifacts directory opens with a README that reconciles the two figures
in it that look contradictory but are not: the `0.2793` in `model.json` is
the training-split validation metric, while the `0.2441` sparse RMSE is the
official evaluator holdout. Same for the two latency numbers — one is the
scoring loop, the other includes adapter serialization.

Artifact SHA-256, matching `release_manifest.json` byte-for-byte:

```text
gate.pt               16fbeba1ecc00bf50326b2b0c283ad1eea18f7f609680d2822d5acd45f74b89a
model.npz             cee6ced4166ed3c14ebe7a5701dcca36444e4a4847b3c0da3b626b2442a5e145
model.json            9114f001542ead66e6f16b61cf5759ebb58b4e785f78094e22c758e2c2c5f882
official_score_full    a663af2176caf2307dc24cda29b959d2c9d6bee048e48cf47fff6b13473d443d
```

## 3. What this is and is not

This is an **offline** holdout score. It is not a claim about live validator
rank, emissions, or mainnet performance.

I have no access to other miners' scores, so I make no claim about live rank yet, and I will not pretend to rank against competitors before a real run. Target is top miner; the post is a credible ask, not a certificate. I am reporting what I can
measure independently and nothing else.

**Limitations I am disclosing up front, because you would find them anyway:**

- The forge resolves **10 anchor values**, which are interpolated to the
  required 100-close output. Points between anchors are linear interpolants,
  not independent predictions.This architecture is strongest under sparse
observation (`k=0.1`) and that is where I would expect it to hold up.
- Offline holdout windows are drawn from recorded history. They do not
  capture live network latency, missed submissions, or adversarial scoring.
- The `0.1082` figure is the mean; the `0.0933` I quote elsewhere is the
  median. Distribution is right-skewed. Both numbers are above.

---

## 4. Questions you would otherwise have to ask me

**"Can I reproduce this?"** Yes, and it takes about 10–15 minutes on a
normal laptop, not two. You need `gate.pt`, `model.npz`, the 5,000-window
dataset, and a checkout of TSPS at the pinned revision. Exact commands are
in `GRANT_BRIEF.md` in this directory.

**"Does the hash prove the score?"** No. A SHA-256 proves file identity —
that the artifact you have is the artifact I measured. It does not
regenerate the number. Reproduction needs the files above.

**"Do you have the trained weights?"** Yes. I am not publishing them in a
public repo — this is a competitive subnet. Ask and I will send them
directly, along with the exact command that produced `official_score_full.json`.

**"Have you registered anything?"** No. No coldkey has touched a subnet, no
registration has been attempted, and no transfer has been made. The release
manifest records this explicitly under `safety`.

**"Testnet or mainnet?"** Both, in that order. Testnet netuid 3 first — it is
cheap and it produces a live number that proves the whole pipeline works end
to end. Mainnet netuid 8 immediately after, because testnet pays nothing and
mainnet registration is the only path to actually earning TAO. That is the
point of building the miner. The 0.5 TAO covers both burns.

**"How much and what exactly does it pay for?"** 0.5 TAO, split across two
registrations:

| Item | Amount |
|---|---|
| Testnet netuid 3 registration burn | dynamic, confirm at time of registration |
| Mainnet netuid 8 registration burn | dynamic, confirm at time of registration |
| Buffer for burn fluctuation across both | remainder |

Registration burn is dynamic on both networks, so I will check the live value
before each registration rather than hardcoding a guess. If both burns together
come in under budget I will report the exact amount spent and what is left. If
the mainnet burn alone would exhaust the grant, I will post that number and stop
before the second registration rather than ask for more. I am not asking for
development time, hardware, or API costs — the model already runs on my
machine.

**"Is this a fork or someone else's code?"** No. The miner is original work.
TSPS is used only as the pinned upstream contract, which is the intended
pattern for mining this subnet.

**"What happens if the live score is bad?"** I post it. A worse number than
the holdout is the single most likely outcome, because live conditions are
harsher than recorded history — and the follow-up report will say so in the
same words I used here. I would rather hand you a mediocre real number than
another optimistic offline one.

**"What do you actually want from this channel?"** One thing: a small
registration grant covering both networks, so I can produce a live testnet
number and then get the miner onto mainnet where it can earn. If the live
number is weak, the grant did its job — it answered a question. It just
answered it negatively.

**"Where is your SS58?"** Public coldkeypub:

```
5FYep5qG4oDaSJLCywgjmSfTcm8sSmLM48dadoWBXNeBGWru
```

If a reviewer approves funding, the transfer is a standard Bittensor coldkey-to-coldkey transfer from the sender's own wallet. Example command a TAO holder would run from their own machine:

```bash
btcli wallet transfer --dest 5FYep5qG4oDaSJLCywgjmSfTcm8sSmLM48dadoWBXNeBGWru --amount 0.5 --wallet.name <sender_wallet> --wallet.hotkey <sender_hotkey>
```

This is the public coldkeypub only. The wallet's 24-word mnemonics are stored offline by the wallet owner and are never shared. The wallet was created locally and the release manifest records that under `safety`.

**"What is your timeline?"** Post today. Wallet local and public address now in this thread. Testnet registration as soon as the burn is funded, live score captured over a real run and reported here with the raw logs attached, then mainnet netuid 8 registration on the same grant.

---

## 5. Contact

`heylelyaka@gmail.com`: or reply in this thread.

No seed phrase, private key, or wallet password is included in this package,
and none will be sent to anyone in any channel, ever. Any message asking me
for them including one that appears to come from a validator or admin — is
a scam and I will not comply with it.
