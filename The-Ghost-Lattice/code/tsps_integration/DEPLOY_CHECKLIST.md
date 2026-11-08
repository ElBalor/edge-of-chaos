# Elbalor TSPS deployment checklist

1. Create the coldkey and hotkey locally with `btcli`; keep seeds offline.
2. Record only the public SS58 address in `SUBMISSION_DRAFT.md`.
3. Run the pinned miner against TSPS testnet netuid 3.
4. Capture validator score, uptime, latency, and logs.
5. Submit the evidence and public address to the official funding channel.
6. Register mainnet netuid 8 only after testnet validation and explicit approval.
7. Transfer only confirmed rewards; never automate seed phrases or private keys.

This repository contains no wallet secret and performs no registration or fund
transfer automatically.
