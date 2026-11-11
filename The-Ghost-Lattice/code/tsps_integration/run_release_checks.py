"""Run the non-destructive TSPS release gate from a terminal.

This never creates a wallet, registers a UID, or transfers funds. It validates
the artifact and writes a release manifest that can accompany a testnet
submission.
"""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--score', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, artifact = args.root.resolve(), args.artifact.resolve()
    required = [artifact/'gate.pt', artifact/'model.npz', artifact/'model.json', args.score.resolve()]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('Missing release files: ' + ', '.join(missing))
    score = json.loads(args.score.resolve().read_text(encoding='utf-8'))
    manifest = {
        'kind': 'ghost_lattice_tsps_release_gate',
        'python': platform.python_version(),
        'upstream_revision': json.loads((artifact/'model.json').read_text())['upstream_revision'],
        'artifact': str(artifact),
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in required},
        'score': score,
        'network_plan': {'testnet_netuid': 3, 'mainnet_netuid': 8},
        'safety': {'wallet_created': False, 'registration_attempted': False, 'transfer_attempted': False},
        'next_live_step': 'run the pinned TSPS miner on testnet netuid 3 after installing Bittensor and supplying a wallet explicitly',
    }
    out = args.output.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
