#!/usr/bin/env bash
set -euo pipefail

# Run inside Ubuntu/WSL, not native PowerShell Python.
sudo apt-get update
sudo apt-get install -y git curl build-essential clang libssl-dev llvm libudev-dev pkg-config protobuf-compiler python3-venv python3-dev

python3 -m venv "$HOME/venvs/bittensor-tsps"
source "$HOME/venvs/bittensor-tsps/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install bittensor

python - <<'PY'
import bittensor as bt
print('bittensor import PASS')
print('version:', getattr(bt, '__version__', 'unknown'))
PY
btcli --version
