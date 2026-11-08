# Colab/Linux execution sequence

Use a Python 3.10/3.11 Linux runtime for the pinned TSPS repository. The
upstream dependency set is old; do not install it into the Ghost-Lattice
desktop environment.

```bash
git clone --depth 1 https://github.com/taoshidev/time-series-prediction-subnet.git
cd time-series-prediction-subnet
git rev-parse HEAD
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pinned repository imports an old Coin Metrics client even though its
BTC/USD configuration disables that source. Prepend the packaged compatibility
namespace so the disabled import does not block the exchange-only path:

```bash
export PYTHONPATH=/path/to/The-Ghost-Lattice/code/tsps_integration/tsps_compat:$PYTHONPATH
```

Generate the official historical file with the upstream tool:

```bash
python runnable/generate_historical_data.py
```

The Ghost-Lattice trainer consumes an exported NPZ with `windows` shaped
`(N, 500, 1)` and `targets` shaped `(N, 10)`. The conversion must use the
same TSPS feature source, scaler, interval, and temporal split as the pinned
repository; do not substitute exchange data from another schema.

After producing the NPZ:

```bash
PYTHONPATH=/path/to/The-Ghost-Lattice/code \
python /path/to/The-Ghost-Lattice/code/tsps_integration/train_tsps_readout.py \
  --data tsps_windows.npz \
  --output artifacts/ghost-lattice-tsps
```

Then run the upstream `runnable/miner_testing.py` with the adapter model and
retain its raw output, commit hash, artifact metadata, and latency report.

## Single Colab cell (official-data path)

This is the shortest reproducible path once the project folder is uploaded to
Drive. It intentionally fails closed if the official `.taosfs` file was not
created; it never silently falls back to an unrelated Binance-only dataset.

```python
from google.colab import drive
drive.mount("/content/drive")

import os, pathlib, subprocess, sys

ROOT = pathlib.Path("/content/drive/MyDrive/Edge-Of-Chaos/The-Ghost-Lattice")
TSPS = pathlib.Path("/content/tsps")
ART = ROOT / "code" / "tsps_integration" / "artifacts" / "tsps-official"
NPZ = ROOT / "code" / "tsps_integration" / "data" / "tsps-official.npz"
TSPS.mkdir(parents=True, exist_ok=True)

if not (TSPS / ".git").exists():
    subprocess.run(["git", "clone", "https://github.com/taoshidev/time-series-prediction-subnet.git", str(TSPS)], check=True)
subprocess.run(["git", "fetch", "--depth", "1", "origin", "5660ff464526daaf2fe2d63e6d0a5ad1416071f5"], cwd=TSPS, check=True)
subprocess.run(["git", "checkout", "--detach", "5660ff464526daaf2fe2d63e6d0a5ad1416071f5"], cwd=TSPS, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=TSPS, check=True)

os.environ["PYTHONPATH"] = ":".join([
    str(ROOT / "code" / "tsps_integration" / "tsps_compat"),
    str(TSPS), str(ROOT / "code"), os.environ.get("PYTHONPATH", "")
])

hist = TSPS / "runnable" / "historical_financial_data" / "data_training.taosfs"
if not hist.exists():
    subprocess.run([sys.executable, "runnable/generate_historical_data.py"], cwd=TSPS, check=True)
if not hist.exists():
    raise RuntimeError(f"Official TSPS historical file was not created: {hist}")

NPZ.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(ROOT / "code" / "tsps_integration" / "export_tsps_windows.py"),
                "--historical", str(hist), "--output", str(NPZ), "--max-samples", "50000"], check=True)
subprocess.run([sys.executable, str(ROOT / "code" / "tsps_integration" / "train_tsps_readout.py"),
                "--data", str(NPZ), "--output", str(ART),
                "--source-metadata", str(NPZ.with_suffix(".json"))], check=True)
print("Official TSPS artifact ready:", ART)
```
