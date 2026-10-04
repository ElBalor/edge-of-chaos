# Ghost-Lattice QNN — Code

Backprop-free quantum-reservoir computing: an evolved Liquid Time-Constant
pre-gate, a 29-qubit Transverse-Field Ising reservoir (tanh-oscillator
mean-field surrogate), a Volterra polynomial forge (~96k features), and a
convex readout (ridge offline / anchored sliding-window ridge online;
RLS is kept only as a comparison baseline).

The gate and the reservoir couplings are **never trained by gradients** —
they are evolved by two genetic algorithms:

- **Chaos Algorithm** — evolves the τ-net (tournament elites, uniform
  crossover, Gaussian mutation).
- **Spectral Genesis** — evolves the couplings `J`, `h` (SVD singular-value
  mutation, symmetry restoration, spectral-radius filter rejecting ρ(J) ≥ 1).

## Layout

```
ghost_lattice/
    __init__.py        public API
    core.py            pipeline + GAs + readouts (Ridge, sliding-window,
                       GPU block-diagonal RLS, full RLS with warm-start)
    symbolic.py        Lasso pruning -> symbolic fit -> Lean 4 theorem
                       emission (Section 4 of the paper)
    motion.py          Zeta critically-damped second-order trajectory ODE
    safety.py          EMG distress-stop detector, hardware relay model,
                       bench self-test
run_mackey_glass.py    Mackey-Glass τ=17 benchmark (ESN / GRU / linear
                       baselines, full-ridge + block-diagonal RLS readouts)
run_lorenz.py          Lorenz attractor benchmark (full RLS warm-started
                       from the closed-form ridge solution)
run_bci.py             BCI motor-imagery classification (bandpower
                       features, per-reservoir selection, GA evolution;
                       --demo for synthetic selftest without the dataset)
```

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+, PyTorch (CUDA optional — falls back to CPU), scikit-learn,
numpy, tqdm; `ghost_lattice.safety` additionally uses scipy (bundled with
conda).

## Running

```bash
# Mackey-Glass, 6-step horizon (default)
python run_mackey_glass.py

# harder 12-step horizon
python run_mackey_glass.py --horizon 12

# Lorenz x-variable forecast
python run_lorenz.py --horizon 6

# BCI selftest (synthetic 4-class motor imagery)
python run_bci.py --demo

# smoke-test any runner in ~2-4 minutes
python run_mackey_glass.py --quick
```

Splits: 2000 train / 500 validation / 500 test windows from a 5000-step
series; GA fitness uses the first 500 training samples and 200 validation
samples.

## Using the library

```python
from ghost_lattice import (
    TauNet, chaos_algorithm, init_reservoir, spectral_genesis,
    combined_features, SlidingWindowRidge,
)

# evolve the gate
tau_net = TauNet(input_dim=4).to("cuda")
tau_net = chaos_algorithm(X_tr, y_tr, X_val_feats, y_val,
                          tau_net, input_dim=1, pop_size=8, generations=5)

# evolve the reservoir
J, h, W_in = init_reservoir(42)
J, h = spectral_genesis(X_tr, y_tr, X_val_feats, y_val, tau_net,
                        J, h, W_in, input_dim=1, pop_size=8, generations=5)

# offline ridge readout, then online adaptation
# feats = combined_features(window, tau_net, J, h, W_in)  -> Ridge / SWR
```

## Windows / conda note

torch and scikit-learn each load an OpenMP runtime; the bundled
`KMP_DUPLICATE_LIB_OK=TRUE` handle plus the hard `os._exit(0)` at the end
of each runner keep imports stable and exits clean on Windows/conda
setups where the two runtimes clash.

## Notes

- `K_PER_BLOCK` selects the top f-regression features per reservoir block
  before RLS (15 000 total by default); the full forge is used for ridge.
- RLS variants carry denominator/gain clamps for numerical stability; the
  full RLS can be warm-started from the closed-form ridge solution
  (`ridge_warm_start`) so online updates begin at the batch optimum.
- The transverse-field Ising Hamiltonian is simulated through its
  mean-field tanh-oscillator surrogate; the spectral properties used by
  the reservoir are preserved while staying GPU-friendly.

## Author

Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0

