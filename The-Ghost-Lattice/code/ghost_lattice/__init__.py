"""Ghost-Lattice QNN core.

A backprop-free quantum-reservoir computing architecture:

    LNN pre-gate -> 29-qubit Ising reservoir -> Volterra polynomial forge
    -> convex ridge readout (offline) / sliding-window ridge (online)

The gate (TauNet) and the reservoir couplings (J, h) are never trained by
gradients; they are evolved by the Chaos Algorithm and Spectral Genesis
respectively. Only the convex ridge readout is solved for.

Author: Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0
"""

from .core import (
    N_QUBITS,
    SPATIAL_R,
    TEMPORAL_V,
    POLY_DEGREE,
    TauNet,
    lnn_gate,
    init_reservoir,
    simulate_reservoir,
    polynomial_forge,
    per_reservoir_features,
    combined_features,
    EchoStateNetwork,
    chaos_algorithm,
    spectral_genesis,
    SlidingWindowRidge,
    GPUBlockDiagonalRLS,
    FullRLS,
)

__all__ = [
    "N_QUBITS",
    "SPATIAL_R",
    "TEMPORAL_V",
    "POLY_DEGREE",
    "TauNet",
    "lnn_gate",
    "init_reservoir",
    "simulate_reservoir",
    "polynomial_forge",
    "per_reservoir_features",
    "combined_features",
    "EchoStateNetwork",
    "chaos_algorithm",
    "spectral_genesis",
    "SlidingWindowRidge",
    "GPUBlockDiagonalRLS",
    "FullRLS",
]

__version__ = "1.0.0"
