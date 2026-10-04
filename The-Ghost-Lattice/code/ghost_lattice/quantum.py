"""Exact state-vector simulation of the 29-qubit TFIM reservoir.

Simulates the REAL quantum dynamics of the Fully Connected Transverse-
Field Ising Hamiltonian

    H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i

on a state vector of 2^N complex amplitudes (N up to 29: 2^29 = 536,870,912
amplitudes, ~4.3 GB in complex64). This is the literal quantum reservoir
of the Ghost-Lattice paper -- not the tanh mean-field surrogate.

Trotterised evolution (2nd order, symmetric):

    U(dt) ~ e^{-i dt/2 sum h_i X_i} e^{-i dt sum J_ij Z_i Z_j}
            e^{-i dt/2 sum h_i X_i}

- The ZZ factor is DIAGONAL in the computational basis: its phase is
  exp(-i dt E(s)) with E(s) the classical Ising energy of basis state s.
  E is built once by an exact O(2^N) incremental doubling and cached;
  input encoding (W_in * x(t)) is folded into the same diagonal.
- Each X_i factor is an EXACT single-qubit rotation applied through a
  strided view of the state -- no approximation there. The only
  approximation is the O(dt^3) per-step Trotter error, which vanishes
  with dt and is measured by the validation harness.

Observables: the reservoir taps are the magnetisations <Z_i>(t_k) at V
evenly spaced times -- the direct quantum analogue of the surrogate's
spin values s_i(t_k).

Memory: peak usage is roughly 21 bytes per amplitude (complex64 state +
complex diagonal transient + real scratch). N = 29 wants ~11.3 GB of
free accelerator/RAM memory; the device is chosen automatically and
smaller machines should stay at N <= 26-28 or use the CPU for N <= 24.

Author: Heylel Yaka (Elbalor / The Digital Necromancer)
License: CC BY-NC 4.0
"""

import numpy as np
import torch

# Import core FIRST: it establishes the sklearn-before-torch import order
# that avoids the Windows/conda OpenMP clash on this machine class.
from .core import DEVICE, N_QUBITS  # noqa: F401

BYTES_PER_AMP = 21  # state + diagonal transient + scratch, conservatively


def _pick_device(n_qubits):
    """GPU if there is headroom for the state plus transients, else CPU."""
    need = BYTES_PER_AMP * (2 ** n_qubits)
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free > need * 1.2:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def ising_energies(J, device, dtype=torch.float32):
    """Diagonal ZZ energy E(s) for every basis state, in O(2^N).

    E(s) = sum_{j<k} J_jk z_j(s) z_k(s) with z_i(s) = 1 - 2*bit_i(s).

    Two nested doublings, both exact. Bit convention: qubit i is bit i
    counting from the RIGHT (little-endian; bit 0 is the fastest-varying
    axis), matching the strided views used by the evolution operators.

    - For each qubit k, its row potential over ALL 2^k prefixes of the
      bit string is built term by term:
          G^{(m+1)}(b, bit_m) = G^{(m)}(b) + J_{k,m} * z_m
      which interleaves as stack([G + c ; G - c], last axis).
    - Placing qubit k then interleaves the energy table with that fresh
      potential:
          E_{k+1}(b, bit_k) = E_k(b) + z_k * G^{(k)}(b)

    Every pair (j, k) contributes exactly once, when qubit k is placed.
    Total work O(2^N); runs once and is cached by TFIMReservoir.prepare.
    """
    N = J.shape[0]
    Jd = np.asarray(J, dtype=np.float32)   # float32: phases need ~1e-7
    E = torch.zeros(1, dtype=torch.float32, device=device)
    for k in range(N):
        # G_k = sum_{j<k} J[k, j] z_j over all 2^k prefix assignments
        # (k = 0 places the first qubit with an empty potential: no pairs)
        G = torch.zeros(1, dtype=torch.float32, device=device)
        for m in range(k):
            c = float(Jd[k, m])
            G = torch.stack([G + c, G - c], dim=-1).reshape(-1)
        # place qubit k: z_k = +1 (bit 0) / -1 (bit 1) branches,
        # interleaved so bit k becomes the fastest-varying axis
        E = torch.stack([E + G, E - G], dim=-1).reshape(-1)
    assert E.numel() == 2 ** N
    return E.to(dtype)


def field_projection(w, device, dtype=torch.float32):
    """Wz(s) = sum_i w_i z_i(s) for every basis state (input encoding).

    Little-endian bit convention, matching ising_energies.
    """
    N = len(w)
    out = torch.zeros(1, dtype=torch.float32, device=device)
    for i in range(N):
        out = torch.stack([out + float(w[i]), out - float(w[i])],
                          dim=-1).reshape(-1)
    assert out.numel() == 2 ** N
    return out.to(dtype)


class TFIMReservoir:
    """Exact Trotterised FC-TFIM quantum reservoir.

    Parameters
    ----------
    n_qubits : int
        Number of qubits N (up to 29; memory grows as 2^N).
    J : (N, N) array, optional
        Symmetric ZZ couplings. Default: scaled symmetric Gaussian.
    h : (N,) array, optional
        Transverse fields. Default: scaled Gaussian.
    W_in : (N,) array, optional
        Input encoding weights; input x(t) enters as a Z phase
        x(t) * Wz(s) on the diagonal.
    dt : float
        Trotter step.
    order : 1 or 2
        Trotter order (2 = symmetric, default).
    device : str, optional
        'cuda' or 'cpu'; auto-selected when None.
    seed : int
        Seed for the default random couplings.
    """

    def __init__(self, n_qubits=N_QUBITS, J=None, h=None, W_in=None,
                 dt=0.1, order=2, device=None, seed=42):
        self.N = n_qubits
        rng = np.random.RandomState(seed)
        if J is None:
            J = rng.randn(n_qubits, n_qubits) * 0.1
            J = (J + J.T) / 2
        if h is None:
            h = rng.randn(n_qubits) * 0.1
        if W_in is None:
            W_in = rng.randn(n_qubits) * 0.1
        self.J = np.asarray(J, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.W_in = np.asarray(W_in, dtype=np.float64).ravel()
        self.dt = float(dt)
        self.order = int(order)
        self.device = device or _pick_device(n_qubits)
        self._diag_ready = False

    # -- setup ------------------------------------------------------------

    def prepare(self):
        """Precompute cached diagonals (call once before evolving)."""
        self.E = ising_energies(self.J, self.device)
        self.Wz = field_projection(self.W_in, self.device)
        self.hx = self.h.astype(np.float64)
        self._diag_ready = True
        return self

    def memory_gb(self):
        return BYTES_PER_AMP * (2 ** self.N) / 1e9

    # -- evolution ---------------------------------------------------------

    def _apply_x_layer(self, psi, theta_scale):
        """Apply prod_i exp(-i theta_scale h_i X_i) exactly, in place.

        Qubit i is isolated through the view (2^i, 2, 2^(N-i-1)); the
        middle axis is the bit. exp(-i t X)|0> = c|0> - i s|1> and
        exp(-i t X)|1> = c|1> - i s|0>.
        """
        N = self.N
        for i in range(N):
            theta = theta_scale * self.hx[i]
            c, s = np.cos(theta), np.sin(theta)
            v = psi.view(2 ** i, 2, 2 ** (N - i - 1))
            a = v[:, 0, :]
            b = v[:, 1, :]
            na = a.mul(c).sub(b.mul(complex(0.0, s)))
            nb = b.mul(c).sub(a.mul(complex(0.0, s)))
            v[:, 0, :].copy_(na)
            v[:, 1, :].copy_(nb)

    def _apply_diag(self, psi, x_t):
        """Apply exp(-i [dt E + x_t Wz](s)) exactly, in place."""
        angle = self.dt * self.E + x_t * self.Wz      # real scratch
        phase = angle.to(torch.complex64)
        del angle
        phase.mul_(1j)
        phase.exp_()
        psi.mul_(phase)

    def evolve(self, input_stream, n_steps, read_taps=None):
        """Evolve under H + input phases; return magnetisation taps.

        input_stream : (n_steps,) real inputs; x(t) enters the diagonal.
        read_taps    : step indices at which to record <Z_i>; default is
                       V = 5 evenly spaced steps (TEMPORAL_V semantics).

        Returns (n_taps, N) array of <Z_i>(t_k) -- the quantum reservoir
        taps. Spatial multiplexing (R reservoirs) = multiple instances.
        """
        if not self._diag_ready:
            self.prepare()
        if read_taps is None:
            stride = max(1, n_steps // 5)
            read_taps = [max(0, n_steps - 1 - v * stride) for v in range(5)]
        read_taps = sorted(set(int(t) for t in read_taps))
        psi = torch.zeros(2 ** self.N, dtype=torch.complex64,
                          device=self.device)
        psi[0] = 1.0  # |00...0>
        mags = []
        for t in range(n_steps):
            x_t = float(input_stream[t]) if input_stream is not None else 0.0
            if self.order == 2:
                self._apply_x_layer(psi, 0.5 * self.dt)
                self._apply_diag(psi, x_t)
                self._apply_x_layer(psi, 0.5 * self.dt)
            else:
                self._apply_diag(psi, x_t)
                self._apply_x_layer(psi, 1.0 * self.dt)
            if t in read_taps:
                mags.append(self.magnetisations(psi))
        return np.array(mags, dtype=np.float32)

    # -- observables ---------------------------------------------------------

    def magnetisations(self, psi):
        """<Z_i> for every qubit, one pass over the state."""
        p2 = psi.abs().square_()                      # float32, in place temp
        N = self.N
        out = np.empty(N, dtype=np.float64)
        total = p2.sum(dtype=torch.float64).item()
        for i in range(N):
            v = p2.view(2 ** i, 2, 2 ** (N - i - 1))
            z0 = v[:, 0, :].sum(dtype=torch.float64).item()
            z1 = v[:, 1, :].sum(dtype=torch.float64).item()
            out[i] = (z0 - z1) / total
        return out


def surrogate_spin_trajectories(J, h, W_in, input_stream, n_steps):
    """Mean-field tanh surrogate spin trajectories, same protocol.

    Mirrors core.simulate_reservoir's recurrence s_i(t+1) = tanh(J s +
    h + W_in x(t)) and returns (n_steps, N) 'magnetisations' s_i(t) --
    the quantity the validation harness compares against the exact
    quantum <Z_i>(t).
    """
    N = J.shape[0]
    Jt = torch.tensor(J, dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)
    Wt = torch.tensor(np.asarray(W_in).reshape(-1)[:N], dtype=torch.float32)
    s = torch.zeros(N, dtype=torch.float32)
    traj = np.zeros((n_steps, N), dtype=np.float32)
    for t in range(n_steps):
        x_t = float(input_stream[t]) if input_stream is not None else 0.0
        s = torch.tanh(Jt @ s + ht + Wt * x_t)
        traj[t] = s.numpy()
    return traj
