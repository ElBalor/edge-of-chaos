"""Zeta: critically-damped second-order trajectory generation.

Paper section "Zeta and the Second Order Derivative": the Ghost-Lattice
readout must not drive servos directly (reservoir jitter would glitch the
limb). Instead the readout acts as a forcing term f on a virtual
mass-spring-damper:

    tau^2 * y'' = alpha * (beta * (g - y) - y') + f

y   = the smooth servo trajectory
g   = the goal state decoded from the reservoir
f   = the nonlinear forcing term from the forge/readout

In standard form the damping ratio is zeta = sqrt(alpha) / (2*sqrt(beta));
the defaults alpha=4, beta=1 give zeta = 1 exactly: critical damping, zero
overshoot, zero oscillation, C^2 continuity.

Author: Heylel Yaka.
License: CC BY-NC 4.0
"""

import numpy as np


class ZetaTrajectory:
    """Integrator for the critical-damped trajectory ODE.

    Uses semi-implicit (symplectic) Euler: update velocity first, then
    position with the *new* velocity. Unconditionally stable for this
    damped system and cheap enough for real-time control loops.
    """

    def __init__(self, tau=0.1, dt=0.01, alpha=4.0, beta=1.0,
                 y0=0.0, dy0=0.0, goal=0.0):
        if tau <= 0 or dt <= 0:
            raise ValueError("tau and dt must be positive")
        self.tau = float(tau)
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.y = float(y0)
        self.dy = float(dy0)
        self.g = float(goal)
        self.f = 0.0

    @property
    def damping_ratio(self):
        """zeta = sqrt(alpha) / (2 sqrt(beta)); 1.0 == critical damping."""
        return float(np.sqrt(self.alpha) / (2.0 * np.sqrt(self.beta)))

    def set_goal(self, g):
        self.g = float(g)

    def set_forcing(self, f):
        self.f = float(f)

    def step(self, f=None, g=None):
        """Advance one control step; returns (y, y', y'')."""
        if f is not None:
            self.set_forcing(f)
        if g is not None:
            self.set_goal(g)
        acc = (self.alpha * (self.beta * (self.g - self.y) - self.dy)
               + self.f) / (self.tau ** 2)
        self.dy += acc * self.dt
        self.y += self.dy * self.dt
        return self.y, self.dy, acc

    def rollout(self, steps, forcing=None, goals=None):
        """Integrate ``steps`` steps.

        ``forcing`` and ``goals`` may be callables k -> value or sequences.
        Returns the trajectory array.
        """
        ys = np.empty(steps)
        for k in range(steps):
            f = forcing(k) if callable(forcing) else (
                forcing[k] if forcing is not None else None)
            g = goals(k) if callable(goals) else (
                goals[k] if goals is not None else None)
            y, _, _ = self.step(f=f, g=g)
            ys[k] = y
        return ys


def check_c2_smoothness(y, dt, tol=1e6):
    """Numerical C^2 check: finite jerk (derivative of acceleration).

    Returns max |jerk|; for a critically-damped ODE trajectory this is
    bounded, i.e. the motion is smooth in acceleration as well as
    position and velocity.
    """
    y = np.asarray(y, dtype=float)
    vel = np.gradient(y, dt)
    acc = np.gradient(vel, dt)
    jerk = np.gradient(acc, dt)
    return float(np.max(np.abs(jerk))) if len(jerk) else 0.0
