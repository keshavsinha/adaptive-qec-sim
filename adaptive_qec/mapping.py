"""Mapping validation (Figure 5): exact amplitude damping vs. its Pauli twirl.

The decoder only ever sees the twirled Pauli model f_twirl. Here we check how
well that model reproduces the stabilizer detection probability that the exact
(non-Pauli) amplitude-damping channel produces, for weight-4 surface-code
checks. Calculations use single-qubit density matrices and are exact (no
sampling): the data qubits of a check are in product states or classical
mixtures of product states, so <S> factorises per term.
"""
from itertools import product

import numpy as np
import pandas as pd

from .core import Params, big_gamma, twirl

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])


def ad_exact(rho, G):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - G)]]); K1 = np.array([[0, np.sqrt(G)], [0, 0]])
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def ad_twirl(rho, G):
    px, py, pz = twirl(G)
    return (1 - px - py - pz) * rho + px * X @ rho @ X + py * Y @ rho @ Y + pz * Z @ rho @ Z


def _detect(states, weights, pauli, channel, G):
    """Detection probability (1 - <S>)/2 for S = pauli^{x4}, averaged over a
    classical mixture of 4-qubit product states (each a list of 4 kets)."""
    exp = 0.0
    for w, kets in zip(weights, states):
        val = 1.0
        for k in kets:
            rho = channel(np.outer(k, k.conj()), G)
            val *= np.real(np.trace(pauli @ rho))
        exp += w * val
    return (1 - exp) / 2


def mapping_validation(P: Params, n_points: int = 30):
    plus = np.array([1, 1]) / np.sqrt(2); zero = np.array([1, 0]); one = np.array([0, 1])
    even = [b for b in product((0, 1), repeat=4) if sum(b) % 2 == 0]
    z_states = [[one if bit else zero for bit in b] for b in even]
    z_w = [1 / len(even)] * len(even)
    rows = []
    for g in np.linspace(0.02, 0.18, n_points):
        G = float(big_gamma(g, P))
        r = {"gamma": g, "Gamma": G}
        for name, st, w, S in [("X_check", [[plus] * 4], [1.0], X),
                               ("Z_check_code", z_states, z_w, Z),
                               ("Z_check_0000", [[zero] * 4], [1.0], Z),
                               ("Z_check_1111", [[one] * 4], [1.0], Z)]:
            e = _detect(st, w, S, ad_exact, G); t = _detect(st, w, S, ad_twirl, G)
            r[f"{name}_exact"], r[f"{name}_twirl"] = e, t
            r[f"{name}_relerr"] = abs(t - e) / e if e > 0 else np.inf
        rows.append(r)
    return pd.DataFrame(rows)
