"""Core simulation: OU drift, circuit construction, Kalman estimator, decoding.

Every number reported by the experiment scripts is produced by the functions in
this file. Nothing is hard-coded or fitted to a target value.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import pymatching
import stim

# Marker value used to locate the start of every syndrome round in the
# Stim-generated circuit. It is replaced before any simulation is run.
_ROUND_MARKER = 0.01234567


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    d: int = 5                 # code distance
    rounds: int | None = None  # syndrome rounds per memory experiment (default d)
    p_phys: float = 0.009      # baseline Pauli fault rate (see README, noise model)
    gamma_bar: float = 0.08    # OU mean relaxation rate, per cycle (Table 2)
    kappa: float = 0.05        # OU mean-reversion rate, per cycle (Table 2)
    sigma: float = 0.025       # OU diffusion strength, per cycle (Table 2)
    dt: float = 1.0            # normalised cycle time (Table 2)
    lam: float = 0.25          # fraction of a cycle the data qubits are exposed to T1
    warmup: int = 60           # cycles the estimator runs before each memory block
    t_cal: int = 15            # recalibration period for the periodic decoder
    noise: str = "phenomenological"  # or "circuit"

    @property
    def R(self) -> int:
        return self.rounds if self.rounds is not None else self.d


# --------------------------------------------------------------------------
# 1. Ornstein-Uhlenbeck relaxation drift (Eq. 1)
# --------------------------------------------------------------------------
def ou_trajectory(n: int, P: Params, rng: np.random.Generator) -> np.ndarray:
    """Euler-Maruyama OU path of length n, started from the stationary law.

    gamma_{t+1} = gamma_t - kappa (gamma_t - gamma_bar) dt + sigma sqrt(dt) w_t
    A relaxation rate cannot be negative, so the path is clipped at 0.
    """
    a = 1.0 - P.kappa * P.dt
    stat_sd = P.sigma * np.sqrt(P.dt / (1.0 - a * a))
    g = np.empty(n)
    g[0] = max(0.0, P.gamma_bar + stat_sd * rng.standard_normal())
    w = rng.standard_normal(n)
    for t in range(n - 1):
        g[t + 1] = g[t] - P.kappa * (g[t] - P.gamma_bar) * P.dt + P.sigma * np.sqrt(P.dt) * w[t]
        g[t + 1] = max(0.0, g[t + 1])
    return g


# --------------------------------------------------------------------------
# 2. Relaxation -> effective Pauli mapping (Eqs. 9-10, f_twirl)
# --------------------------------------------------------------------------
def big_gamma(gamma: np.ndarray | float, P: Params):
    """Per-cycle amplitude-damping parameter Gamma = 1 - exp(-lam * gamma * dt)."""
    return 1.0 - np.exp(-P.lam * np.asarray(gamma) * P.dt)


def twirl(G: float) -> tuple[float, float, float]:
    """Pauli twirl of amplitude damping with parameter G: (pX, pY, pZ)."""
    G = float(min(max(G, 0.0), 1.0))
    px = py = G / 4.0
    pz = (2.0 - G - 2.0 * np.sqrt(1.0 - G)) / 4.0
    return px, py, max(pz, 0.0)


# --------------------------------------------------------------------------
# 3. Circuit with a different relaxation level in every round
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _template(d: int, rounds: int, p: float, noise: str) -> stim.Circuit:
    kw = dict(distance=d, rounds=rounds, before_round_data_depolarization=_ROUND_MARKER,
              before_measure_flip_probability=p)
    if noise == "circuit":
        kw.update(after_clifford_depolarization=p, after_reset_flip_probability=p)
    elif noise != "phenomenological":
        raise ValueError(noise)
    return stim.Circuit.generated("surface_code:rotated_memory_z", **kw).flattened()


def build_circuit(P: Params, gammas: np.ndarray) -> stim.Circuit:
    """Rotated surface-code Z-memory with len(gammas) rounds.

    At the start of round r every data qubit receives
        DEPOLARIZE1(p_phys)  followed by  PAULI_CHANNEL_1(f_twirl(Gamma_r)).
    Measurement flips (and, for noise='circuit', gate/reset faults) occur at p_phys.
    """
    rounds = len(gammas)
    tmpl = _template(P.d, rounds, P.p_phys, P.noise)
    out = stim.Circuit()
    r = 0
    for inst in tmpl:
        if inst.name == "DEPOLARIZE1" and abs(inst.gate_args_copy()[0] - _ROUND_MARKER) < 1e-12:
            tg = inst.targets_copy()
            if P.p_phys > 0:
                out.append("DEPOLARIZE1", tg, P.p_phys)
            out.append("PAULI_CHANNEL_1", tg, twirl(big_gamma(gammas[r], P)))
            r += 1
        else:
            out.append(inst)
    assert r == rounds, (r, rounds)
    return out


def matcher(P: Params, gammas: np.ndarray) -> pymatching.Matching:
    """MWPM decoder whose edge weights are log((1-p)/p) of the DEM built from gammas."""
    dem = build_circuit(P, gammas).detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


# --------------------------------------------------------------------------
# 4. Syndrome-derived observable y_t and its calibration h(gamma) (Eq. 2)
# --------------------------------------------------------------------------
def _round_index(circ: stim.Circuit) -> np.ndarray:
    co = circ.get_detector_coordinates()
    return np.array([int(round(co[i][2])) for i in range(circ.num_detectors)])


def click_fraction(circ: stim.Circuit, det: np.ndarray) -> np.ndarray:
    """Fraction of detectors that fired in each round, per shot: shape (shots, rounds+1)."""
    t = _round_index(circ)
    n_t = t.max() + 1
    out = np.zeros((det.shape[0], n_t))
    for k in range(n_t):
        m = t == k
        out[:, k] = det[:, m].mean(axis=1)
    return out


@lru_cache(maxsize=None)
def calibrate_h(P: Params, n_grid: int = 25, shots: int = 4000, seed: int = 7):
    """Lookup-table calibration (Section 5.2, route i): bulk detector-click rate
    versus a constant relaxation rate gamma. Returns (gamma_grid, h_grid, n_det)."""
    grid = np.linspace(0.0, 0.40, n_grid)
    rounds = 12
    h = np.empty(n_grid)
    n_det = None
    for i, g in enumerate(grid):
        c = build_circuit(P, np.full(rounds, g))
        s = c.compile_detector_sampler(seed=seed + i)
        det = s.sample(shots)
        cf = click_fraction(c, det)
        h[i] = cf[:, 1:rounds].mean()          # bulk rounds only
        if n_det is None:
            t = _round_index(c)
            n_det = int((t == 1).sum())
    return grid, h, n_det


# --------------------------------------------------------------------------
# 5. Scalar extended Kalman filter on gamma (Eqs. 5-8)
# --------------------------------------------------------------------------
@dataclass
class KalmanResult:
    gamma_hat: np.ndarray
    P: np.ndarray


def kalman(y: np.ndarray, P: Params, calib, observed: np.ndarray) -> KalmanResult:
    """State model (OU):   gamma_{t+1} = a gamma_t + (1-a) gamma_bar + w_t,  Q = sigma^2 dt
    Observation:          y_t = h(gamma_t) + v_t,  R_t = h(1-h)/n_det (binomial)
    h is the calibrated lookup table, linearised around the prediction (EKF)."""
    grid, hg, n_det = calib
    a = 1.0 - P.kappa * P.dt
    Q = P.sigma ** 2 * P.dt
    hf = lambda g: np.interp(g, grid, hg)
    dh = lambda g: (hf(g + 1e-3) - hf(max(g - 1e-3, 0.0))) / (g + 1e-3 - max(g - 1e-3, 0.0))

    n = len(y)
    gh = np.empty(n)
    Pv = np.empty(n)
    x, p = P.gamma_bar, P.sigma ** 2 / (1 - a * a)    # stationary prior
    for t in range(n):
        if t > 0:
            x = a * x + (1 - a) * P.gamma_bar
            p = a * a * p + Q
        if observed[t]:
            hx = hf(max(x, 0.0))
            H = dh(max(x, 0.0))
            Rt = max(hx * (1 - hx), 1e-6) / n_det
            K = p * H / (H * H * p + Rt)
            x = x + K * (y[t] - hx)
            p = (1 - K * H) * p
        gh[t], Pv[t] = x, p
    return KalmanResult(gh, Pv)


def monitor_and_estimate(P: Params, gam: np.ndarray, rng: np.random.Generator, calib):
    """Run the syndrome-monitoring stream over the whole trajectory with Stim
    (one shot, true noise), form y_t from its detection events, and filter it."""
    L = len(gam)
    circ = build_circuit(P, np.append(gam, gam[-1]))   # L+1 rounds; round t holds cycle t
    det = circ.compile_detector_sampler(seed=int(rng.integers(2**63))).sample(1)
    y = click_fraction(circ, det)[0, :L]
    observed = np.zeros(L, bool)
    observed[1:] = True                                  # round 0 has no previous syndrome
    return kalman(y, P, calib, observed), y


# --------------------------------------------------------------------------
# 6. Statistics
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return ph, max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------
# 7. One operating point: all decoders on the same syndrome samples
# --------------------------------------------------------------------------
def run_point(P: Params, n_traj: int, shots: int, seed: int, delays=(0,)):
    """For each of n_traj independent OU trajectories:
      * cycles [0, warmup) are estimator warm-up, the memory block is
        cycles [warmup, warmup + R);
      * `shots` memory experiments are sampled from the TRUE-noise circuit;
      * the same detection events are decoded by every decoder.
    Logical failure = MWPM's predicted observable flip != the actual flip.
    Returns a dict of failure counts and the number of trials."""
    rng = np.random.default_rng(seed)
    calib = calibrate_h(P)
    R, W = P.R, P.warmup
    L = W + R
    m_static = matcher(P, np.full(R, P.gamma_bar))
    fails = {"static": 0, "estimator_only": 0, "periodic": 0, "oracle": 0}
    for k in delays:
        fails[f"adaptive_d{k}"] = 0

    for _ in range(n_traj):
        gam = ou_trajectory(L, P, rng)
        est, _ = monitor_and_estimate(P, gam, rng, calib)
        blk = slice(W, W + R)

        true_c = build_circuit(P, gam[blk])
        det, obs = true_c.compile_detector_sampler(seed=int(rng.integers(2**63))).sample(
            shots, separate_observables=True)
        obs = obs[:, 0]

        def nfail(m):
            return int((m.decode_batch(det)[:, 0] != obs).sum())

        f_static = nfail(m_static)
        fails["static"] += f_static
        # Estimator runs but its output is not used: weights stay at p0.
        fails["estimator_only"] += f_static
        fails["oracle"] += nfail(matcher(P, gam[blk]))
        cal_idx = (np.arange(W, W + R) // P.t_cal) * P.t_cal
        fails["periodic"] += nfail(matcher(P, gam[cal_idx]))
        for k in delays:
            idx = np.arange(W, W + R) - k
            ghat = np.clip(est.gamma_hat[idx], 0.0, None)
            fails[f"adaptive_d{k}"] += nfail(matcher(P, ghat))

    return fails, n_traj * shots
