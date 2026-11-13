# Noise-Adaptive Surface-Code Decoding under Nonstationary Relaxation

Simulation code for *Real-Time Noise-Adaptive Quantum Error Correction under
Nonstationary Relaxation* (S. Choudhury, K. Sinha, A. K. Agrawal, UPES).

Every table and figure in the revised manuscript is produced by
`run_experiments.py`. Every logical error rate comes from circuit sampling with
[Stim](https://github.com/quantumlib/Stim) and decoding with
[PyMatching](https://github.com/oscarhiggott/PyMatching). Nothing is hard-coded.

```bash
pip install -r requirements.txt
python run_experiments.py            # full run: N = 20,000 trials per operating point
python run_experiments.py --quick    # N = 2,000, about 30 s
python run_experiments.py --only table4 fig4
```

Outputs: `results/*.csv` (raw failure counts, p_L and Wilson 95% CIs) and `figures/*.png`.

## Full simulation specification

### Code and experiment
* Rotated surface code, distance d ∈ {3, 5, 7, 9}. The circuit comes from Stim's
  `surface_code:rotated_memory_z` generator: Z-basis memory, standard
  nearest-neighbour CNOT schedule, X and Z stabilizers measured every round.
* **Number of QEC rounds:** each memory experiment (one trial) uses **R = d
  syndrome rounds**, followed by a transversal Z measurement of the data qubits.
  You can change this with `Params.rounds`.
* One trial means: prepare logical |0⟩, run R noisy rounds, measure the data qubits.

### Noise model and where it is inserted
At the **start of every syndrome round r**, every data qubit receives:
1. `DEPOLARIZE1(p_phys)`, the baseline Pauli fault;
2. `PAULI_CHANNEL_1(pX, pY, pZ)`, the Pauli twirl of amplitude damping with
   parameter Γ_r. Here pX = pY = Γ/4 and pZ = (2 − Γ − 2√(1−Γ))/4.

Every measurement (ancilla and final data) is flipped with probability p_phys.
In the default `phenomenological` model, gates and resets are ideal.
`Params.noise="circuit"` adds `DEPOLARIZE2(p_phys)` after every CNOT,
`DEPOLARIZE1(p_phys)` after every single-qubit gate, and reset flips at p_phys.

Relaxation follows the Ornstein–Uhlenbeck process of Eq. (1) with the values in
Table 2: γ̄ = 0.08, κ = 0.05, σ = 0.025 per cycle, and Δt = 1. Each path starts
from the stationary distribution and is clipped at γ ≥ 0. The per-round
damping parameter is Γ_r = 1 − exp(−λ γ_r Δt). Here λ = 0.25 is the fraction of
a cycle during which the data qubits are exposed to T₁ decay. Without this
factor (λ = 1), Γ ≈ 0.077 per round, which puts the code above threshold at
every distance.

Each trial has its own independent OU path. Cycles 0 … W−1 (W = 60) are an
estimator warm-up. The memory block occupies cycles W … W+R−1.

### How detection events are generated
Detection events are Stim's `DETECTOR`s: the XOR of consecutive measurements of
the same stabilizer (d_t = s_t ⊕ s_{t−1}). In the first round, a Z detector is
just the Z-stabilizer outcome. The final round compares Z stabilizers
reconstructed from data-qubit measurements. Samples come from
`compile_detector_sampler().sample(shots, separate_observables=True)`.

### Syndrome-derived observable and Kalman estimator
* **Monitoring stream:** for each trial, a separate Stim circuit with the same
  true noise path, covering all W+R cycles, is sampled once. The observable y_t
  is the fraction of detectors that fired in round t (24 detectors per bulk
  round at d = 5).
* **Calibration h(γ):** a lookup table built by sampling 12-round circuits at 25
  constant γ values in [0, 0.4], with 4,000 shots each and bulk rounds only
  (Section 5.2, route i).
* **Filter:** a scalar extended Kalman filter on γ (Eqs. 5–8):
  * a = 1 − κΔt and Q = σ²Δt;
  * y_t = h(γ_t) + v_t, with R_t = h(1−h)/n_det (binomial);
  * h is linearised by finite differences around the prediction;
  * the prior is the OU stationary law.

  The estimate is clipped at 0 before it is mapped to weights.

### Decoders (Section 7.2 ablation, plus periodic)
All decoders are PyMatching MWPM built from the Stim detector error model
(`decompose_errors=True`), with weights log((1−p)/p). They differ only in which
γ path is used to build the DEM:

| Decoder | γ used for round r of the block |
|---|---|
| Static | γ̄ (constant) |
| Estimator-only | γ̄ (filter runs, output unused: identical to static by construction) |
| Periodic | true γ at the last calibration cycle ⌊c/15⌋·15 |
| Oracle | true γ_r |
| Adaptive, delay k | Kalman estimate γ̂ at cycle c − k |

**All decoders decode the same detection-event samples** (paired comparison).

### Logical-failure criterion
A trial fails when MWPM's predicted flip of the logical Z observable (Stim
`OBSERVABLE_INCLUDE(0)`, a Z-logical string of data qubits) differs from the
actual flip in the sample. p̂_L = k/N, and the CI is Wilson 95% (Eq. 13).

### Sample sizes and seeds
* N = 20,000 trials per operating point: 200 independent OU paths × 100 shots.
* Master seed is 42. Each experiment uses a fixed offset:

| Experiment | Seed |
|---|---|
| Fig. 1 | 142 |
| Table 4 | 242 + i, where i is the distance index |
| Tables 5–6 | 342 |
| Fig. 3 | 442 |
| Fig. 4 | 542 + i, where i is the σ index |

* The h-calibration uses seeds 7 to 31.
* Fig. 3 uses one OU path and 80 consecutive start cycles, with 4,000 shots per start cycle.

### Mapping validation (Fig. 5)
`adaptive_qec/mapping.py` computes, exactly with density matrices, the detection
probability of a weight-4 stabilizer after one round of **exact** amplitude
damping and after its **Pauli twirl**:
* X check on |++++⟩;
* Z check averaged over the even-parity computational states, which is the
  code-state ensemble;
* Z check on |0000⟩ and on |1111⟩ separately.

## Repository layout
```
adaptive_qec/core.py      OU drift, circuits, calibration, Kalman filter, decoding, Wilson CI
adaptive_qec/mapping.py   exact vs twirled amplitude damping (Fig. 5)
run_experiments.py        regenerates every table and figure
results/  figures/        outputs
```

## Requirements
Python ≥ 3.10, stim, pymatching, numpy, pandas, matplotlib. Tested with stim
1.16 and pymatching 2.4.
