#!/usr/bin/env python3
"""Regenerate every table and figure of the manuscript.

    python run_experiments.py            # full run, N = 20,000 trials per point
    python run_experiments.py --quick    # N = 2,000 trials per point (smoke test)

Outputs go to results/ (CSV) and figures/ (PNG). All randomness is seeded
from --seed (default 42); each experiment derives its own fixed sub-seed so
that experiments can be re-run individually with identical results.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_qec.core import (Params, build_circuit, calibrate_h, matcher,
                               monitor_and_estimate, ou_trajectory, run_point, wilson)
from adaptive_qec.mapping import mapping_validation

BASE = Params()          # d=5, p_phys=0.009, sigma=0.025, Table 2 OU parameters


def ci_row(prefix, k, n):
    p, lo, hi = wilson(k, n)
    return {f"{prefix}_fails": k, f"{prefix}_pL": p, f"{prefix}_lo": lo, f"{prefix}_hi": hi}


def ratio(a, b):
    return a / b if b > 0 else np.inf


# ---------------------------------------------------------------- Figure 1
def fig1_tracking(seed, out):
    P = BASE
    rng = np.random.default_rng(seed)
    gam = ou_trajectory(300, P, rng)
    est, y = monitor_and_estimate(P, gam, rng, calibrate_h(P))
    sd = np.sqrt(est.P)
    pd.DataFrame({"cycle": np.arange(len(gam)), "gamma_true": gam, "gamma_hat": est.gamma_hat,
                  "kalman_sd": sd, "y_obs": y}).to_csv(f"{out}/fig1_tracking.csv", index=False)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    t = np.arange(len(gam))
    ax[0].plot(t, gam, color="0.4", label="true γ(t)")
    ax[0].plot(t, est.gamma_hat, "--", color="tab:blue", label="estimated γ̂(t)")
    ax[0].fill_between(t, est.gamma_hat - 1.96 * sd, est.gamma_hat + 1.96 * sd, color="tab:blue", alpha=.2,
                       label="95% filter interval")
    ax[0].axhline(P.gamma_bar, ls=":", color="k", lw=.8)
    ax[0].set_ylabel("γ (cycle⁻¹)"); ax[0].legend(fontsize=8)
    ax[1].plot(t, np.abs(gam - est.gamma_hat), color="tab:red", label="|γ − γ̂|")
    ax[1].plot(t, sd, "--", color="tab:blue", label="√P_t")
    ax[1].set_xlabel("QEC cycle"); ax[1].set_ylabel("error"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig1_tracking.png", dpi=200); plt.close(fig)
    rmse = float(np.sqrt(np.mean((gam[50:] - est.gamma_hat[50:]) ** 2)))
    cover = float(np.mean(np.abs(gam[50:] - est.gamma_hat[50:]) <= 1.96 * sd[50:]))
    return {"rmse_after_50": rmse, "coverage_95": cover}


# ---------------------------------------------------------------- Table 4
def table4_distance(seed, n_traj, shots, out):
    rows = []
    for i, d in enumerate((3, 5, 7, 9)):
        P = replace(BASE, d=d)
        f, n = run_point(P, n_traj, shots, seed + i)
        r = {"d": d, "rounds": P.R, "N": n}
        r.update(ci_row("static", f["static"], n)); r.update(ci_row("adaptive", f["adaptive_d0"], n))
        r.update(ci_row("oracle", f["oracle"], n))
        r["gain"] = ratio(f["static"], f["adaptive_d0"])
        rows.append(r); print("  d=%d done" % d, flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/table4_distance.csv", index=False)


# ---------------------------------------------------------------- Table 5 + Table 6
def table5_6(seed, n_traj, shots, out):
    delays = (0, 1, 2, 4, 6, 8)
    f, n = run_point(BASE, n_traj, shots, seed, delays=delays)
    rows = []
    for k in delays:
        r = {"delay": k, "N": n}; r.update(ci_row("adaptive", f[f"adaptive_d{k}"], n))
        r["ratio_to_static"] = ratio(f[f"adaptive_d{k}"], f["static"]); rows.append(r)
    r = {"delay": "static", "N": n}; r.update(ci_row("adaptive", f["static"], n)); r["ratio_to_static"] = 1.0
    rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/table5_delay.csv", index=False)

    abl = []
    for name, key in [("Static decoder", "static"), ("Estimator-only (weights fixed)", "estimator_only"),
                      ("Periodic recalibration (T=15)", "periodic"),
                      ("Oracle Gamma_t (true, zero lag)", "oracle"), ("Full adaptive pipeline", "adaptive_d0")]:
        p, lo, hi = wilson(f[key], n)
        abl.append({"configuration": name, "fails": f[key], "N": n, "pL": p, "lo": lo, "hi": hi,
                    "gain_vs_static": ratio(f["static"], f[key])})
    pd.DataFrame(abl).to_csv(f"{out}/table6_ablation.csv", index=False)


# ---------------------------------------------------------------- Figure 3
def fig3_time(seed, shots, out, n_start=80):
    """One OU realisation. For each start cycle c a d-round memory block covering
    cycles [W+c, W+c+R) is sampled `shots` times and decoded by each decoder."""
    P = BASE
    rng = np.random.default_rng(seed)
    R, W = P.R, P.warmup
    gam = ou_trajectory(W + n_start + R, P, rng)
    est, _ = monitor_and_estimate(P, gam, rng, calibrate_h(P))
    ghat = np.clip(est.gamma_hat, 0, None)
    m_static = matcher(P, np.full(R, P.gamma_bar))
    rows = []
    for c in range(n_start):
        idx = np.arange(W + c, W + c + R)
        circ = build_circuit(P, gam[idx])
        det, obs = circ.compile_detector_sampler(seed=int(rng.integers(2**63))).sample(shots, separate_observables=True)
        obs = obs[:, 0]
        res = {"cycle": c, "gamma_true": gam[W + c], "gamma_hat": ghat[W + c]}
        for name, m in [("static", m_static), ("periodic", matcher(P, gam[(idx // P.t_cal) * P.t_cal])),
                        ("adaptive", matcher(P, ghat[idx]))]:
            res[name] = float(np.mean(m.decode_batch(det)[:, 0] != obs))
        rows.append(res)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/fig3_time.csv", index=False)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(df.cycle, df.gamma_true, color="tab:red", label="true γ(t)")
    ax[0].plot(df.cycle, df.gamma_hat, "--", color="tab:blue", label="estimated γ̂(t)")
    ax[0].axhline(P.gamma_bar, ls=":", color="k", lw=.8); ax[0].legend(fontsize=8); ax[0].set_ylabel("γ")
    ax[1].plot(df.cycle, df.static, color="tab:red", label="static")
    ax[1].plot(df.cycle, df.periodic, "-.", color="tab:orange", label=f"periodic (T={P.t_cal})")
    ax[1].plot(df.cycle, df.adaptive, "--", color="tab:blue", label="adaptive")
    ax[1].set_xlabel("start cycle of memory block"); ax[1].set_ylabel("logical error rate p_L")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig3_time.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def fig4_sigma(seed, n_traj, shots, out):
    rows = []
    for i, s in enumerate((0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)):
        P = replace(BASE, sigma=s)
        f, n = run_point(P, n_traj, shots, seed + i)
        r = {"sigma": s, "N": n}
        r.update(ci_row("static", f["static"], n)); r.update(ci_row("adaptive", f["adaptive_d0"], n))
        r["gain"] = ratio(f["static"], f["adaptive_d0"]); rows.append(r)
        print("  sigma=%.3f done" % s, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/fig4_sigma.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, c, mk in [("static", "tab:red", "o"), ("adaptive", "tab:blue", "s")]:
        ax[0].errorbar(df.sigma, df[f"{k}_pL"], yerr=[df[f"{k}_pL"] - df[f"{k}_lo"], df[f"{k}_hi"] - df[f"{k}_pL"]],
                       color=c, marker=mk, capsize=3, label=k)
    ax[0].set_yscale("log"); ax[0].set_xlabel("OU σ"); ax[0].set_ylabel("p_L"); ax[0].legend()
    ax[1].plot(df.sigma, df.gain, "D-", color="teal"); ax[1].axhline(1, ls=":", color="k")
    ax[1].set_xlabel("OU σ"); ax[1].set_ylabel("G = p_L(static) / p_L(adaptive)")
    fig.tight_layout(); fig.savefig("figures/fig4_sigma.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig5_mapping(out):
    df = mapping_validation(BASE)
    df.to_csv(f"{out}/fig5_mapping.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for col, lab, mk in [("X_check", "X check, |++++>", "o"), ("Z_check_code", "Z check, even-parity mixture", "s")]:
        ax[0].scatter(df[f"{col}_exact"], df[f"{col}_twirl"], marker=mk, label=lab)
    lim = [0, max(df.X_check_exact.max(), df.Z_check_code_exact.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=.8); ax[0].set_xlabel("detection prob., exact amplitude damping")
    ax[0].set_ylabel("detection prob., twirled Pauli model"); ax[0].legend(fontsize=8)
    ax[1].plot(df.gamma, 100 * df.X_check_relerr, "o-", label="X check")
    ax[1].plot(df.gamma, 100 * df.Z_check_code_relerr, "s-", label="Z check (code-state average)")
    ax[1].set_xlabel("γ (cycle⁻¹)"); ax[1].set_ylabel("relative mismatch (%)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig5_mapping.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: fig1 table4 table56 fig3 fig4 fig5")
    a = ap.parse_args()
    n_traj, shots = (20, 100) if a.quick else (200, 100)      # N = n_traj * shots
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    todo = a.only or ["fig1", "table4", "table56", "fig3", "fig4", "fig5"]
    s = a.seed
    if "fig1" in todo:
        print("Figure 1", fig1_tracking(s + 100, "results"))
    if "table4" in todo:
        print("Table 4"); table4_distance(s + 200, n_traj, shots, "results")
    if "table56" in todo:
        print("Tables 5-6"); table5_6(s + 300, n_traj, shots, "results")
    if "fig3" in todo:
        print("Figure 3"); fig3_time(s + 400, 400 if a.quick else 4000, "results")
    if "fig4" in todo:
        print("Figure 4"); fig4_sigma(s + 500, n_traj, shots, "results")
    if "fig5" in todo:
        print("Figure 5"); fig5_mapping("results")
    print("done")


if __name__ == "__main__":
    main()
