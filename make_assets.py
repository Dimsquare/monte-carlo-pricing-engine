"""Regenerate every figure and headline number quoted in the README.

Run `python3 make_assets.py` to rebuild `assets/*.png` and print the numbers that
appear in the README tables. Every experiment is seeded, so the outputs are
reproducible run to run.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

from gbm import gbm_price_array
from pricer import price_option, antithetic_price_option, control_pricer
from payoffs import European, Asian, Geometric_Asian, Barrier, TerminalPrice
from closed_forms import (
    bsm_call, bsm_put, kmv_call, kmv_call_discrete,
    bsm_delta, bsm_vega, bsm_gamma,
)
from greeks import delta, vega, gamma

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Base contract used everywhere unless stated otherwise.
S0, K, R, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
SEED = 20240903
results = {}


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote assets/{name}")


def loglog_slope(x, y):
    """Least-squares slope of log y against log x."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ---------------------------------------------------------------- 1. GBM check
def fig_gbm_distribution():
    print("[1/6] GBM terminal distribution")
    paths = gbm_price_array(S0, R, VOL, T, 500_000, 13, SEED)
    S_T = paths[:, -1]

    theo_mean = S0 * math.exp(R * T)
    se = float(np.std(S_T) / math.sqrt(len(S_T)))
    results["gbm_moment_check"] = {
        "theoretical_mean": theo_mean,
        "observed_mean": float(S_T.mean()),
        "standard_error": se,
        "z_score": float((S_T.mean() - theo_mean) / se),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(S_T, bins=120, density=True, color="skyblue",
            edgecolor="none", label="Simulated $S_T$ (500,000 paths)")
    x = np.linspace(0, np.percentile(S_T, 99.9), 600)
    ax.plot(x, lognorm.pdf(x, VOL * math.sqrt(T),
                           scale=S0 * math.exp((R - VOL ** 2 / 2) * T)),
            "r-", lw=2, label="Exact lognormal density")
    ax.set_xlabel("Terminal price $S_T$")
    ax.set_ylabel("Density")
    ax.set_title("GBM terminal distribution vs exact lognormal law\n"
                 f"$S_0$={S0:.0f}, r={R}, $\\sigma$={VOL}, T={T:.0f}")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gbm_distribution.png")


# ------------------------------------------------------------- 2. Convergence
def fig_convergence():
    print("[2/6] Convergence to Black-Scholes")
    call = European(K, True)
    n_array = np.round(np.logspace(3, 6, 13)).astype(int)
    bs = bsm_call(S0, K, R, VOL, T)

    prices, ses = [], []
    for i, n in enumerate(n_array):
        p, e = price_option(S0, R, VOL, T, int(n), 13, call.return_payoff, SEED + i)
        prices.append(p)
        ses.append(e)
    prices, ses = np.array(prices), np.array(ses)
    abs_err = np.abs(prices - bs)

    results["convergence"] = {
        "black_scholes_call": float(bs),
        "n": n_array.tolist(),
        "mc_price": prices.tolist(),
        "mc_se": ses.tolist(),
        "se_loglog_slope": loglog_slope(n_array, ses),
        "abs_error_loglog_slope": loglog_slope(n_array, abs_err),
        "largest_n_price": float(prices[-1]),
        "largest_n_se": float(ses[-1]),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(n_array, prices, yerr=1.96 * ses, fmt="o", capsize=4,
                 color="tab:blue", label="MC price $\\pm$ 1.96 SE")
    ax1.axhline(bs, color="red", ls="--", lw=2,
                label=f"Black-Scholes = {bs:.4f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of paths $n$")
    ax1.set_ylabel("Call price")
    ax1.set_title("MC estimate converges to the closed form")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.loglog(n_array, ses, "k--o", ms=4, label="Reported standard error")
    ax2.loglog(n_array, abs_err, "b-o", ms=4, label="Observed $|MC - BS|$")
    ref = ses[0] * (n_array / n_array[0]) ** -0.5
    ax2.loglog(n_array, ref, color="grey", ls=":", lw=2,
               label="$n^{-1/2}$ reference")
    ax2.set_xlabel("Number of paths $n$ (log)")
    ax2.set_ylabel("Error (log)")
    ax2.set_title(f"Error decay: fitted slope = "
                  f"{results['convergence']['se_loglog_slope']:.3f}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    save(fig, "convergence.png")


# ------------------------------------------------ 3. Variance reduction ladder
def fig_variance_reduction():
    print("[3/6] Three-estimator variance reduction")
    call = European(K, True)
    control = TerminalPrice(K, True)
    known_mean = S0 * math.exp(R * T)          # E[S_T], undiscounted
    n_array = np.round(np.logspace(3, 6, 13)).astype(int)

    plain, anti, cv = [], [], []
    for i, n in enumerate(n_array):
        n = int(n)
        plain.append(price_option(S0, R, VOL, T, n, 13,
                                  call.return_payoff, SEED + i)[1])
        anti.append(antithetic_price_option(S0, R, VOL, T, n, 13,
                                            call.return_payoff, SEED + i)[1])
        cv.append(control_pricer(S0, R, VOL, T, n, 13, call.return_payoff,
                                 control.return_payoff, known_mean, SEED + i)[1])
    plain, anti, cv = np.array(plain), np.array(anti), np.array(cv)

    # Predicted CV ratio is sqrt(1 - rho^2) between payoff and control.
    paths = gbm_price_array(S0, R, VOL, T, 1_000_000, 13, SEED)
    y = call.return_payoff(paths) * math.exp(-R * T)
    x = control.return_payoff(paths)
    rho = float(np.corrcoef(x, y)[0, 1])

    # Predicted antithetic ratio is sqrt(1 + rho_a), rho_a = corr(payoff, mirror).
    from gbm import antithetic_gbm_price_array
    pa, pb = antithetic_gbm_price_array(S0, R, VOL, T, 500_000, 13, SEED)
    rho_a = float(np.corrcoef(call.return_payoff(pa),
                              call.return_payoff(pb))[0, 1])

    results["variance_reduction"] = {
        "n": n_array.tolist(),
        "se_plain": plain.tolist(),
        "se_antithetic": anti.tolist(),
        "se_control_variate": cv.tolist(),
        "slope_plain": loglog_slope(n_array, plain),
        "slope_antithetic": loglog_slope(n_array, anti),
        "slope_control_variate": loglog_slope(n_array, cv),
        "ratio_antithetic_measured": float(np.mean(anti / plain)),
        "ratio_cv_measured": float(np.mean(cv / plain)),
        "ratio_cv_predicted_sqrt_1_minus_rho2": math.sqrt(1 - rho ** 2),
        "ratio_antithetic_predicted_sqrt_1_plus_rho": math.sqrt(1 + rho_a),
        "corr_payoff_control": rho,
        "corr_payoff_antithetic_pair": rho_a,
        "se_at_1e6": {"plain": float(plain[-1]),
                      "antithetic": float(anti[-1]),
                      "control_variate": float(cv[-1])},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.loglog(n_array, plain, "-o", ms=4, color="tab:blue", label="Plain")
    ax1.loglog(n_array, anti, "-s", ms=4, color="tab:orange", label="Antithetic")
    ax1.loglog(n_array, cv, "-^", ms=4, color="tab:green", label="Control variate")
    ax1.set_xlabel("Number of paths $n$ (log)")
    ax1.set_ylabel("Standard error (log)")
    ax1.set_title("Same $n^{-1/2}$ slope, lower intercept")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    r = results["variance_reduction"]
    ax2.bar(["Plain", "Antithetic", "Control\nvariate"],
            [r["se_at_1e6"]["plain"], r["se_at_1e6"]["antithetic"],
             r["se_at_1e6"]["control_variate"]],
            color=["tab:blue", "tab:orange", "tab:green"], width=0.6)
    for i, v in enumerate([r["se_at_1e6"]["plain"], r["se_at_1e6"]["antithetic"],
                           r["se_at_1e6"]["control_variate"]]):
        ax2.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Standard error")
    ax2.set_title("SE at $n = 10^6$")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    save(fig, "variance_reduction.png")


# --------------------------------------------- 4. Asian discretisation bias
def fig_asian_bias():
    print("[4/6] Geometric Asian vs Kemna-Vorst")
    geo = Geometric_Asian(K, True)
    kv = kmv_call(S0, K, R, VOL, T)
    steps = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    n_paths = 200_000

    prices, ses, exact = [], [], []
    for i, m in enumerate(steps):
        p, e = price_option(S0, R, VOL, T, n_paths, m + 1,
                            geo.return_payoff, SEED + i)
        prices.append(p)
        ses.append(e)
        exact.append(kmv_call_discrete(S0, K, R, VOL, T, m))
    prices, ses, exact = np.array(prices), np.array(ses), np.array(exact)
    disc_bias = exact - kv           # deterministic discretisation bias
    mc_resid = prices - exact        # should be pure MC noise

    results["asian"] = {
        "kemna_vorst_continuous": float(kv),
        "n_steps": steps,
        "mc_price": prices.tolist(),
        "mc_se": ses.tolist(),
        "exact_discrete": exact.tolist(),
        "discretisation_bias": disc_bias.tolist(),
        "discretisation_bias_loglog_slope": loglog_slope(np.array(steps),
                                                         np.abs(disc_bias)),
        "mc_residual_vs_exact_in_se": (mc_resid / ses).tolist(),
        "max_abs_residual_in_se": float(np.max(np.abs(mc_resid / ses))),
        "finest": {"n_steps": steps[-1], "mc": float(prices[-1]),
                   "se": float(ses[-1]), "exact_discrete": float(exact[-1]),
                   "bias_vs_continuous": float(disc_bias[-1])},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(steps, prices, yerr=1.96 * ses, fmt="o", capsize=4,
                 color="tab:purple", zorder=3,
                 label="MC geometric Asian $\\pm$ 1.96 SE")
    ax1.plot(steps, exact, "-", color="tab:green", lw=2,
             label="Exact discrete closed form")
    ax1.axhline(kv, color="red", ls="--", lw=2,
                label=f"Kemna-Vorst continuous limit = {kv:.4f}")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Monitoring dates $m$")
    ax1.set_ylabel("Price")
    ax1.set_title("MC tracks the discrete law, which tends to the\n"
                  "continuous Kemna-Vorst price")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    slope = results["asian"]["discretisation_bias_loglog_slope"]
    ax2.loglog(steps, np.abs(disc_bias), "o-", color="tab:green", lw=2,
               label=f"Exact discretisation bias (slope {slope:.2f})")
    ax2.loglog(steps, np.abs(mc_resid), "s", color="tab:purple", ms=5,
               label="$|$MC $-$ exact discrete$|$")
    ax2.loglog(steps, ses, "k--", label="MC noise floor (1 SE)")
    ax2.set_xlabel("Monitoring dates $m$ (log)")
    ax2.set_ylabel("Absolute error (log)")
    ax2.set_title("Discretisation bias is exactly $O(1/m)$;\n"
                  "MC residual stays at the noise floor")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    save(fig, "asian_bias.png")


# ---------------------------------------------------- 5. Barrier consistency
def fig_barrier():
    print("[5/6] Barrier identities and CV regime")
    n_paths, n_steps = 100_000, 101
    levels = [105, 110, 115, 120, 130, 140, 150, 160, 180, 200, 220]
    vanilla_undisc = bsm_call(S0, K, R, VOL, T) * math.exp(R * T)
    control_call = European(K, True)

    in_ratio, out_ratio, parity_rows = [], [], []
    for i, B in enumerate(levels):
        b_in = Barrier(K, True, B, True, True)
        b_out = Barrier(K, True, B, True, False)
        seed = SEED + i

        pi, ei = price_option(S0, R, VOL, T, n_paths, n_steps,
                              b_in.return_payoff, seed)
        po, eo = price_option(S0, R, VOL, T, n_paths, n_steps,
                              b_out.return_payoff, seed)
        _, ci = control_pricer(S0, R, VOL, T, n_paths, n_steps,
                               b_in.return_payoff, control_call.return_payoff,
                               vanilla_undisc, seed)
        _, co = control_pricer(S0, R, VOL, T, n_paths, n_steps,
                               b_out.return_payoff, control_call.return_payoff,
                               vanilla_undisc, seed)
        in_ratio.append(ci / ei)
        out_ratio.append(co / eo)
        # in + out = vanilla, on identical paths (errors cancel exactly)
        pv, ev = price_option(S0, R, VOL, T, n_paths, n_steps,
                              control_call.return_payoff, seed)
        parity_rows.append({"B": B, "in": pi, "out": po, "sum": pi + po,
                            "vanilla_mc": pv, "residual": pi + po - pv})

    results["barrier"] = {
        "levels": levels,
        "se_ratio_up_in": in_ratio,
        "se_ratio_up_out": out_ratio,
        "in_plus_out_identity": parity_rows,
        "max_identity_residual": max(abs(r["residual"]) for r in parity_rows),
        # Crossover: first barrier where the control stops helping the
        # knock-IN leg more than the knock-OUT leg.
        "crossover_level": next(
            (levels[i] for i in range(len(levels))
             if in_ratio[i] > out_ratio[i]), None),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(levels, in_ratio, "-o", color="tab:red", label="Up-and-in")
    ax.plot(levels, out_ratio, "-s", color="tab:blue", label="Up-and-out")
    ax.axhline(1.0, color="grey", ls=":", lw=1.5, label="No benefit (ratio = 1)")
    ax.set_xlabel("Barrier level $B$")
    ax.set_ylabel("SE(control variate) / SE(plain)")
    ax.set_title("Where the vanilla-call control variate earns its keep\n"
                 "(lower is better; the control works on whichever leg still "
                 "resembles the vanilla)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "barrier_cv_regime.png")


# --------------------------------------------------------------- 6. Greeks
def fig_greeks():
    print("[6/6] Greeks and common random numbers")
    call = European(K, True)
    n_paths, n_steps = 500_000, 13

    rows = []
    d_mc, d_se = delta(S0, R, VOL, T, n_paths, n_steps, call.return_payoff, 0.5, SEED)
    rows.append(("Delta", d_mc, d_se, bsm_delta(S0, K, R, VOL, T)))
    v_mc, v_se = vega(S0, R, VOL, T, n_paths, n_steps, call.return_payoff, 0.005, SEED)
    rows.append(("Vega", v_mc, v_se, bsm_vega(S0, K, R, VOL, T)))
    g_mc, g_se = gamma(S0, R, VOL, T, n_paths, n_steps, call.return_payoff, 2.0, SEED)
    rows.append(("Gamma", g_mc, g_se, bsm_gamma(S0, K, R, VOL, T)))

    results["greeks"] = [
        {"greek": n, "mc": float(m), "se": float(s), "closed_form": float(c),
         "z": float((m - c) / s)} for n, m, s, c in rows
    ]

    # CRN A/B: same seed both bumps vs independent seeds.
    h = 0.5
    disc = math.exp(-R * T)
    up_crn = call.return_payoff(gbm_price_array(S0 + h, R, VOL, T, n_paths, n_steps, SEED)) * disc
    dn_crn = call.return_payoff(gbm_price_array(S0 - h, R, VOL, T, n_paths, n_steps, SEED)) * disc
    up_ind = call.return_payoff(gbm_price_array(S0 + h, R, VOL, T, n_paths, n_steps, SEED)) * disc
    dn_ind = call.return_payoff(gbm_price_array(S0 - h, R, VOL, T, n_paths, n_steps, SEED + 999)) * disc

    crn = (up_crn - dn_crn) / (2 * h)
    ind = (up_ind - dn_ind) / (2 * h)
    crn_se = float(np.std(crn) / math.sqrt(n_paths))
    ind_se = float(np.std(ind) / math.sqrt(n_paths))
    results["crn"] = {
        "h": h, "n_paths": n_paths,
        "crn_delta": float(crn.mean()), "crn_se": crn_se,
        "independent_delta": float(ind.mean()), "independent_se": ind_se,
        "se_ratio": ind_se / crn_se,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(70, 130, 13)
    mc_d = [delta(s, R, VOL, T, 200_000, n_steps, call.return_payoff, 0.5, SEED)
            for s in grid]
    ax1.errorbar(grid, [m for m, _ in mc_d], yerr=[1.96 * e for _, e in mc_d],
                 fmt="o", capsize=3, color="tab:blue", label="MC delta (CRN) $\\pm$ 1.96 SE")
    fine = np.linspace(70, 130, 200)
    ax1.plot(fine, [bsm_delta(s, K, R, VOL, T) for s in fine], "r-", lw=2,
             label="Black-Scholes delta")
    ax1.set_xlabel("Spot $S_0$")
    ax1.set_ylabel("Delta")
    ax1.set_title("Finite-difference delta across the strike")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.bar(["Independent\ndraws", "Common random\nnumbers"], [ind_se, crn_se],
            color=["tab:red", "tab:green"], width=0.55)
    ax2.set_yscale("log")
    ax2.set_ylabel("Standard error of delta (log)")
    ax2.set_title(f"CRN cuts delta SE by {ind_se / crn_se:.1f}x")
    for i, v in enumerate([ind_se, crn_se]):
        ax2.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    ax2.grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    save(fig, "greeks.png")


# ------------------------------------------------------- numbers-only checks
def numeric_checks():
    print("[+] Put-call parity")
    n = 1_000_000
    call, put = European(K, True), European(K, False)
    cp, ce = price_option(S0, R, VOL, T, n, 13, call.return_payoff, SEED)
    pp, pe = price_option(S0, R, VOL, T, n, 13, put.return_payoff, SEED + 1)
    lhs = cp - pp
    rhs = S0 - K * math.exp(-R * T)
    se = math.sqrt(ce ** 2 + pe ** 2)
    results["put_call_parity"] = {
        "mc_call": cp, "mc_put": pp, "mc_difference": lhs,
        "theoretical_difference": rhs, "combined_se": se,
        "residual": lhs - rhs, "residual_in_se": (lhs - rhs) / se,
        "bs_call": float(bsm_call(S0, K, R, VOL, T)),
        "bs_put": float(bsm_put(S0, K, R, VOL, T)),
    }

    print("[+] Barrier limiting cases")
    deep = Barrier(K, True, 1e6, True, False)      # unreachable barrier
    near = Barrier(K, True, 100.0001, True, False)  # barrier at spot
    dp, de = price_option(S0, R, VOL, T, 200_000, 101, deep.return_payoff, SEED)
    npv, ne = price_option(S0, R, VOL, T, 200_000, 101, near.return_payoff, SEED)
    results["barrier_limits"] = {
        "up_and_out_B_1e6": {"price": dp, "se": de,
                             "bs_call": float(bsm_call(S0, K, R, VOL, T))},
        "up_and_out_B_at_spot": {"price": npv, "se": ne},
    }

    print("[+] Asian ordering")
    ar, gr = Asian(K, True), Geometric_Asian(K, True)
    ap, ae = price_option(S0, R, VOL, T, 500_000, 13, ar.return_payoff, SEED)
    gp, ge = price_option(S0, R, VOL, T, 500_000, 13, gr.return_payoff, SEED)
    results["asian_ordering"] = {
        "arithmetic": {"price": ap, "se": ae},
        "geometric": {"price": gp, "se": ge},
    }


if __name__ == "__main__":
    fig_gbm_distribution()
    fig_convergence()
    fig_variance_reduction()
    fig_asian_bias()
    fig_barrier()
    fig_greeks()
    numeric_checks()

    out = os.path.join(ASSETS, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nAll figures in assets/  |  numbers in assets/results.json")
    print(json.dumps(results, indent=2, default=float))
