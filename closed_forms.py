"""Closed-form option pricing formulas used as reference benchmarks.

This module contains the Black-Scholes and Kim-Madison-Vasicek formulas used to
validate Monte Carlo estimates against analytic prices.
"""

import math
import numpy as np
from scipy.stats import norm


def d1d2(S0, K, r, vol, T):
    """Return the standard Black-Scholes d1 and d2 terms.

    Args:
        S0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        vol: Volatility.
        T: Time to maturity.
    """
    # The d-values are the standardised log-price ratios used in GBM pricing.
    d1 = (math.log(S0 / K) + (r + (vol ** 2) / 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return d1, d2


def bsm_call(S0, K, r, vol, T):
    """Price a European call under the Black-Scholes model."""
    d1, d2 = d1d2(S0, K, r, vol, T)
    N1 = norm.cdf(d1, loc=0, scale=1)
    N2 = norm.cdf(d2, loc=0, scale=1)
    call_price = S0 * N1 - K * np.exp(-r * T) * N2
    return call_price


def bsm_put(S0, K, r, vol, T):
    """Price a European put under the Black-Scholes model."""
    d1, d2 = d1d2(S0, K, r, vol, T)
    N1 = norm.cdf(-d1, loc=0, scale=1)
    N2 = norm.cdf(-d2, loc=0, scale=1)
    put_price = K * np.exp(-r * T) * N2 - S0 * N1
    return put_price


def bsm_delta(S0, K, r, vol, T, call=True):
    """Closed-form Black-Scholes delta, used as the benchmark for the MC Greek."""
    d1, _ = d1d2(S0, K, r, vol, T)
    return norm.cdf(d1) if call else norm.cdf(d1) - 1


def bsm_vega(S0, K, r, vol, T):
    """Closed-form Black-Scholes vega (per unit of volatility, not per 1%)."""
    d1, _ = d1d2(S0, K, r, vol, T)
    return S0 * norm.pdf(d1) * math.sqrt(T)


def bsm_gamma(S0, K, r, vol, T):
    """Closed-form Black-Scholes gamma (identical for calls and puts)."""
    d1, _ = d1d2(S0, K, r, vol, T)
    return norm.pdf(d1) / (S0 * vol * math.sqrt(T))


def kmv_call(S0, K, r, vol, T):
    """Price a European call under the Kemna–Vorst approximation."""
    # The KMV model uses adjusted volatility and drift terms for the transformed process.
    vol_g = vol / math.sqrt(3)
    r_g = (1 / 2) * (r - (1 / 6) * (vol ** 2))
    d1, d2 = d1d2(S0, K, r_g, vol_g, T)
    N1 = norm.cdf(d1, loc=0, scale=1)
    N2 = norm.cdf(d2, loc=0, scale=1)
    call_price = S0 * np.exp((r_g - r) * T) * N1 - K * np.exp(-r * T) * N2
    return call_price


def kmv_put(S0, K, r, vol, T):
    """Price a European put under the Kemna–Vorst approximation."""
    vol_g = vol / math.sqrt(3)
    r_g = (1 / 2) * (r - (1 / 6) * (vol ** 2))
    d1, d2 = d1d2(S0, K, r_g, vol_g, T)
    N1 = norm.cdf(-d1, loc=0, scale=1)
    N2 = norm.cdf(-d2, loc=0, scale=1)
    put_price = K * np.exp(-r * T) * N2 - S0 * np.exp((r_g - r) * T) * N1
    return put_price


def kmv_call_discrete(S0, K, r, vol, T, m):
    """Exact price of a DISCRETELY monitored geometric-average Asian call.

    Used to separate discretisation bias from MC noise.
    """
    mean = math.log(S0) + (r - (vol ** 2) / 2) * (T / 2)
    var = (vol ** 2) * T * (2 * m + 1) / (6 * (m + 1))
    sd = math.sqrt(var)

    d1 = (mean - math.log(K) + var) / sd
    d2 = d1 - sd
    return math.exp(-r * T) * (
        math.exp(mean + var / 2) * norm.cdf(d1) - K * norm.cdf(d2)
    )


if __name__ == "__main__":
    # Example usage for quick manual verification.
    print(kmv_call(100, 100, 0.05, 0.2, 1))
