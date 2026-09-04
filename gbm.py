"""Geometric Brownian motion generators for Monte Carlo simulation.

This module creates single-period and multi-step GBM paths, plus a few diagnostic
functions used to verify the simulated distribution against theoretical moments.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm


def terminal_price_array(S0, r, vol, T, n_paths, seed=None):
    """Generate terminal prices for a single-step GBM model.

    The model assumes log-normal terminal prices under the standard GBM dynamics.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(size=n_paths)
    mean = (r - (vol ** 2) / 2) * T
    std = vol * math.sqrt(T)
    S_T = S0 * np.exp(mean + (std * samples))
    return S_T


def antithetic_terminal_price_array(S0, r, vol, T, n_paths, seed=None):
    """Generate a standard sample and its antithetic counterpart.

    Antithetic sampling helps reduce variance by pairing each normal draw with its
    negative counterpart.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(size=n_paths)
    mean = (r - (vol ** 2) / 2) * T
    std = vol * math.sqrt(T)

    S_T = S0 * np.exp(mean + (std * samples))
    antithetic_S_T = S0 * np.exp(mean - (std * samples))
    return S_T, antithetic_S_T


def gbm_price_array(S0, r, vol, T, n_paths, n_steps, seed=None):
    """Return an array of simulated GBM paths across a time grid.

    The first column is fixed at the initial spot level, and the remaining
    columns represent cumulative log-price increments over time.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(size=(n_paths, n_steps))
    mean = (r - (vol ** 2) / 2) * (T / (n_steps - 1))
    std = vol * math.sqrt(T / (n_steps - 1))

    gbm_matrix = (samples * std) + mean
    gbm_matrix[:, 0] = np.log(S0)
    gbm_result = np.cumsum(gbm_matrix, axis=1)
    return np.exp(gbm_result)


def antithetic_gbm_price_array(S0, r, vol, T, n_paths, n_steps, seed=None):
    """Generate standard and antithetic GBM path matrices.

    Each path is paired with a mirrored shock trajectory to improve variance
    reduction in Monte Carlo estimates.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(size=(n_paths, n_steps))
    mean = (r - (vol ** 2) / 2) * (T / (n_steps - 1))
    std = vol * math.sqrt(T / (n_steps - 1))

    gbm_matrix = (samples * std) + mean
    gbm_matrix[:, 0] = np.log(S0)
    gbm_result = np.cumsum(gbm_matrix, axis=1)

    antithetic_gbm_matrix = (-samples * std) + mean
    antithetic_gbm_matrix[:, 0] = np.log(S0)
    antithetic_gbm_result = np.cumsum(antithetic_gbm_matrix, axis=1)

    return np.exp(gbm_result), np.exp(antithetic_gbm_result)


def mean_check(S_T, S0, r, vol, T, n_paths):
    """Check whether the sample mean is consistent with the theoretical GBM mean."""
    mean = (r - (vol ** 2) / 2) * T
    std = vol * math.sqrt(T)
    theo_mean = S0 * math.exp(mean + ((std ** 2) / 2))
    error = np.std(S_T) / math.sqrt(n_paths)

    if S_T.mean() < theo_mean - (3 * error) or S_T.mean() > theo_mean + (3 * error):
        print(f"Theoretical mean:{theo_mean}\n Observed mean:{S_T.mean()}\n Standard error:{error}\n Check: Failed")
    else:
        print(f"Theoretical mean:{theo_mean}\n Observed mean:{S_T.mean()}\n Standard error:{error}\n Check: Passed")
    return 0


def sanity_check(S_T, S0, r, vol, T, n_paths):
    """Plot the terminal price distribution against the lognormal benchmark."""
    plt.hist(S_T, bins=50, density=True, color='skyblue', edgecolor='black')

    mean = (r - (vol ** 2) / 2) * T
    std = vol * math.sqrt(T)
    s = std
    scale = S0 * np.exp(mean)
    x = np.linspace(0, np.max(S_T), 500)
    pdf_curve = lognorm.pdf(x, s, scale=scale)

    plt.plot(x, pdf_curve, 'r-', lw=2, label=f'Fitted PDF\n(s={s:.2f}, scale={scale:.2f})')
    plt.title('Terminal Price Fit Check')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Quick demonstration of the GBM path generator and distribution check.
    test_matrix = gbm_price_array(100, 0.05, 0.2, 1, 10 ** 6, 12)
    # mean_check(terminal_price_array(100, 0.05, 0.2, 1, 10 ** 6), 100, 0.05, 0.2, 1, 10 ** 6)
    sanity_check(test_matrix[:, -1], 100, 0.05, 0.2, 1, 10 ** 6)
