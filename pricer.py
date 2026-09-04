"""Monte Carlo pricing utilities for equity-style option payoffs.

This module prices options by simulating GBM paths and applying a payoff function
before discounting the cash flows back to present value.
"""

import math
import numpy as np
from gbm import gbm_price_array, antithetic_gbm_price_array
from payoffs import European, Asian, Barrier, TerminalPrice, Geometric_Asian
from closed_forms import kmv_call, kmv_put, bsm_call, bsm_put


def price_option(S0, r, vol, T, n_paths, n_steps, option_payoff, seed=None):
    """Estimate the discounted expected payoff for a given option payoff object."""
    # Simulate a full GBM path matrix for all required paths and time steps.
    S_T = gbm_price_array(S0, r, vol, T, n_paths, n_steps, seed)
    payoff = option_payoff(S_T)
    discounted_pa = payoff * (np.exp(-r * T))
    discounted_avg = discounted_pa.mean()
    discounted_error = np.std(discounted_pa) / math.sqrt(n_paths)
    return (discounted_avg, discounted_error)


def antithetic_price_option(S0, r, vol, T, n_paths, n_steps, option_payoff, seed=None):
    """Use antithetic variates to reduce estimator variance in the MC price."""
    S_T, antithetic_S_T = antithetic_gbm_price_array(
        S0, r, vol, T, int(n_paths / 2), n_steps, seed
    )

    # Average the payoff across each pair of positively and negatively correlated paths.
    payoff = (option_payoff(S_T) + option_payoff(antithetic_S_T)) / 2
    discounted_pa = payoff * (np.exp(-r * T))
    discounted_avg = discounted_pa.mean()
    discounted_error = np.std(discounted_pa) / math.sqrt(n_paths / 2)
    return (discounted_avg, discounted_error)


def control_pricer(S0, r, vol, T, n_paths, n_steps, option_Y, control_X, known_mean, seed=None):
    """Apply a control variate to reduce Monte Carlo variance.

    option_Y is the target payoff, while control_X is a correlated instrument with
    known expectation. The sample covariance is used to compute the optimal beta.
    """
    S_T = gbm_price_array(S0, r, vol, T, n_paths, n_steps, seed)
    control = control_X(S_T)
    payoff = option_Y(S_T)
    discounted_pa = payoff * (np.exp(-r * T))

    cov_matrix = np.cov(control, discounted_pa, ddof=0)
    cov = cov_matrix[0, 1]
    beta = cov / cov_matrix[0, 0]

    adj_discounted_pa = discounted_pa - beta * (control - known_mean)
    adj_discounted_avg = adj_discounted_pa.mean()
    adj_discounted_error = np.std(adj_discounted_pa) / math.sqrt(n_paths)
    return (adj_discounted_avg, adj_discounted_error)


if __name__ == "__main__":
    # Simple smoke-test examples for pricing functions.
    S0, K, r, vol, T = 100, 100, 0.05, 0.2, 1.0
    test_obj = European(K, True)
    control_obj = TerminalPrice(K, True)

    print(price_option(S0, r, vol, T, 10 ** 6, 12, test_obj.return_payoff))
    print(antithetic_price_option(S0, r, vol, T, 10 ** 6, 12, test_obj.return_payoff))

    known_mean = S0 * np.exp(r * T)
    print(control_pricer(S0, r, vol, T, 10 ** 6, 12,
                         test_obj.return_payoff, control_obj.return_payoff, known_mean))
