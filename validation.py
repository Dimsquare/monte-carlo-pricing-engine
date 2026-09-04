"""Validation and experimental analysis for the Monte Carlo pricer.

These functions compare simulated prices against closed-form benchmarks and show
variance-reduction diagnostics for different option structures.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from gbm import gbm_price_array
from pricer import price_option, antithetic_price_option, control_pricer
from closed_forms import bsm_call, bsm_put, kmv_put, kmv_call, d1d2
from payoffs import European, Asian, Barrier, Geometric_Asian, TerminalPrice
from greeks import delta,vega,gamma


def convergence_study(S0, K, r, vol, T, n_steps, option_payoff, closed_form_fn):
    """Compare MC prices against a closed-form benchmark as n grows."""
    n_array = np.logspace(3, 6, num=10)
    n_array = np.round(n_array).astype(int)

    vectorised_price = np.vectorize(price_option, otypes=[float, float])
    # vectorised_antithetic = np.vectorize(antithetic_price_option, otypes=[float, float])
    # vectorised_control = np.vectorize(control_pricer, otypes=[float, float])

    mc_prices, mc_se = vectorised_price(S0, r, vol, T, n_array, n_steps, option_payoff)
    # antithetic_prices, antithetic_se = vectorised_antithetic(S0,r,vol,T,n_array,payoff_fn)
    # control_prices, control_se = vectorised_control(S0,r,vol,T,n_array,payoff_fn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.errorbar(n_array, mc_prices, yerr=mc_se, fmt='o', capsize=5)

    closed_price = closed_form_fn(S0, K, r, vol, T)
    plt.axhline(y=closed_price, color='red', linestyle='--', linewidth=2, label='Closed Form Call Price')
    plt.xscale('log')
    plt.xlabel("n")
    plt.ylabel("Option Price")
    plt.title("MC price vs N compared to Closed Form value")
    plt.legend()

    observed_error = abs(mc_prices - closed_price)
    plt.subplot(1, 2, 2)
    plt.loglog(n_array, observed_error, color='blue', linestyle='-', marker='o', markersize=4, label='Observed |MC − BS|, plain')
    plt.loglog(n_array, mc_se, color='black', linestyle='--', marker='o', markersize=4, label='Monte Carlo Error')
    # plt.loglog(n_array, antithetic_se, color='red', linestyle='--', marker='o', markersize=4, label= 'Antithetic Error')
    # plt.loglog(n_array, control_se, color='yellow', linestyle='--', marker='o', markersize=4, label= 'Control Variate Error')
    plt.xlabel("n (Logscale)")
    plt.ylabel("Error (Logscale)")
    plt.title("Log-log plot of error against n")
    plt.legend()

    plt.tight_layout()
    return 0


def comparison_experiment(S0, r, vol, T, n_paths, n_steps, option_payoff,control_X,known_mean):
    """Compare standard error under different variance-reduction methods."""
    default_price, default_error = price_option(S0, r, vol, T, n_paths, n_steps, option_payoff)
    antithetic_price, antithetic_error = antithetic_price_option(S0, r, vol, T, n_paths, n_steps, option_payoff)
    control_price, control_error = control_pricer(S0, r, vol, T, n_paths, n_steps, option_payoff,control_X,known_mean)

    bar_category = ['Standard', 'Antithetic', 'Control Variate']
    bar_values = [default_error, antithetic_error, control_error]
    plt.bar(bar_category, bar_values, color='skyblue', width=0.6)
    plt.title('Standard error comparison at n=10⁶ of a European ATM call')
    plt.xlabel("Error reduction Method")
    plt.ylabel("Standard error")
    return 0


def barrier_variance_comparison(S0, K, r, vol, T, n_paths, n_steps):
    """Evaluate barrier-related variance reduction across different barrier levels."""
    test_values = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
    in_list = []
    out_list = []

    for i in range(0, len(test_values)):
        test_barrier_in = Barrier(100, True, test_values[i], True, True)
        control_call = European(100, True)
        # control_pricer compares against the UNDISCOUNTED control payoff,
        # so the benchmark must be un-discounted too.
        known_mean = bsm_call(S0, K, r, vol, T) * np.exp(r * T)

        in_price, in_error = price_option(S0, r, vol, T, n_paths, n_steps, test_barrier_in)
        in_control_price, in_control_error = control_pricer(
            S0, r, vol, T, n_paths, n_steps,
            test_barrier_in.return_payoff,
            control_call.return_payoff,
            known_mean,
        )

        test_barrier_out = Barrier(100, True, test_values[i], True, False)
        out_price, out_error = price_option(S0, r, vol, T, n_paths, n_steps, test_barrier_out)
        out_control_price, out_control_error = control_pricer(
            S0, r, vol, T, n_paths, n_steps,
            test_barrier_out.return_payoff,
            control_call.return_payoff,
            known_mean,
        )

        in_SE_ratio = in_control_error / in_error
        out_SE_ratio = out_control_error / out_error

        in_list.append(in_SE_ratio)
        out_list.append(out_SE_ratio)

    plt.plot(test_values, in_list, label="Up and In", color="Red")
    plt.plot(test_values, out_list, label="Up and Out", color="Blue")
    plt.xlabel("Barrier Levels")
    plt.ylabel("SE ratio")
    plt.legend()
    plt.show()
    return 0


def parity_check(S0, K, r, vol, T, n_paths, n_steps, option_payoff,k=3):
    """Check whether simulated call-put parity is consistent with theory."""
    call_price, call_error = price_option(S0, r, vol, T, n_paths, n_steps, option_payoff.return_payoff)
    option_payoff.call = False
    put_price, put_error = price_option(S0, r, vol, T, n_paths, n_steps, option_payoff.return_payoff)

    simulated_diff = call_price - put_price
    call_put_diff = S0 - K * np.exp(-r * T)
    error = math.sqrt((call_error ** 2) + (put_error ** 2))

    if call_put_diff < simulated_diff - (k * error) or call_put_diff > simulated_diff + (k * error):
        print(f"Theoretical difference:{call_put_diff}\n Observed difference:{simulated_diff}\n Standard error:{error}\n Check: Failed")
    else:
        print(f"Theoretical difference:{call_put_diff}\n Observed difference:{simulated_diff}\n Standard error:{error}\n Check: Passed")

    return 0

def greek_checks(S0, K, r, vol, T, n_paths, n_steps, option_payoff, h, seed, greek_type, reference_value,k):
    greek_value, greek_error = greek_type(S0, r, vol, T, n_paths, n_steps, option_payoff, h, seed)
    if reference_value < greek_value - (k * greek_error) or reference_value > greek_value + (k * greek_error):
        print(f"Theoretical value:{reference_value}\n Observed value:{greek_value}\n Standard error:{greek_error}\n Check: Failed")
    else:
        print(f"Theoretical value:{reference_value}\n Observed value:{greek_value}\n Standard error:{greek_error}\n Check: Passed")

    return 0

def CRN_demo(S0,r,vol,T,n_paths,n_steps,option_payoff,h,seed1,seed2):
    price_up = (option_payoff(gbm_price_array(S0+h,r,vol,T,n_paths,n_steps,seed1)))*(np.exp(-r * T))
    price_down = (option_payoff(gbm_price_array(S0-h,r,vol,T,n_paths,n_steps,seed2)))*(np.exp(-r * T))
    delta_arr = (price_up-price_down)/(2*h)
    delta_mean = np.mean(delta_arr)
    delta_error = np.std(delta_arr) / math.sqrt(n_paths)
    _ , CRN_delta = delta(S0,r,vol,T,n_paths,n_steps,option_payoff,h,seed1)
    print(f"CRN delta:{CRN_delta} vs Independent delta:{delta_error}; Ratio:{(delta_error/CRN_delta).round(2)}x")
    return 0
    



if __name__ == "__main__":
    # Example validation experiments for local debugging and comparison.
    test_obj = European(100,True)

    S0 = 100
    r = 0.05
    T = 1
    control_obj = TerminalPrice(100,True)
    known_mean = S0*np.exp(r*T)
    convergence_study(100,100,0.05,0.2,1,12, test_obj.return_payoff, bsm_call)
    plt.show()
    comparison_experiment(100,0.05,0.2,1,10**6,12,test_obj.return_payoff,control_obj.return_payoff,known_mean)
    plt.show()
    parity_check(100,100,0.05,0.2,1,10**6,12,test_obj)
    

    # barrier_variance_comparison(100, 100, 0.05, 0.2, 1, 10 ** 5, 500)
    CRN_demo(100,0.05,0.2,1,10**5,12,test_obj.return_payoff,1,40,42)
