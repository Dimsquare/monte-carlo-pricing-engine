import math
import numpy as np
from gbm import gbm_price_array
from payoffs import European, Asian, Geometric_Asian, TerminalPrice, Barrier

def delta(S0,r,vol,T,n_paths,n_steps,option_payoff,h,seed):
    price_up = (option_payoff(gbm_price_array(S0+h,r,vol,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    price_down = (option_payoff(gbm_price_array(S0-h,r,vol,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    delta_arr = (price_up-price_down)/(2*h)
    delta_mean = np.mean(delta_arr)
    delta_error = np.std(delta_arr) / math.sqrt(n_paths)
    return (delta_mean, delta_error)

def vega(S0,r,vol,T,n_paths,n_steps,option_payoff,h,seed):
    price_up = (option_payoff(gbm_price_array(S0,r,vol+h,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    price_down = (option_payoff(gbm_price_array(S0,r,vol-h,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    vega_arr = (price_up-price_down)/(2*h)
    vega_mean = np.mean(vega_arr)
    vega_error = np.std(vega_arr) / math.sqrt(n_paths)
    return (vega_mean, vega_error)

def gamma(S0,r,vol,T,n_paths,n_steps,option_payoff,h,seed):
    price_up = (option_payoff(gbm_price_array(S0+h,r,vol,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    price = (option_payoff(gbm_price_array(S0,r,vol,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    price_down = (option_payoff(gbm_price_array(S0-h,r,vol,T,n_paths,n_steps,seed)))*(np.exp(-r * T))
    gamma_arr = (price_up - 2 * price + price_down) / h ** 2
    gamma_mean = np.mean(gamma_arr)
    gamma_error = np.std(gamma_arr) / math.sqrt(n_paths)
    return (gamma_mean, gamma_error)

