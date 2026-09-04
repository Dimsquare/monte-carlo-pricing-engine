"""Payoff structures for different option styles.

Each class defines how a simulated price path is reduced to a terminal payoff for
Monte Carlo pricing.
"""

import numpy as np
from abc import ABC


class Payoffs(ABC):
    """Abstract base class that defines the shared payoff logic."""

    def __init__(self, K, call):
        self.K = K
        self.call = call

    def payoff_call(self, S_T, K):
        """Return the intrinsic payoff of a call option."""
        call_payoff = np.maximum(S_T - K, 0)
        return call_payoff

    def payoff_put(self, S_T, K):
        """Return the intrinsic payoff of a put option."""
        put_payoff = np.maximum(K - S_T, 0)
        return put_payoff

    def payoff_fn(self, S_T):
        """Dispatch to the correct call/put payoff based on option type."""
        if self.call:
            return self.payoff_call(S_T, self.K)
        else:
            return self.payoff_put(S_T, self.K)


class TerminalPrice(Payoffs):
    """Payoff based only on the final asset value."""

    def return_payoff(self, S_T):
        self.K = 0
        S_T = S_T[:, -1]
        return self.payoff_fn(S_T)


class European(Payoffs):
    """Standard European option payoff using the terminal asset value."""

    def return_payoff(self, S_T):
        S_T = S_T[:, -1]
        return self.payoff_fn(S_T)


class Asian(Payoffs):
    """Arithmetic Asian option payoff based on the path average."""

    def return_payoff(self, S_T):
        S_average = np.mean(S_T, axis=1)
        return self.payoff_fn(S_average)


class Geometric_Asian(Payoffs):
    """Geometric Asian option payoff based on the geometric mean."""

    def return_payoff(self, S_T):
        S_geom_avg = np.exp(np.mean(np.log(S_T), axis=1))
        return self.payoff_fn(S_geom_avg)


class Barrier(Payoffs):
    """Barrier option payoff that depends on whether the path breaches a barrier."""

    def __init__(self, K, call: bool, B, up: bool, type_in: bool):
        super().__init__(K, call)
        self.B = B
        self.up = up
        self.type_in = type_in

    def return_payoff(self, S_T):
        """Apply the knock-in or knock-out condition using the barrier level."""
        S_final = S_T[:, -1]

        if self.up:
            S_max = np.max(S_T, axis=1)
            hit = S_max > self.B
        else:
            S_min = np.min(S_T, axis=1)
            hit = S_min < self.B

        if self.type_in:
            alive = hit
            return alive * self.payoff_fn(S_final)
        else:
            alive = ~hit
            return alive * self.payoff_fn(S_final)
