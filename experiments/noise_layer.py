"""Seeded action-label noise for learner-visible opponent behavior only."""

import math
import random


class ObservationNoise:
    """Replace an action with a uniformly chosen legal alternative with probability rate.

    The proposal leaves the corruption distribution unspecified; this implements
    categorical substitution. A single legal action cannot be corrupted. Call
    once per opponent decision and reuse the result in callbacks and histories.
    """

    def __init__(self, rate=0.0, seed=44):
        self.rate = float(rate)
        if not math.isfinite(self.rate) or not 0 <= self.rate <= 1:
            raise ValueError('noise_rate must be finite and between 0 and 1')
        self.rng = random.Random(seed)

    def observe_action(self, action, legal_actions):
        legal = list(legal_actions)
        if action not in legal:
            raise ValueError('Observed action must be legal')
        alternatives = [candidate for candidate in legal if candidate != action]
        if self.rate == 0 or not alternatives:
            return action
        if self.rng.random() < self.rate:
            return self.rng.choice(alternatives)
        return action
