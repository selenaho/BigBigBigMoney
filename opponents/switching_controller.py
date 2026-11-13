"""Deterministic switching, evaluated once before each hand."""

import random

class SwitchingController:
    def __init__(self, strategy_names, switch_every, *, randomize=False, seed=None):
        if not strategy_names:
            raise ValueError('At least one strategy is required')
        if switch_every < 1:
            raise ValueError('switch_every must be positive')
        self.strategy_names = tuple(strategy_names)
        self.switch_every = switch_every
        self.randomize = randomize
        self.rng = random.Random(seed)
        self.blocks = []
        if randomize and len(set(self.strategy_names)) < 2:
            raise ValueError('Random switching requires at least two distinct strategies')

    def strategy_for_hand(self, hand_index):
        if hand_index < 0:
            raise ValueError('hand_index must be nonnegative')
        block = hand_index // self.switch_every
        if self.randomize:
            # Cache blocks so repeated or out-of-order lookups remain stable.
            while len(self.blocks) <= block:
                previous = self.blocks[-1] if self.blocks else None
                choices = [name for name in self.strategy_names if name != previous]
                self.blocks.append(self.rng.choice(choices))
            return self.blocks[block]
        return self.strategy_names[block % len(self.strategy_names)]
