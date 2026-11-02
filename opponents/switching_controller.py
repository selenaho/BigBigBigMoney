"""Deterministic switching, evaluated once before each hand."""

class SwitchingController:
    def __init__(self, strategy_names, switch_every):
        if not strategy_names:
            raise ValueError('At least one strategy is required')
        if switch_every < 1:
            raise ValueError('switch_every must be positive')
        self.strategy_names = tuple(strategy_names)
        self.switch_every = switch_every

    def strategy_for_hand(self, hand_index):
        if hand_index < 0:
            raise ValueError('hand_index must be nonnegative')
        return self.strategy_names[(hand_index // self.switch_every) % len(self.strategy_names)]
