"""Standard per-action LinUCB agent for RLCard's Leduc Hold'em state."""

from collections import deque

import numpy as np


class LinUCBAgent:
    """Learn a linear reward model for each action using terminal hand rewards.

    Decisions made during a hand are stored and receive that hand's final payoff
    when ``observe_reward`` is called. The matrices persist across hands.
    """

    use_raw = False

    def __init__(self, action_names, alpha=1.0, variant='standard',
                 window_size=None, discount_factor=None):
        self.action_names = tuple(action_names)
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError('alpha must be nonnegative')
        if variant not in ('standard', 'windowed', 'discounted'):
            raise ValueError("variant must be 'standard', 'windowed', or 'discounted'")
        if variant == 'windowed' and (window_size is None or int(window_size) < 1):
            raise ValueError('window_size must be positive for windowed LinUCB')
        if variant == 'discounted' and (discount_factor is None or
                                         not 0 < float(discount_factor) < 1):
            raise ValueError('discount_factor must be between 0 and 1 for discounted LinUCB')
        self.variant = variant
        self.window_size = int(window_size) if variant == 'windowed' else None
        self.discount_factor = float(discount_factor) if variant == 'discounted' else None
        self.name = {'standard': 'linucb', 'windowed': 'linucb_windowed',
                     'discounted': 'linucb_discounted'}[variant]
        self.dimension = 9
        # Each action has its own linear reward model; identity matrices start with uncertainty.
        self._reset_statistics()
        # Hold decisions until the hand's terminal payoff is known.
        self.pending = []
        self.history = deque(maxlen=self.window_size) if variant == 'windowed' else None

    def _reset_statistics(self):
        self.A = {action: np.eye(self.dimension) for action in self.action_names}
        self.b = {action: np.zeros(self.dimension) for action in self.action_names}

    @staticmethod
    def _rank(card):
        if not card:
            return 0.0
        return {'J': 1.0, 'Q': 2.0, 'K': 3.0}.get(card[-1], 0.0) / 3.0

    def _features(self, state):
        # Build a compact context from information visible to the acting player.
        raw = state['raw_obs']
        hand = raw.get('hand')
        public = raw.get('public_card')
        hand_rank = self._rank(hand)
        public_rank = self._rank(public)
        paired = float(bool(hand and public and hand[-1] == public[-1]))
        chips = raw.get('all_chips', [0, 0])
        total_pot = float(sum(chips))
        own_chips = float(raw.get('my_chips', 0))
        round_number = float(raw.get('round_counter', 0))
        # Bounded scales keep the linear model numerically well behaved.
        return np.asarray([
            1.0, hand_rank, public_rank, paired,
            min(total_pot / 20.0, 1.0), min(own_chips / 20.0, 1.0),
            min(round_number / 2.0, 1.0),
            float(any(self.action_names[i] == 'raise' for i in state['legal_actions'])),
            float(any(self.action_names[i] == 'call' for i in state['legal_actions'])),
        ], dtype=float)

    def step(self, state):
        x = self._features(state)
        legal = list(state['legal_actions'])
        if not legal:
            raise ValueError('No legal actions available')
        best_action = None
        best_score = float('-inf')
        for action_id in legal:
            name = self.action_names[action_id]
            inverse = np.linalg.inv(self.A[name])
            theta = np.linalg.solve(self.A[name], self.b[name])
            # LinUCB adds an uncertainty bonus to the predicted reward.
            score = float(x @ theta + self.alpha * np.sqrt(max(0.0, x @ inverse @ x)))
            if score > best_score:
                best_score, best_action = score, action_id
        self.pending.append((self.action_names[best_action], x))
        return best_action

    def observe_reward(self, reward):
        """Update all action decisions from the completed hand, then clear them."""
        reward = float(reward)
        # Apply feedback in decision order; each decision from this hand gets its payoff.
        for action, x in self.pending:
            if self.variant == 'windowed':
                self.history.append((action, x.copy(), reward))
                self._rebuild_windowed_statistics()
            elif self.variant == 'discounted':
                # Decay old data while keeping the identity regularizer fixed.
                gamma = self.discount_factor
                for name in self.action_names:
                    self.A[name] = gamma * self.A[name] + (1 - gamma) * np.eye(self.dimension)
                    self.b[name] *= gamma
                self.A[action] += np.outer(x, x)
                self.b[action] += reward * x
            else:
                self.A[action] += np.outer(x, x)
                self.b[action] += reward * x
        self.pending.clear()

    def _rebuild_windowed_statistics(self):
        """Rebuild models from the most recent N completed decision samples."""
        self._reset_statistics()
        for action, x, reward in self.history:
            self.A[action] += np.outer(x, x)
            self.b[action] += reward * x

    def eval_step(self, state):
        action = self.step(state)
        legal = state['legal_actions']
        return action, {'probs': {self.action_names[i]: float(i == action) for i in legal}}
