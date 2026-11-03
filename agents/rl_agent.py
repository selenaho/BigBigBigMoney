"""Standard per-action LinUCB agent for RLCard's Leduc Hold'em state."""

import numpy as np


class LinUCBAgent:
    """Learn a linear reward model for each action using terminal hand rewards.

    Decisions made during a hand are stored and receive that hand's final payoff
    when ``observe_reward`` is called. The matrices persist across hands.
    """

    use_raw = False

    def __init__(self, action_names, alpha=1.0):
        self.action_names = tuple(action_names)
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError('alpha must be nonnegative')
        self.dimension = 9
        self.A = {action: np.eye(self.dimension) for action in self.action_names}
        self.b = {action: np.zeros(self.dimension) for action in self.action_names}
        self.pending = []

    @staticmethod
    def _rank(card):
        if not card:
            return 0.0
        return {'J': 1.0, 'Q': 2.0, 'K': 3.0}.get(card[-1], 0.0) / 3.0

    def _features(self, state):
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
            score = float(x @ theta + self.alpha * np.sqrt(max(0.0, x @ inverse @ x)))
            if score > best_score:
                best_score, best_action = score, action_id
        self.pending.append((self.action_names[best_action], x))
        return best_action

    def observe_reward(self, reward):
        """Update all action decisions from the completed hand, then clear them."""
        reward = float(reward)
        for action, x in self.pending:
            self.A[action] += np.outer(x, x)
            self.b[action] += reward * x
        self.pending.clear()

    def eval_step(self, state):
        action = self.step(state)
        legal = state['legal_actions']
        return action, {'probs': {self.action_names[i]: float(i == action) for i in legal}}
