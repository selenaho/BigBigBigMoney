"""Define the opponent-policy interface, RLCard adapter, and demonstration policies."""

import random
from typing import Protocol
from agents.rule_based_agent import RuleBasedAgent
from opponents.styles import STRATEGY_NAMES, StylePolicy


class OpponentPolicy(Protocol):
    def select_action(self, state, rng: random.Random) -> int:
        """Receive the acting player's observation and return a legal action ID."""
        ...


class RandomPolicy:
    def select_action(self, state, rng):
        return rng.choice(list(state['legal_actions']))


class RuleBasedPolicy:
    def __init__(self, action_names):
        self.agent = RuleBasedAgent(action_names)

    def select_action(self, state, rng):
        return self.agent.step(state)


class OpponentAgent:
    use_raw = False

    def __init__(self, policy, seed):
        self.policy = policy
        self.rng = random.Random(seed)

    def step(self, state):
        action = self.policy.select_action(state, self.rng)
        if action not in state['legal_actions']:
            raise ValueError(f'Opponent selected illegal action: {action}')
        return action

    def eval_step(self, state):
        return self.step(state), {}


def build_policy_registry(action_names):
    registry = {'random': RandomPolicy(), 'rule_based': RuleBasedPolicy(action_names)}
    registry.update({name: StylePolicy(name, action_names) for name in STRATEGY_NAMES})
    return registry
