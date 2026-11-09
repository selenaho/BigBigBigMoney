"""Fit action likelihoods on independent seeded simulated games."""

import argparse
import random

import rlcard

from agents.bayesian_model import ActionLikelihoodModel
from agents.rule_based_agent import RuleBasedAgent
from opponents.strategies import OpponentAgent, build_policy_registry
from opponents.styles import STRATEGY_NAMES


def calibrate_likelihoods(action_names=None, num_hands=2000, seed=1729):
    if num_hands < 1:
        raise ValueError('num_hands must be positive')
    env = rlcard.make('leduc-holdem', config={'seed': seed})
    if action_names is not None and tuple(action_names) != tuple(env.actions):
        raise ValueError('Unexpected action mapping')
    model = ActionLikelihoodModel(env.actions)
    registry = build_policy_registry(env.actions)
    rng = random.Random(seed + 1)
    baseline = RuleBasedAgent(env.actions)
    for strategy in STRATEGY_NAMES:
        opponent = OpponentAgent(registry[strategy], seed + 2)
        # Alternate seats and baseline/random probes to cover different contexts.
        for hand in range(num_hands):
            opponent_seat = hand % 2
            state, player = env.reset()
            while not env.is_over():
                if player == opponent_seat:
                    action = opponent.step(state)
                    model.record(strategy, state['raw_obs']['hand'][-1], state, action)
                else:
                    action = baseline.step(state) if hand % 4 < 2 else rng.choice(list(state['legal_actions']))
                state, player = env.step(action)
    model.training = {'hands_per_strategy': num_hands, 'seed': seed,
                      'probe': 'alternating seats; random and rule_based'}
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-hands', type=int, default=2000, help='Hands per strategy')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', default='results/bayesian-likelihoods.json')
    args = parser.parse_args()
    if args.num_hands < 1:
        parser.error('--num-hands must be positive')
    model = calibrate_likelihoods(num_hands=args.num_hands, seed=args.seed)
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f'Wrote likelihood model to {args.output}')


if __name__ == '__main__':
    main()
