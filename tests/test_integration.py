import csv
import json
import tempfile
import unittest
from pathlib import Path

import rlcard
from agents.rule_based_agent import RuleBasedAgent
from main import run_simulation
from opponents.strategies import OpponentAgent
from opponents.switching_controller import SwitchingController


class IntegrationTests(unittest.TestCase):
    def test_agent_parameter_selects_player_zero_only(self):
        class RecordingAgent(RuleBasedAgent):
            def __init__(self, actions):
                super().__init__(actions)
                self.rewards = []

            def observe_reward(self, reward):
                self.rewards.append(float(reward))

        instance = RecordingAgent(rlcard.make('leduc-holdem').actions)
        with tempfile.TemporaryDirectory() as directory:
            for selected, label in [('rule_based', 'rule_based'), ('linucb', 'linucb'),
                                    (instance, 'RecordingAgent')]:
                output = run_simulation(num_games=12, agent=selected,
                                        opponent_names=['random', 'rule_based'],
                                        switch_every=3, output_dir=directory)
                with (output / 'hands.csv').open() as f:
                    hands = list(csv.DictReader(f))
                self.assertEqual({h['agent_0'] for h in hands}, {label})
                self.assertEqual([h['agent_1'] for h in hands],
                                 ['random'] * 3 + ['rule_based'] * 3
                                 + ['random'] * 3 + ['rule_based'] * 3)
                self.assertEqual([int(h['hand_id']) for h in hands if h['switched'] == '1'],
                                 [4, 7, 10])
                config = json.loads((output / 'config.json').read_text())
                self.assertEqual(config['learner'], label)
                self.assertEqual(config['learner_seat'], 0)
                if selected is instance:
                    self.assertEqual(instance.rewards, [float(h['agent_0_payoff']) for h in hands])
            for invalid in ['unknown', object()]:
                with self.assertRaises(ValueError):
                    run_simulation(agent=invalid, output_dir=directory)

    def test_hand_rules_and_legal_fallback(self):
        env = rlcard.make('leduc-holdem')
        agent = RuleBasedAgent(env.actions)
        cases = [('SJ', None, ['fold', 'call', 'raise'], 'fold'),
                 ('SQ', None, ['fold', 'call', 'raise'], 'call'),
                 ('SK', None, ['fold', 'call', 'raise'], 'raise'),
                 ('SJ', 'HJ', ['fold', 'call', 'raise'], 'raise'),
                 ('SJ', None, ['check', 'raise'], 'check'),
                 ('SK', None, ['call', 'fold'], 'call')]
        for hand, public, legal, expected in cases:
            state = {'raw_obs': {'hand': hand, 'public_card': public},
                     'legal_actions': {env.actions.index(a): None for a in legal}}
            self.assertEqual(env.actions[agent.step(state)], expected)

    def test_switch_boundaries(self):
        schedule = SwitchingController(['a', 'b'], 25)
        self.assertEqual([schedule.strategy_for_hand(i) for i in [0, 24, 25, 49, 50]],
                         ['a', 'a', 'b', 'b', 'a'])
        with self.assertRaises(ValueError):
            SwitchingController(['a'], 0)

    def test_policy_contract_rejects_illegal_action(self):
        class InvalidPolicy:
            def select_action(self, state, rng):
                return 999
        with self.assertRaises(ValueError):
            OpponentAgent(InvalidPolicy(), 1).step({'legal_actions': {0: None}})

    def test_runs_logs_and_repeatability(self):
        def read(path, name):
            with (path / name).open() as f:
                return [{k: v for k, v in row.items() if k != 'run_id'}
                        for row in csv.DictReader(f)]

        with tempfile.TemporaryDirectory() as directory:
            outputs = [run_simulation(100, 42, ['random', 'rule_based'], 25, directory)
                       for _ in range(2)]
            for name in ['hands.csv', 'actions.csv']:
                self.assertEqual(read(outputs[0], name), read(outputs[1], name))
            hands = read(outputs[0], 'hands.csv')
            actions = read(outputs[0], 'actions.csv')
            self.assertEqual(len(hands), 100)
            self.assertEqual([int(h['hand_id']) for h in hands if h['switched'] == '1'],
                             [26, 51, 76])
            for hand in hands:
                self.assertEqual(float(hand['agent_0_payoff']) + float(hand['agent_1_payoff']), 0)
                decisions = [a for a in actions if a['hand_id'] == hand['hand_id']]
                self.assertEqual(len(decisions), int(hand['num_actions']))
                self.assertEqual([int(a['decision_index']) for a in decisions],
                                 list(range(1, len(decisions) + 1)))
            for action in actions:
                self.assertIn(action['action'], json.loads(action['legal_actions']))
            fixed = run_simulation(20, 9, ['rule_based'], 5, directory)
            self.assertTrue(all(h['switched'] == '0' and h['opponent_strategy'] == 'rule_based'
                                for h in read(fixed, 'hands.csv')))
            with self.assertRaises(ValueError):
                run_simulation(opponent_names=['unknown'], output_dir=directory)


if __name__ == '__main__':
    unittest.main()
