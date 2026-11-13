import csv
import json
import tempfile
import unittest

import rlcard

from agents.rule_based_agent import RuleBasedAgent
from experiments.noise_layer import ObservationNoise
from main import run_simulation


class NoiseTests(unittest.TestCase):
    def test_validation_endpoints_and_repeatability(self):
        for rate in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ObservationNoise(rate)
        for rate in (0, .4, 1):
            layers = [ObservationNoise(rate, 17) for _ in range(2)]
            sequences = [[layer.observe_action(2, [2, 5, 8]) for _ in range(1000)]
                         for layer in layers]
            self.assertEqual(*sequences)
            self.assertTrue(set(sequences[0]) <= {2, 5, 8})
            if rate == 0:
                self.assertEqual(set(sequences[0]), {2})
            elif rate == 1:
                self.assertEqual(set(sequences[0]), {5, 8})
            else:
                self.assertTrue(350 < sum(a != 2 for a in sequences[0]) < 450)
        self.assertEqual(ObservationNoise(1).observe_action(5, [5]), 5)
        with self.assertRaises(ValueError):
            ObservationNoise().observe_action(2, [5])

    def test_noise_only_changes_observations_and_is_consistent_across_interfaces(self):
        action_names = rlcard.make('leduc-holdem').actions

        class RecordingAgent(RuleBasedAgent):
            def __init__(self):
                super().__init__(action_names)
                self.events = []

            def begin_hand(self, state):
                self.events.append(('begin', state, None))

            def step(self, state):
                action = super().step(state)
                self.events.append(('step', state, action))
                return action

            def observe_opponent_action(self, state, action):
                self.events.append(('observe', state, action))

        def rows(output, filename):
            with (output / filename).open() as f:
                return [{k: v for k, v in row.items() if k != 'run_id'}
                        for row in csv.DictReader(f)]

        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            agents = []
            for rate in (0, 1, 1):
                agent = RecordingAgent()
                agents.append(agent)
                outputs.append(run_simulation(60, 12, ['maniac', 'tight_passive'], 10,
                                              directory, agent=agent, noise_rate=rate,
                                              noise_seed=91))
            clean, noisy, repeated = [rows(out, 'actions.csv') for out in outputs]
            self.assertEqual(noisy, repeated)
            self.assertEqual(rows(outputs[0], 'hands.csv'), rows(outputs[1], 'hands.csv'))
            metadata = {'observed_action_id', 'observed_action', 'noise_applied'}
            self.assertEqual([{k: v for k, v in r.items() if k not in metadata} for r in clean],
                             [{k: v for k, v in r.items() if k not in metadata} for r in noisy])
            for row in noisy:
                self.assertIn(row['observed_action'], json.loads(row['legal_actions']))
                self.assertEqual(row['noise_applied'], row['player_id'])
            observed = [e[2] for e in agents[1].events if e[0] == 'observe']
            self.assertEqual(observed, [int(r['observed_action_id']) for r in noisy
                                        if r['player_id'] == '1'])
            history = []
            street = []
            previous_public = None
            for kind, state, action in agents[1].events:
                if kind == 'begin':
                    history, street = [], []
                    previous_public = state['raw_obs']['public_card']
                    continue
                public = state['raw_obs']['public_card']
                if public != previous_public:
                    street = []
                    previous_public = public
                self.assertEqual(state['action_record'], history)
                self.assertEqual(state['street_action_record'], street)
                if kind == 'observe':
                    self.assertNotIn('hand', state['raw_obs'])
                    self.assertNotIn('obs', state)
                entry = (int(kind == 'observe'), action_names[action])
                history.append(entry)
                street.append(entry)
            config = json.loads((outputs[1] / 'config.json').read_text())
            self.assertEqual(config['observation_noise']['rate'], 1)
            self.assertEqual(config['observation_noise']['seed'], 91)


if __name__ == '__main__':
    unittest.main()
