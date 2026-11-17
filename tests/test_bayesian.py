import csv
import json
import math
import tempfile
import unittest

import numpy as np
import rlcard

from agents.bayesian_agent import BayesianAgent
from agents.bayesian_model import ActionLikelihoodModel, BayesianBelief
from experiments.calibrate_bayesian import calibrate_likelihoods
from experiments.evaluate import analyze_run
from main import run_simulation
from opponents.strategies import build_policy_registry
from opponents.styles import STRATEGY_NAMES


class BayesianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions = tuple(rlcard.make('leduc-holdem').actions)
        cls.model = calibrate_likelihoods(cls.actions, 40, 73)

    def state(self, hand='SJ', public=None, legal=('call', 'raise', 'fold'), chips=(1, 2)):
        return {'raw_obs': {'hand': hand, 'public_card': public, 'all_chips': list(chips),
                            'my_chips': chips[0]},
                'legal_actions': {self.actions.index(a): None for a in legal},
                'action_record': []}

    def test_standard_likelihood_formula(self):
        observations = [np.array([.2, .8]), np.array([.9, .1]), np.array([.6, .4])]
        model = BayesianBelief(('a', 'b'), prior=[.3, .7])
        for values in observations:
            model.update(values)
        expected = np.array([.3, .7]) * np.prod(observations, axis=0)
        np.testing.assert_allclose(model.probabilities, expected / expected.sum())
        for _ in range(10000):
            model.update([.001, .999])
        self.assertTrue(np.isfinite(model.probabilities).all())
        self.assertAlmostEqual(model.probabilities.sum(), 1)

    def test_all_history_persists_after_switch(self):
        model = BayesianBelief(('a', 'b'))
        for _ in range(100):
            model.update([.9, .1])
        for _ in range(15):
            model.update([.1, .9])
        self.assertGreater(model.probabilities[0], .99)
        # Enough contrary evidence eventually outweighs the earlier observations.
        for _ in range(100):
            model.update([.1, .9])
        self.assertGreater(model.probabilities[1], .99)
        self.assertEqual(model.observations, 215)

    def test_window_matches_recent_likelihoods_and_fixed_prior(self):
        prior = np.array([.3, .7])
        observations = [np.array([.2, .8]), np.array([.9, .1]),
                        np.array([.6, .4]), np.array([.05, .95])]
        for size in (1, 2, 10):
            model = BayesianBelief(('a', 'b'), prior=prior, window_size=size)
            np.testing.assert_allclose(model.probabilities, prior)
            for i, likelihood in enumerate(observations):
                model.update(likelihood)
                expected = prior * np.prod(observations[max(0, i + 1 - size):i + 1], axis=0)
                np.testing.assert_allclose(model.probabilities, expected / expected.sum())
                self.assertEqual(len(model.history), min(i + 1, size))
                self.assertEqual(model.observations, i + 1)

    def test_window_forgets_earlier_style_after_switch(self):
        standard = BayesianBelief(('a', 'b'))
        windowed = BayesianBelief(('a', 'b'), window_size=10)
        for model in (standard, windowed):
            for _ in range(100):
                model.update([.9, .1])
            for _ in range(10):
                model.update([.1, .9])
        self.assertGreater(standard.probabilities[0], .99)
        self.assertGreater(windowed.probabilities[1], .99)

    def test_window_is_stable_and_invalid_updates_do_not_evict(self):
        model = BayesianBelief(('a', 'b'), window_size=3)
        for _ in range(10000):
            model.update([1e-200, .9])
        for _ in range(3):
            model.update([.8, .2])
        expected = np.array([.8, .2]) ** 3
        np.testing.assert_allclose(model.probabilities, expected / expected.sum())
        before = model.probabilities.copy()
        count = model.observations
        with self.assertRaises(ValueError):
            model.update([0, 1])
        np.testing.assert_array_equal(model.probabilities, before)
        self.assertEqual(model.observations, count)
        self.assertEqual(len(model.history), 3)

    def test_window_spans_hand_boundaries(self):
        agent = BayesianAgent(self.actions, likelihood_model=self.model, window_size=1)
        agent.begin_hand(self.state())
        agent.observe_opponent_action(self.state(), self.actions.index('raise'))
        posterior = agent.posterior
        agent.begin_hand(self.state(hand='SK'))
        self.assertEqual(agent.posterior, posterior)
        self.assertEqual(agent.hand_observations, [])
        agent.observe_opponent_action(self.state(hand='SK'), self.actions.index('fold'))
        self.assertEqual(agent.belief.observations, 2)
        self.assertEqual(len(agent.belief.history), 1)

    def test_validation(self):
        for kwargs in ({'strategies': ()}, {'strategies': ('a', 'a')},
                       {'prior': [0] * 6}, {'prior': [float('nan')] * 6}, {'prior': [.5]},
                       {'window_size': 0}, {'window_size': -1}, {'window_size': 1.5},
                       {'window_size': True}):
            with self.assertRaises(ValueError):
                BayesianBelief(**kwargs)
        belief = BayesianBelief(('a', 'b'))
        for invalid in ([0, 1], [float('nan'), .5], [-1, .1], [.1], [2, .5]):
            with self.assertRaises(ValueError):
                belief.update(invalid)

    def test_calibration_and_serialization(self):
        repeated = calibrate_likelihoods(self.actions, 40, 73)
        self.assertEqual(set(self.model.counts), set(repeated.counts))
        for key in self.model.counts:
            np.testing.assert_array_equal(self.model.counts[key], repeated.counts[key])
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/likelihoods.json'
            self.model.save(path)
            restored = ActionLikelihoodModel.load(path)
            for strategy in STRATEGY_NAMES:
                for rank in ('J', 'Q', 'K'):
                    for public in (None, 'HK'):
                        state = self.state(public=public)
                        p = restored.probabilities(strategy, state, rank)
                        self.assertEqual(p, self.model.probabilities(strategy, state, rank))
                        self.assertAlmostEqual(sum(p.values()), 1)
                        self.assertTrue(all(0 < x < 1 for x in p.values()))

    def test_style_distributions(self):
        policies = build_policy_registry(self.actions)
        for strategy in STRATEGY_NAMES:
            for hand in ('SJ', 'SQ', 'SK'):
                for public in (None, 'HJ', 'HQ', 'HK'):
                    for legal in (('call', 'raise', 'fold'), ('raise', 'fold', 'check'), ('call', 'fold'), ('fold', 'check')):
                        state = self.state(hand, public, legal)
                        p = policies[strategy].action_probabilities(state)
                        self.assertEqual(set(p), set(legal))
                        self.assertAlmostEqual(sum(p.values()), 1)
                        self.assertTrue(all(0 <= x <= 1 for x in p.values()))

    def test_card_removal_equity_and_chip_units(self):
        agent = BayesianAgent(self.actions, likelihood_model=self.model)
        state = self.state(hand='SK', public='HK', chips=(2, 4))
        agent.begin_hand(state)
        for strategy in STRATEGY_NAMES:
            self.assertEqual(agent._range(strategy, 'K')[2], 0)
        self.assertEqual(agent._equity('Q', 'K'), 1)
        values = agent.expected_utilities(state)
        self.assertEqual(values['fold'], -1)
        self.assertAlmostEqual(values['call'], 2)
        agent.begin_hand(self.state())
        # J versus K: of four possible board cards, one J wins, all others lose.
        self.assertAlmostEqual(agent._equity('K', None), .25)
        self.assertEqual(agent._equity('J', None), .5)
        with self.assertRaises(ValueError):
            agent.step({'legal_actions': {}})

    def test_decisions_use_strategy_posterior_and_response_legality(self):
        class ResponseModel(ActionLikelihoodModel):
            def probabilities(self, strategy, state, rank=None):
                legal = {self.action_names[i] for i in state['legal_actions']}
                # RLCard permits folding even when checking is free.
                assert 'fold' in legal
                preferred = ('fold' if strategy == 'folder' else 'call') if 'call' in legal else 'check'
                return {a: float(a == preferred) for a in legal}
        model = ResponseModel(self.actions, strategies=('folder', 'caller'))
        state = self.state(public='HK', legal=('raise', 'fold', 'check'), chips=(2, 2))
        folder_agent = BayesianAgent(self.actions, likelihood_model=model, prior=[999, 1])
        caller_agent = BayesianAgent(self.actions, likelihood_model=model, prior=[1, 999])
        self.assertEqual(self.actions[folder_agent.step(state)], 'raise')
        self.assertEqual(self.actions[caller_agent.step(state)], 'check')
        # A second check closes the street; there is no further response to predict.
        state['street_action_record'] = [(1, 'check')]
        values = folder_agent.expected_utilities(state)
        self.assertAlmostEqual(values['check'], -.75)

    def test_terminal_observation_and_no_private_information(self):
        class RecordingAgent(BayesianAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seen = []
            def observe_opponent_action(self, state, action):
                self.seen.append(state)
                self.assert_public(state)
                super().observe_opponent_action(state, action)
            @staticmethod
            def assert_public(state):
                assert 'hand' not in state['raw_obs']
                assert 'obs' not in state
                assert 'strategy' not in state
        agent = RecordingAgent(self.actions, likelihood_model=self.model)
        with tempfile.TemporaryDirectory() as directory:
            out = run_simulation(30, 8, ['tight_passive', 'maniac'], 10, directory, agent=agent)
            with (out / 'actions.csv').open() as f:
                actions = list(csv.DictReader(f))
            opponent = [a for a in actions if a['player_id'] == '1']
            self.assertEqual(agent.belief.observations, len(opponent))
            self.assertTrue(any(a['action'] == 'fold' for a in opponent))
            self.assertTrue(all(math.isfinite(x) for x in agent.last_utilities.values()))

    def test_standard_agent_and_reproducible_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/model.json'
            self.model.save(path)
            rows = []
            for _ in range(2):
                out = run_simulation(20, 42, ['maniac', 'calling_station'], 5, directory,
                                     agent='bayesian', likelihood_path=path)
                config = json.loads((out / 'config.json').read_text())
                self.assertEqual(config['bayesian']['variant'], 'standard')
                with (out / 'beliefs.csv').open() as f:
                    beliefs = list(csv.DictReader(f))
                for b in beliefs:
                    del b['run_id']
                    self.assertAlmostEqual(sum(json.loads(b['posterior']).values()), 1)
                rows.append(beliefs)
                with (out / 'actions.csv').open() as f:
                    for action in csv.DictReader(f):
                        self.assertIn(action['action'], json.loads(action['legal_actions']))
            self.assertEqual(rows[0], rows[1])

    def test_noisy_actions_drive_belief_updates(self):
        class RecordingAgent(BayesianAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.observed = []

            def observe_opponent_action(self, state, action):
                super().observe_opponent_action(state, action)
                self.observed.append(self.action_names[action])

        agent = RecordingAgent(self.actions, likelihood_model=self.model)
        with tempfile.TemporaryDirectory() as directory:
            out = run_simulation(30, 8, ['tight_passive', 'maniac'], 10, directory,
                                 agent=agent, noise_rate=1, noise_seed=29)
            with (out / 'actions.csv').open() as f:
                opponent = [r for r in csv.DictReader(f) if r['player_id'] == '1']
            with (out / 'beliefs.csv').open() as f:
                beliefs = list(csv.DictReader(f))
            self.assertEqual(agent.observed, [r['observed_action'] for r in opponent])
            self.assertEqual(agent.observed, [r['observed_action'] for r in beliefs])
            self.assertTrue(all(r['action'] != r['observed_action'] for r in opponent))
            self.assertEqual(agent.belief.observations, len(opponent))
            for row in beliefs:
                probabilities = json.loads(row['posterior']).values()
                self.assertTrue(all(math.isfinite(p) for p in probabilities))
                self.assertAlmostEqual(sum(probabilities), 1)

    def test_windowed_simulation_noise_logs_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/model.json'
            self.model.save(path)
            for noise_rate in (0, .4):
                repeated = []
                for _ in range(2):
                    out = run_simulation(30, 42, ['tight_passive', 'maniac'], 5, directory,
                                         agent='bayesian_windowed', likelihood_path=path,
                                         bayesian_window_size=3, noise_rate=noise_rate)
                    config = json.loads((out / 'config.json').read_text())
                    self.assertEqual(config['learner'], 'bayesian_windowed')
                    self.assertEqual(config['bayesian']['variant'], 'windowed')
                    self.assertEqual(config['bayesian']['window_size'], 3)
                    with (out / 'actions.csv').open() as f:
                        actions = list(csv.DictReader(f))
                    for row in actions:
                        self.assertIn(row['action'], json.loads(row['legal_actions']))
                    opponent = [row for row in actions if row['player_id'] == '1']
                    with (out / 'beliefs.csv').open() as f:
                        beliefs = list(csv.DictReader(f))
                    self.assertEqual([r['observed_action'] for r in beliefs],
                                     [r['observed_action'] for r in opponent])
                    self.assertEqual([int(r['observations']) for r in beliefs],
                                     list(range(1, len(opponent) + 1)))
                    for row in beliefs:
                        row.pop('run_id')
                        self.assertAlmostEqual(sum(json.loads(row['posterior']).values()), 1)
                    repeated.append(beliefs)
                    report = analyze_run(out)
                    self.assertEqual(report['summary']['learner'], 'bayesian_windowed')
                    self.assertEqual(report['summary']['hands'], 30)
                    self.assertTrue(all(s['belief_available'] for s in report['segments']))
                self.assertEqual(*repeated)


if __name__ == '__main__':
    unittest.main()
