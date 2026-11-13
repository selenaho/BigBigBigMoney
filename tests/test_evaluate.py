import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.evaluate import analyze_run, write_reports


class EvaluationTests(unittest.TestCase):
    def test_belief_carry_forward_and_censored_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'config.json').write_text(json.dumps({
                'learner': 'bayesian', 'seed': 1,
                'bayesian': {'strategies': ['a', 'b'], 'prior': [.5, .5]}}))
            def write(name, fields, rows):
                with (p / name).open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    writer.writerows(rows)
            write('hands.csv', ['hand_id', 'opponent_strategy', 'switched', 'agent_0_payoff'],
                  [[1, 'a', 0, 1], [2, 'a', 0, -2], [3, 'b', 1, 3], [4, 'b', 0, 0]])
            write('beliefs.csv', ['hand_id', 'posterior'],
                  [[1, json.dumps({'a': .9, 'b': .1})],
                   [4, json.dumps({'a': .2, 'b': .8})]])
            run = analyze_run(p, threshold=.5, sustain=2)
            self.assertEqual(run['true_probability'], [.9, .9, .1, .8])
            self.assertEqual(run['summary']['mean_reward'], .5)
            self.assertEqual(run['summary']['strategy_accuracy'], .75)
            self.assertIsNone(run['summary']['observed_noise_rate'])
            self.assertIsNone(run['summary']['mean_decision_ms'])
            self.assertEqual(run['segments'][0]['recovery_hands'], 2)
            self.assertIsNone(run['segments'][1]['recovery_hands'])
            groups = write_reports([run], p / 'report', 2, .5, 2)
            self.assertIsNone(groups[0]['std_across_runs'])
            self.assertTrue((p / 'report' / 'adaptation.csv').exists())


if __name__ == '__main__':
    unittest.main()
