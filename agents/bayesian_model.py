"""Empirical context likelihoods and numerically stable standard Bayes updates."""

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from opponents.styles import STRATEGY_NAMES


def context_key(state, action_names):
    """Only public information, from the acting player's perspective."""
    raw = state['raw_obs']
    public = raw.get('public_card')
    legal = tuple(sorted(action_names[i] for i in state['legal_actions']))
    pot = sum(raw['all_chips'])
    history = state.get('action_record', ())
    return (public[-1] if public else '-', min(int(pot // 4), 4),
            min(sum(a == 'raise' for _, a in history), 3), legal)


class BayesianBelief:
    """Standard Bayes: retain all action evidence across hands and switches."""

    def __init__(self, strategies=STRATEGY_NAMES, prior=None):
        self.strategies = tuple(strategies)
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError('Strategies must be nonempty and unique')
        values = np.ones(len(self.strategies)) if prior is None else np.asarray(prior, dtype=float)
        if values.shape != (len(self.strategies),) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('prior must have one finite positive weight per strategy')
        self.prior = values / values.sum()
        self.log_evidence = np.zeros(len(values))
        self.observations = 0

    @property
    def probabilities(self):
        logp = np.log(self.prior) + self.log_evidence
        weights = np.exp(logp - logp.max())
        return weights / weights.sum()

    def update(self, likelihoods):
        values = np.asarray(likelihoods, dtype=float)
        if values.shape != self.prior.shape or np.any(values <= 0) or np.any(values > 1) or not np.all(np.isfinite(values)):
            raise ValueError('likelihoods must be finite probabilities in (0, 1]')
        evidence = np.log(values)
        self.log_evidence += evidence
        # Remove a shared offset: it cannot affect posterior odds.
        self.log_evidence -= self.log_evidence.max()
        self.observations += 1
        return self.probabilities


class ActionLikelihoodModel:
    """Smoothed action counts with context -> rank/legal -> legal backoff.

    Private ranks are used only during offline calibration and as hypothetical
    hidden hands during inference. No opponent private cards are needed at runtime.
    """
    def __init__(self, action_names, strategies=STRATEGY_NAMES, smoothing=1.0):
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError('smoothing must be finite and positive')
        self.action_names = tuple(action_names)
        self.strategies = tuple(strategies)
        self.smoothing = smoothing
        self.counts = defaultdict(lambda: np.zeros(len(self.action_names)))
        self.training = {}

    @staticmethod
    def _keys(strategy, rank, context):
        return [(strategy, '*', context[-1]), (strategy, rank, context[-1]),
                (strategy, rank, context)]

    def record(self, strategy, rank, state, action):
        if strategy not in self.strategies or action not in state['legal_actions']:
            raise ValueError('Unknown strategy or illegal calibration action')
        for key in self._keys(strategy, rank, context_key(state, self.action_names)):
            self.counts[key][action] += 1

    def probabilities(self, strategy, state, rank=None):
        if strategy not in self.strategies:
            raise ValueError('Unknown strategy')
        legal = list(state['legal_actions'])
        if not legal:
            raise ValueError('No legal actions available')
        context = context_key(state, self.action_names)
        keys = self._keys(strategy, rank, context)
        probs = np.ones(len(legal)) / len(legal)
        for key in keys if rank is not None else keys[:1]:
            counts = self.counts.get(key)
            if counts is not None:
                selected = counts[legal]
                probs = (selected + self.smoothing * probs) / (selected.sum() + self.smoothing)
        return {self.action_names[a]: float(p) for a, p in zip(legal, probs)}

    def save(self, path):
        data = {'version': 1, 'action_names': self.action_names, 'strategies': self.strategies,
                'smoothing': self.smoothing, 'training': self.training,
                'counts': [[key, value.tolist()] for key, value in self.counts.items()]}
        Path(path).write_text(json.dumps(data, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['version'] != 1:
            raise ValueError('Unsupported likelihood model version')
        model = cls(data['action_names'], data['strategies'], data['smoothing'])
        def freeze(value):
            return tuple(map(freeze, value)) if isinstance(value, list) else value
        for key, values in data['counts']:
            values = np.asarray(values, dtype=float)
            if values.shape != (len(model.action_names),) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError('Invalid likelihood counts')
            model.counts[freeze(key)] = values
        model.training = data['training']
        return model
