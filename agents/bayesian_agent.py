"""Bayesian Leduc agent with hidden-card enumeration and chip utility planning."""

import math
import numpy as np

from agents.bayesian_model import ActionLikelihoodModel, BayesianBelief

RANKS = ('J', 'Q', 'K')


class BayesianAgent:
    """Persistent strategy beliefs, with hand-local evidence for hidden-card ranges.

    Action values use one predicted opponent response followed by a showdown
    rollout (no further betting). A predicted raise is answered optimally by
    folding or calling. This is an approximation, not a full poker game solver.
    Utilities include sunk contributions and use RLCard's big-blind units.
    """
    use_raw = False

    def __init__(self, action_names, likelihood_model=None, prior=None):
        self.action_names = tuple(action_names)
        if set(self.action_names) != {'call', 'raise', 'fold', 'check'}:
            raise ValueError('BayesianAgent requires the four Leduc actions')
        self.likelihood_model = likelihood_model or ActionLikelihoodModel(action_names)
        if self.likelihood_model.action_names != self.action_names:
            raise ValueError('Likelihood model action mapping does not match environment')
        self.belief = BayesianBelief(self.likelihood_model.strategies, prior=prior)
        self.hand = None
        self.hand_observations = []
        self.last_utilities = {}

    @property
    def posterior(self):
        return dict(zip(self.belief.strategies, map(float, self.belief.probabilities)))

    def begin_hand(self, state):
        self.hand = state['raw_obs']['hand'][-1]
        self.hand_observations.clear()

    def _range(self, strategy, public):
        counts = np.array([2.0 - (r == self.hand) - (r == public) for r in RANKS])
        logs = np.full(3, -np.inf)
        for i, rank in enumerate(RANKS):
            if counts[i] > 0:
                logs[i] = math.log(counts[i])
                for state, action in self.hand_observations:
                    logs[i] += math.log(self.likelihood_model.probabilities(strategy, state, rank)[action])
        weights = np.exp(logs - logs.max())
        return weights / weights.sum()

    def observe_opponent_action(self, state, action):
        """Receive a sanitized pre-action public state, even for terminal folds."""
        if self.hand is None:
            raise ValueError('Call begin_hand with the learner observation first')
        if action not in state['legal_actions']:
            raise ValueError('Observed action must be legal in its pre-action state')
        public = state['raw_obs'].get('public_card')
        public = public[-1] if public else None
        name = self.action_names[action]
        likelihoods = []
        for strategy in self.belief.strategies:
            weights = self._range(strategy, public)
            likelihoods.append(sum(w * self.likelihood_model.probabilities(strategy, state, rank)[name]
                                   for rank, w in zip(RANKS, weights)))
        self.belief.update(likelihoods)
        # Copy only public fields; RLCard mutates its action record after step().
        raw = state['raw_obs']
        snapshot = {'raw_obs': {'public_card': raw.get('public_card'),
                                'all_chips': list(raw['all_chips']),
                                'my_chips': raw['my_chips']},
                    'legal_actions': dict(state['legal_actions']),
                    'action_record': list(state.get('action_record', ()))}
        self.hand_observations.append((snapshot, name))

    def _equity(self, opponent_rank, public):
        def score(rank, board):
            return (int(rank == board), RANKS.index(rank))
        def result(board):
            own, other = score(self.hand, board), score(opponent_rank, board)
            return 1.0 if own > other else .5 if own == other else 0.0
        if public:
            return result(public)
        remaining = [2 - (r == self.hand) - (r == opponent_rank) for r in RANKS]
        return sum(n * result(r) for n, r in zip(remaining, RANKS)) / sum(remaining)

    def expected_utilities(self, state):
        raw = state['raw_obs']
        if self.hand is None:
            self.begin_hand(state)
        public = raw.get('public_card')
        public = public[-1] if public else None
        own = float(raw['my_chips'])
        other = float(sum(raw['all_chips']) - own)
        call_cost = max(0., other - own)
        bet = 4.0 if public else 2.0
        history = list(state.get('action_record', ()))
        street_history = state.get('street_action_record', history if not public else ())
        legal_names = [self.action_names[i] for i in state['legal_actions']]
        values = {a: 0.0 for a in legal_names}
        # Posterior over (style, hidden rank); response probabilities are
        # conditioned on each rank, preserving response/strength correlation.
        worlds = [(s, r, float(p * w), self._equity(r, public))
                  for s, p in self.posterior.items()
                  for r, w in zip(RANKS, self._range(s, public)) if w > 0]
        for action in legal_names:
            if action == 'fold':
                values[action] = -own / 2
                continue
            paid = call_cost + bet if action == 'raise' else call_cost if action == 'call' else 0.
            new_own = own + paid
            # A call closes the showdown rollout; checks and raises admit one
            # opponent response. Respect the current street's two-raise cap.
            closes_check = (action == 'check' and street_history
                            and street_history[-1][1] in ('check', 'call'))
            if action == 'call' or closes_check:
                values[action] = sum(w * (eq * (new_own + other) - new_own) / 2
                                     for _, _, w, eq in worlds)
                continue
            raises = sum(a == 'raise' for _, a in street_history) + (action == 'raise')
            response_names = ['call', 'fold'] if action == 'raise' else ['fold', 'check']
            if raises < 2:
                response_names.append('raise')
            response_state = {
                'raw_obs': {'public_card': raw.get('public_card'),
                            'all_chips': [new_own, other], 'my_chips': other},
                'legal_actions': {self.action_names.index(a): None for a in response_names},
                'action_record': history + [(0, action)],
            }
            # If the opponent raises, choose fold/call using the posterior
            # conditional on that raise, rather than looking at its hidden card.
            raised_mass = raised_value = 0.0
            value = 0.0
            for strategy, rank, weight, equity in worlds:
                responses = self.likelihood_model.probabilities(strategy, response_state, rank)
                for response, probability in responses.items():
                    mass = weight * probability
                    if response == 'fold':
                        value += mass * other / 2
                    elif response == 'raise':
                        raised_mass += mass
                        # Opponent matches our commitment then raises one bet.
                        raised_value += mass * (equity * (2 * (new_own + bet)) - (new_own + bet)) / 2
                    else:
                        matched = new_own if response == 'call' else other
                        value += mass * (equity * (new_own + matched) - new_own) / 2
            values[action] = value + max(-new_own / 2 * raised_mass, raised_value)
        return values

    def step(self, state):
        if not state['legal_actions']:
            raise ValueError('No legal actions available')
        self.last_utilities = self.expected_utilities(state)
        best = max(self.last_utilities, key=self.last_utilities.get)
        return self.action_names.index(best)

    def eval_step(self, state):
        action = self.step(state)
        return action, {'probs': {self.action_names[i]: float(i == action) for i in state['legal_actions']},
                        'posterior': self.posterior, 'expected_utilities': dict(self.last_utilities)}
