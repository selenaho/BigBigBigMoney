"""Controlled Leduc styles, inspired by tight/loose and aggressive/passive axes.

These stochastic parameters are experimental definitions, not published estimates.
"""

STRATEGY_NAMES = ('tight_aggressive', 'loose_aggressive', 'tight_passive',
                  'loose_passive', 'maniac', 'calling_station')
# Continue with a weak hand, raise with a strong hand, bluff, raise with medium hand.
STYLE_PARAMETERS = {
    'tight_aggressive': (.18, .85, .08, .35),
    'loose_aggressive': (.72, .85, .30, .65),
    'tight_passive': (.18, .20, .02, .08),
    'loose_passive': (.72, .20, .05, .12),
    'maniac': (.95, .95, .80, .90),
    'calling_station': (.98, .05, .01, .03),
}


class StylePolicy:
    def __init__(self, name, action_names):
        self.name = name
        self.action_names = tuple(action_names)
        self.parameters = STYLE_PARAMETERS[name]

    def action_probabilities(self, state):
        raw = state['raw_obs']
        rank = raw['hand'][-1]
        public = raw['public_card']
        paired = public is not None and rank == public[-1]
        strong = paired or (public is None and rank == 'K')
        medium = not strong and (rank == 'K' or (public is None and rank == 'Q'))
        continuation, value_raise, bluff, medium_raise = self.parameters
        aggression = value_raise if strong else medium_raise if medium else bluff
        legal = [self.action_names[i] for i in state['legal_actions']]
        facing = 'call' in legal
        stay = .99 if strong else (continuation + 1) / 2 if medium else continuation
        weights = {name: 0.0 for name in legal}
        if facing:
            weights['fold'] = 1 - stay
            weights['call'] = stay * (1 - aggression if 'raise' in legal else 1)
            if 'raise' in legal:
                weights['raise'] = stay * aggression
        else:
            weights['check'] = 1 - aggression if 'raise' in legal else 1
            if 'raise' in legal:
                weights['raise'] = aggression
        return weights

    def select_action(self, state, rng):
        probabilities = self.action_probabilities(state)
        names = list(probabilities)
        return self.action_names.index(rng.choices(names, [probabilities[a] for a in names])[0])
