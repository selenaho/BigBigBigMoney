"""Fixed hand-strength baseline; no learning or opponent modeling."""

def choose_legal_action(preferences, legal_action_ids, action_names):
    legal = {action_names[i]: i for i in legal_action_ids}
    for name in preferences:
        if name in legal:
            return legal[name]
    raise ValueError('No supported legal action available')


class RuleBasedAgent:
    use_raw = False

    def __init__(self, action_names):
        self.action_names = tuple(action_names)

    def step(self, state):
        raw = state['raw_obs']
        rank = raw['hand'][-1]
        public = raw['public_card']
        paired = public is not None and public[-1] == rank
        if paired or rank == 'K':
            preferences = ('raise', 'check', 'call', 'fold')
        elif rank == 'Q':
            preferences = ('check', 'call', 'fold', 'raise')
        else:
            preferences = ('check', 'fold', 'call', 'raise')
        return choose_legal_action(preferences, state['legal_actions'], self.action_names)

    def eval_step(self, state):
        action = self.step(state)
        return action, {'probs': {self.action_names[i]: float(i == action)
                                  for i in state['legal_actions']}}
