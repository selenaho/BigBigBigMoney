import argparse
import copy
import csv
import json
import math
import tempfile
import time
from pathlib import Path
import rlcard
from agents.bayesian_agent import BayesianAgent
from agents.bayesian_model import ActionLikelihoodModel
from experiments.calibrate_bayesian import calibrate_likelihoods
from experiments.noise_layer import ObservationNoise
from agents.rl_agent import LinUCBAgent
from agents.rule_based_agent import RuleBasedAgent
from opponents.strategies import OpponentAgent, build_policy_registry
from opponents.switching_controller import SwitchingController
from opponents.styles import STRATEGY_NAMES


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def build_evaluated_agent(agent, action_names, *, likelihood_model=None):
    """Select player 0's strategy"""
    if isinstance(agent, str):
        name = agent
        if agent == 'rule_based':
            agent = RuleBasedAgent(action_names)
        elif agent == 'linucb':
            agent = LinUCBAgent(action_names)
        elif agent == 'bayesian':
            agent = BayesianAgent(action_names, likelihood_model=likelihood_model)
        else:
            raise ValueError(f'Unknown player-0 agent: {agent}')
    else:
        name = type(agent).__name__
    if not callable(getattr(agent, 'step', None)) or getattr(agent, 'use_raw', False):
        raise ValueError('Player-0 agent must implement step(state) returning legal action IDs')
    return agent, name


def run_simulation(num_games=100, seed=42, opponent_names=('random',),
                   switch_every=25, output_dir='results', policy_registry=None, *, agent='rule_based',
                   likelihood_path=None, calibration_hands=2000, calibration_seed=1729,
                   noise_rate=0.0, noise_seed=None):
    """Evaluate one player-0 agent against switching player-1 policies.

    agent is rule_based, linucb, or bayesian.
    The same player-0 instance persists across all hands and opponent switches.
    An optional observe_reward hook receives player 0's payoff after each hand.
    """
    if num_games < 1:
        raise ValueError('num_games must be positive')
    noise_seed = seed + 2 if noise_seed is None else noise_seed
    noise = ObservationNoise(noise_rate, noise_seed)
    env = rlcard.make('leduc-holdem', config={'seed': seed})
    registry = build_policy_registry(env.actions) if policy_registry is None else policy_registry
    opponent_names = tuple(opponent_names)
    random_switching = opponent_names == ('random',)
    if 'random' in opponent_names and not random_switching:
        raise ValueError('Use random alone for random style switching; use random_actions in an ordered schedule')
    strategy_names = STRATEGY_NAMES if random_switching else opponent_names
    schedule = SwitchingController(strategy_names, switch_every,
                                   randomize=random_switching, seed=seed + 3)
    unknown = set(strategy_names) - registry.keys()
    if unknown:
        raise ValueError(f'Unknown opponent policies: {sorted(unknown)}')
    model = None
    if isinstance(agent, str) and agent == 'bayesian':
        model = (ActionLikelihoodModel.load(likelihood_path) if likelihood_path else
                 calibrate_likelihoods(env.actions, calibration_hands, calibration_seed))
    learner, agent_name = build_evaluated_agent(
        agent, env.actions, likelihood_model=model)
    opponent = OpponentAgent(registry[schedule.strategy_for_hand(0)], seed=seed + 1)
    agents = [learner, opponent]
    env.set_agents(agents)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=root))
    run_id = run_dir.name
    config = {'game': 'leduc-holdem', 'num_games': num_games, 'seed': seed,
              'opponent_seed': seed + 1, 'opponents': list(opponent_names),
              'opponent_schedule': {'mode': 'random' if random_switching else 'ordered',
                                    'seed': seed + 3 if random_switching else None,
                                    'strategies': list(strategy_names)},
              'switch_every': switch_every, 'learner': agent_name, 'learner_seat': 0,
              'action_names': list(env.actions),
              'observation_noise': {'rate': noise.rate, 'seed': noise_seed,
                                    'model': 'uniform_other_legal_action',
                                    'target': 'learner_observations_of_opponent'}}
    if isinstance(learner, LinUCBAgent):
        config['linucb_alpha'] = learner.alpha
    if isinstance(learner, BayesianAgent):
        config['bayesian'] = {'variant': 'standard',
                              'strategies': list(learner.belief.strategies),
                              'prior': learner.belief.prior.tolist(),
                              'calibration': learner.likelihood_model.training,
                              'planner': 'one response then showdown; optimal fold/call to raise'}
        learner.likelihood_model.save(run_dir / 'likelihoods.json')
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    hand_fields = ['run_id', 'seed', 'hand_id', 'agent_0', 'agent_1',
                   'opponent_strategy', 'switched', 'num_actions',
                   'agent_0_payoff', 'agent_1_payoff']
    action_fields = ['run_id', 'hand_id', 'decision_index', 'player_id',
                     'public_card', 'pot', 'player_chips', 'legal_actions',
                     'action_id', 'action', 'observed_action_id', 'observed_action',
                     'noise_applied']
    belief_fields = ['run_id', 'hand_id', 'decision_index', 'observed_action',
                     'observations', 'posterior', 'entropy']
    total = 0.0
    previous = None
    with (run_dir / 'hands.csv').open('w', newline='') as hf, \
            (run_dir / 'actions.csv').open('w', newline='') as af, \
            (run_dir / 'beliefs.csv').open('w', newline='') as bf, \
            (run_dir / 'decisions.csv').open('w', newline='') as df:
        hands = csv.DictWriter(hf, fieldnames=hand_fields)
        actions = csv.DictWriter(af, fieldnames=action_fields)
        beliefs = csv.DictWriter(bf, fieldnames=belief_fields)
        decisions = csv.DictWriter(df, fieldnames=['run_id', 'hand_id', 'decision_index',
                                                   'seconds', 'expected_utilities', 'posterior'])
        beliefs.writeheader()
        decisions.writeheader()
        hands.writeheader()
        actions.writeheader()
        for index in range(num_games):
            strategy = schedule.strategy_for_hand(index)
            opponent.policy = registry[strategy]
            state, player_id = env.reset()
            begin_hook = getattr(learner, 'begin_hand', None)
            if callable(begin_hook):
                begin_hook(env.get_state(0))
            street_history = []
            observed_history = []
            observed_street_history = []
            previous_public = state['raw_obs']['public_card']
            decision = 0
            while not env.is_over():
                public = state['raw_obs']['public_card']
                if public != previous_public:
                    street_history = []
                    observed_street_history = []
                    previous_public = public
                state['street_action_record'] = list(street_history)
                decision_state = state
                if player_id == 0:
                    decision_state = copy.deepcopy(state)
                    decision_state['action_record'] = list(observed_history)
                    decision_state['street_action_record'] = list(observed_street_history)
                started = time.perf_counter()
                action = agents[player_id].step(decision_state)
                elapsed = time.perf_counter() - started
                if action not in state['legal_actions']:
                    raise ValueError(f'Illegal action from player {player_id}')
                decision += 1
                raw = state['raw_obs']
                observed_action = (noise.observe_action(action, state['legal_actions'])
                                   if player_id == 1 else action)
                actions.writerow({
                    'run_id': run_id, 'hand_id': index + 1,
                    'decision_index': decision, 'player_id': player_id,
                    'public_card': raw['public_card'] or '',
                    'pot': sum(raw['all_chips']), 'player_chips': raw['my_chips'],
                    'legal_actions': json.dumps([env.actions[i] for i in state['legal_actions']]),
                    'action_id': action, 'action': env.actions[action],
                    'observed_action_id': observed_action,
                    'observed_action': env.actions[observed_action],
                    'noise_applied': int(observed_action != action),
                })
                if player_id == 0 and isinstance(learner, BayesianAgent):
                    decisions.writerow({'run_id': run_id, 'hand_id': index + 1,
                                        'decision_index': decision, 'seconds': elapsed,
                                        'expected_utilities': json.dumps(learner.last_utilities),
                                        'posterior': json.dumps(learner.posterior)})
                if player_id == 1:
                    hook = getattr(learner, 'observe_opponent_action', None)
                    if callable(hook):
                        # Build an allowlisted public observation. Neither raw_obs.hand
                        # nor the private-card encoded obs vector leaves this boundary.
                        observation = {'raw_obs': {'public_card': raw['public_card'],
                                                  'all_chips': list(raw['all_chips']),
                                                  'my_chips': raw['my_chips']},
                                       'legal_actions': dict(state['legal_actions']),
                                       'action_record': list(observed_history),
                                       'street_action_record': list(observed_street_history)}
                        hook(observation, observed_action)
                    if isinstance(learner, BayesianAgent):
                        beliefs.writerow({'run_id': run_id, 'hand_id': index + 1,
                                          'decision_index': decision, 'observed_action': env.actions[observed_action],
                                          'observations': learner.belief.observations,
                                          'posterior': json.dumps(learner.posterior),
                                          'entropy': -sum(p * math.log(p) for p in learner.posterior.values() if p > 0)})
                street_history.append((player_id, env.actions[action]))
                observed_history.append((player_id, env.actions[observed_action]))
                observed_street_history.append((player_id, env.actions[observed_action]))
                state, player_id = env.step(action)
            payoffs = env.get_payoffs()
            reward_hook = getattr(learner, 'observe_reward', None)
            if callable(reward_hook):
                reward_hook(payoffs[0])
            total += float(payoffs[0])
            hands.writerow({
                'run_id': run_id, 'seed': seed, 'hand_id': index + 1,
                'agent_0': agent_name, 'agent_1': strategy,
                'opponent_strategy': strategy,
                'switched': int(previous is not None and previous != strategy),
                'num_actions': decision, 'agent_0_payoff': float(payoffs[0]),
                'agent_1_payoff': float(payoffs[1]),
            })
            previous = strategy
    print(f'Completed {num_games} hands; {agent_name} average payoff: {total / num_games:.3f}')
    print(f'Logs saved to {run_dir}')
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-games', type=positive_int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--agent', choices=['rule_based', 'linucb', 'bayesian'], default='rule_based',
                        help='Player-0 agent')
    parser.add_argument('--opponent', nargs='+', default=['random'],
                        help='random switches among six styles; otherwise ordered policy names (including random_actions)')
    parser.add_argument('--switch-every', type=positive_int, default=25,
                        help='Number of completed hands between opponent strategy switches')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--likelihood-model', type=Path)
    parser.add_argument('--calibration-hands', type=positive_int, default=2000)
    parser.add_argument('--calibration-seed', type=int, default=1729)
    parser.add_argument('--noise-rate', type=float, default=0.0,
                        help='Probability of corrupting each observed opponent action (0 to 1)')
    parser.add_argument('--noise-seed', type=int,
                        help='Independent observation-noise seed (default: seed + 2)')
    args = parser.parse_args()
    try:
        run_simulation(args.num_games, args.seed, args.opponent,
                       args.switch_every, args.output_dir, agent=args.agent,
                       likelihood_path=args.likelihood_model,
                       calibration_hands=args.calibration_hands, calibration_seed=args.calibration_seed,
                       noise_rate=args.noise_rate, noise_seed=args.noise_seed)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
