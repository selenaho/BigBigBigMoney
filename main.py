import argparse
import csv
import json
import tempfile
from pathlib import Path
import rlcard
# from agents.bayesian_agent import BayesianAgent
from agents.rl_agent import LinUCBAgent
from agents.rule_based_agent import RuleBasedAgent
from opponents.strategies import OpponentAgent, build_policy_registry
from opponents.switching_controller import SwitchingController


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def build_evaluated_agent(agent, action_names):
    """Select player 0's strategy"""
    if isinstance(agent, str):
        name = agent
        if agent == 'rule_based':
            agent = RuleBasedAgent(action_names)
        elif agent == 'linucb':
            agent = LinUCBAgent(action_names)
        elif agent == 'bayesian':
            agent = BayesianAgent(action_names)
        else:
            raise ValueError(f'Unknown player-0 agent: {agent}')
    else:
        name = type(agent).__name__
    if not callable(getattr(agent, 'step', None)) or getattr(agent, 'use_raw', False):
        raise ValueError('Player-0 agent must implement step(state) returning legal action IDs')
    return agent, name


def run_simulation(num_games=100, seed=42, opponent_names=('random',),
                   switch_every=25, output_dir='results', policy_registry=None, *, agent='rule-based'):
    """Evaluate one player-0 agent against switching player-1 policies.

    agent is rule_based, linucb, or bayesian.
    The same player-0 instance persists across all hands and opponent switches.
    An optional observe_reward hook receives player 0's payoff after each hand.
    """
    if num_games < 1:
        raise ValueError('num_games must be positive')
    env = rlcard.make('leduc-holdem', config={'seed': seed})
    registry = build_policy_registry(env.actions) if policy_registry is None else policy_registry
    schedule = SwitchingController(opponent_names, switch_every)
    unknown = set(opponent_names) - registry.keys()
    if unknown:
        raise ValueError(f'Unknown opponent policies: {sorted(unknown)}')
    learner, agent_name = build_evaluated_agent(agent, env.actions)
    opponent = OpponentAgent(registry[opponent_names[0]], seed=seed + 1)
    agents = [learner, opponent]
    env.set_agents(agents)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=root))
    run_id = run_dir.name
    config = {'game': 'leduc-holdem', 'num_games': num_games, 'seed': seed,
              'opponent_seed': seed + 1, 'opponents': list(opponent_names),
              'switch_every': switch_every, 'learner': agent_name, 'learner_seat': 0,
              'action_names': list(env.actions)}
    if isinstance(learner, LinUCBAgent):
        config['linucb_alpha'] = learner.alpha
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    hand_fields = ['run_id', 'seed', 'hand_id', 'agent_0', 'agent_1',
                   'opponent_strategy', 'switched', 'num_actions',
                   'agent_0_payoff', 'agent_1_payoff']
    action_fields = ['run_id', 'hand_id', 'decision_index', 'player_id',
                     'public_card', 'pot', 'player_chips', 'legal_actions',
                     'action_id', 'action']
    total = 0.0
    previous = None
    with (run_dir / 'hands.csv').open('w', newline='') as hf, \
            (run_dir / 'actions.csv').open('w', newline='') as af:
        hands = csv.DictWriter(hf, fieldnames=hand_fields)
        actions = csv.DictWriter(af, fieldnames=action_fields)
        hands.writeheader()
        actions.writeheader()
        for index in range(num_games):
            strategy = schedule.strategy_for_hand(index)
            opponent.policy = registry[strategy]
            state, player_id = env.reset()
            decision = 0
            while not env.is_over():
                action = agents[player_id].step(state)
                if action not in state['legal_actions']:
                    raise ValueError(f'Illegal action from player {player_id}')
                decision += 1
                raw = state['raw_obs']
                actions.writerow({
                    'run_id': run_id, 'hand_id': index + 1,
                    'decision_index': decision, 'player_id': player_id,
                    'public_card': raw['public_card'] or '',
                    'pot': sum(raw['all_chips']), 'player_chips': raw['my_chips'],
                    'legal_actions': json.dumps([env.actions[i] for i in state['legal_actions']]),
                    'action_id': action, 'action': env.actions[action],
                })
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
    parser.add_argument('--agent', choices=['rule_based', 'linucb', 'bayesian'], default='rule-based',
                        help='Player-0 agent')
    parser.add_argument('--opponent', nargs='+', default=['random'],
                        help='Ordered policy names; multiple names enable switching')
    parser.add_argument('--switch-every', type=positive_int, default=25)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    try:
        run_simulation(args.num_games, args.seed, args.opponent,
                       args.switch_every, args.output_dir, agent=args.agent)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
