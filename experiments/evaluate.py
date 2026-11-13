"""Summarize saved simulations and plot reward, noise, and Bayesian adaptation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def analyze_run(path, threshold=0.5, sustain=5):
    """Use end-of-hand beliefs, carrying forward when no opponent action occurs.

    Recovery is measured within each strategy segment, in completed hands since
    the switch. None means the threshold was not sustained before that segment
    ended; it is not a zero-length recovery.
    """
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    hands = read_csv(path / 'hands.csv')
    if not hands:
        raise ValueError(f'No completed hands in {path}')
    actions = read_csv(path / 'actions.csv')
    beliefs = read_csv(path / 'beliefs.csv')
    decisions = read_csv(path / 'decisions.csv')
    posterior_by_hand = {}
    for row in beliefs:
        posterior_by_hand[int(row['hand_id'])] = json.loads(row['posterior'])
    bayes = config.get('bayesian', {})
    posterior = dict(zip(bayes.get('strategies', []), bayes.get('prior', [])))
    rewards, true_prob, accuracy, entropy = [], [], [], []
    for hand in hands:
        posterior = posterior_by_hand.get(int(hand['hand_id']), posterior)
        rewards.append(float(hand['agent_0_payoff']))
        truth = hand['opponent_strategy']
        if truth in posterior:
            true_prob.append(posterior[truth])
            # Tied maxima receive fractional credit rather than favoring key order.
            winners = [s for s, p in posterior.items() if p == max(posterior.values())]
            accuracy.append(1 / len(winners) if truth in winners else 0)
            entropy.append(-sum(p * np.log(p) for p in posterior.values() if p > 0))
        else:
            true_prob.append(None)
    boundaries = [0] + [i for i, h in enumerate(hands) if i > 0 and h['switched'] == '1'] + [len(hands)]
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        streak = 0
        recovery = None
        available = all(p is not None for p in true_prob[start:end])
        if available:
            for offset, p in enumerate(true_prob[start:end], 1):
                streak = streak + 1 if p >= threshold else 0
                if streak >= sustain:
                    recovery = offset
                    break
        segments.append({'start_hand': int(hands[start]['hand_id']),
                         'strategy': hands[start]['opponent_strategy'], 'hands': end - start,
                         'is_switch': start > 0, 'belief_available': available,
                         'recovery_hands': recovery,
                         'mean_reward': float(np.mean(rewards[start:end]))})
    opponent = [a for a in actions if a['player_id'] == '1']
    noise_logged = bool(opponent) and 'noise_applied' in opponent[0]
    corrupted = sum(int(a['noise_applied']) for a in opponent) if noise_logged else None
    seconds = [float(d['seconds']) for d in decisions]
    valid_prob = [p for p in true_prob if p is not None]
    summary = {'run_id': path.name, 'learner': config.get('learner'), 'seed': config.get('seed'),
               'configured_noise_rate': config.get('observation_noise', {}).get('rate', 0),
               'hands': len(hands), 'total_reward': sum(rewards),
               'mean_reward': float(np.mean(rewards)), 'opponent_actions': len(opponent),
               'corrupted_actions': corrupted,
               'observed_noise_rate': corrupted / len(opponent) if noise_logged else None,
               'mean_true_strategy_probability': float(np.mean(valid_prob)) if valid_prob else None,
               'strategy_accuracy': float(np.mean(accuracy)) if accuracy else None,
               'mean_entropy_nats': float(np.mean(entropy)) if entropy else None,
               'mean_decision_ms': 1000 * float(np.mean(seconds)) if seconds else None,
               'p95_decision_ms': 1000 * float(np.percentile(seconds, 95)) if seconds else None}
    return {'summary': summary, 'segments': segments, 'rewards': rewards,
            'true_probability': true_prob, 'hand_ids': [int(h['hand_id']) for h in hands]}


def report_directory(output, runs):
    """Append the source run names to the requested output folder name."""
    run_names = '__'.join(run['summary']['run_id'] for run in runs)
    suffix = f'__{run_names}'
    return output if output.name.endswith(suffix) else output.with_name(output.name + suffix)


def write_reports(runs, output, window, threshold, sustain):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0]['summary']))
        writer.writeheader()
        writer.writerows(r['summary'] for r in runs)
    with (output / 'adaptation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id'] + list(runs[0]['segments'][0]))
        writer.writeheader()
        for run in runs:
            writer.writerows(dict(run_id=run['summary']['run_id'], **s) for s in run['segments'])
    groups = {}
    for run in runs:
        s = run['summary']
        groups.setdefault((s['learner'], s['configured_noise_rate']), []).append(s['mean_reward'])
    grouped = [{'learner': agent, 'noise_rate': rate, 'runs': len(values),
                'mean_reward': float(np.mean(values)),
                'std_across_runs': float(np.std(values, ddof=1)) if len(values) > 1 else None}
               for (agent, rate), values in sorted(groups.items())]
    (output / 'report.json').write_text(json.dumps(
        {'settings': {'rolling_window': window, 'recovery_probability': threshold,
                      'recovery_consecutive_hands': sustain},
         'runs': runs, 'noise_groups': grouped}, indent=2, allow_nan=False) + '\n')
    return grouped


def plot_reports(runs, groups, output, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for run in runs:
        s = run['summary']
        label = f"{s['run_id']} ({s['learner']}, noise={s['configured_noise_rate']:g})"
        rewards = np.array(run['rewards'])
        x = run['hand_ids']
        axes[0, 0].plot(x, np.cumsum(rewards), label=label)
        rolling = [np.mean(rewards[max(0, i - window + 1):i + 1]) for i in range(len(rewards))]
        axes[0, 1].plot(x, rolling, label=label)
        axes[1, 0].plot(x, [np.nan if p is None else p for p in run['true_probability']], label=label)
        for segment in run['segments']:
            if segment['is_switch']:
                for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
                    ax.axvline(segment['start_hand'], color='gray', alpha=.15)
        # Each switch contributes only until its next switch, without crossing segments.
        switched = [seg for seg in run['segments'] if seg['is_switch']]
        if switched:
            curves = []
            for seg in switched:
                start = x.index(seg['start_hand'])
                curves.append(np.cumsum(rewards[start:start + seg['hands']]) / np.arange(1, seg['hands'] + 1))
            mean = [np.mean([c[i] for c in curves if len(c) > i]) for i in range(max(map(len, curves)))]
            axes[1, 1].plot(range(1, len(mean) + 1), mean, label=label)
    for ax, title, ylabel in zip(axes.flat,
            ['Cumulative reward', f'Rolling reward (up to {window} hands)',
             'End-of-hand belief in the true style', 'Mean reward since switch (averaged over switches)'],
            ['Total payoff', 'Payoff per hand', 'Probability', 'Payoff per hand']):
        ax.set(title=title, xlabel='Hand', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 1].set_xlabel('Completed hands since switch')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'trajectories.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in sorted({g['learner'] for g in groups}):
        data = [g for g in groups if g['learner'] == agent]
        ax.errorbar([g['noise_rate'] for g in data], [g['mean_reward'] for g in data],
                    yerr=[g['std_across_runs'] or 0 for g in data], marker='o', capsize=4, label=agent)
    ax.set(xlabel='Configured noise probability', ylabel='Mean payoff per hand',
           title='Reward versus noise (error bars: sample SD across runs)')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'noise_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Run directories to compare')
    parser.add_argument('--output-dir', type=Path,
                        help='Output folder prefix (default: results/metrics/noise-analysis for one run, '
                             'results/metrics/comparison for multiple runs); run names are appended')
    parser.add_argument('--window', type=int, default=25)
    parser.add_argument('--recovery-threshold', type=float, default=.5)
    parser.add_argument('--recovery-sustain', type=int, default=5)
    args = parser.parse_args()
    if args.window < 1 or args.recovery_sustain < 1 or not 0 < args.recovery_threshold <= 1:
        parser.error('Window and sustain must be positive; threshold must be in (0, 1]')
    try:
        runs = [analyze_run(p, args.recovery_threshold, args.recovery_sustain) for p in args.runs]
        prefix = args.output_dir or Path('results/metrics') / (
            'noise-analysis' if len(runs) == 1 else 'comparison')
        output = report_directory(prefix, runs)
        grouped = write_reports(runs, output, args.window,
                                args.recovery_threshold, args.recovery_sustain)
        plot_reports(runs, grouped, output, args.window)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Wrote metrics and plots to {output}')
    print('Compare matched settings and multiple seeds; one run per noise level has no variability estimate.')


if __name__ == '__main__':
    main()
