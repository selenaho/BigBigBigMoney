# BigBigBigMoney

RLCard Leduc Hold'em, two players. Player 0 is the evaluated agent (rule-based, LinUCB, or Bayesian). Player 1 uses the six controlled opponent policies and switches between them. 

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

The default number of games is 100 and the default seed is 42:

```sh
python main.py --agent rule_based --opponent random --switch-every 25 --num-games 100 --seed 42
python main.py --agent linucb --opponent random --switch-every 25 --num-games 100 --seed 42
python main.py --agent bayesian --opponent random --switch-every 25 --num-games 100 --seed 42
```

Each run creates a new `results/run-*` directory.
`config.json` records the seeds, opponent schedule, action mapping, and game count.
`hands.csv` records strategy labels, changes, action counts, and final payoffs.
`actions.csv` records pre-action public cards, total committed chips (`pot`),
acting-player committed chips (`player_chips`), legal actions, and the chosen action.
Payoffs use RLCard's native units. Hand IDs and decision indices start at 1.
A switch occurs between hands: interval 25 gives changes at hands 26, 51, 76.
The first hand is not marked as a switch. No private cards are written to logs.

`--opponent random` selects an initial style from the six controlled styles and
randomly selects a different style every `--switch-every` completed hands. With
25, switches occur at hands 26, 51, and 76. This interval counts hands, not
individual betting actions. The schedule uses its own RNG seeded with `seed + 3`,
so the same seed reproduces the style schedule independently of observation noise.
Actual style labels appear in `hands.csv`, and `config.json` records the schedule
mode, candidate styles, and seed.

Use `--opponent random_actions` for the original policy that picks a random legal
action on every decision. Explicit lists such as `--opponent tight_aggressive
maniac` still cycle in order. The `random` switching mode must be used alone.

#### Unit test to verify if poker enviroment is working

```sh
python -m unittest discover -s tests -v
```

## Baseline rules

Pair or king: raise, then check, call, fold in legal preference order.
Unpaired queen: check, call, fold, raise.
Unpaired jack: check, fold, call, raise.

The baseline uses its own card and the public card, with no opponent modeling or 
learning.

## Standard Bayesian agent

Starts with equal probabilities for six opponent styles: `tight_aggressive`,
`loose_aggressive`, `tight_passive`, `loose_passive`, `maniac`, and `calling_station`.
Updates beliefs after each opponent action and retains all historical evidence.
Opponent private cards and true strategy labels are hidden from the agent.

```sh
python main.py --agent bayesian --opponent tight_aggressive maniac --switch-every 100 --num-games 1000
```

The first step automatically trains action likelihoods using 2,000 simulated hands
per style. Adjust this with `--calibration-hands`, or reuse a saved model with
`--likelihood-model results/run-XXXX/likelihoods.json`.

Actions maximize estimated chip payoff using possible opponent hands and predicted
responses. Planning considers one opponent response followed by a showdown estimate.

Bayesian runs additionally save `likelihoods.json` (trained model), `beliefs.csv`
(strategy probabilities), and `decisions.csv` (action values and timing).

## Observation noise

The proposal's observation-noise experiment is implemented as action-label
substitution. At each opponent decision, `--noise-rate` is the probability of
reporting a uniformly sampled *different* legal action to the learner. A decision
with only one legal action is left unchanged. The default rate is zero.

```sh
python main.py --agent bayesian --opponent random --switch-every 25 --num-games 100 --seed 42 --noise-rate 0.2 --noise-seed 44
```

One noisy observation is shared by the learner's opponent-action callback and
its full-hand and current-street action histories. The opponent and game environment
use actual actions; cards, pot sizes, legal moves, rewards, and strategy labels
are not corrupted. Calibration continues to use clean simulated actions.
Rule-based and current LinUCB agents do not consume opponent action histories,
so their behavior is unaffected by this noise model.

Noise uses an independent random generator (`--noise-seed`, default `seed + 2`).
`config.json` records its rate, seed, and model. In `actions.csv`, `action` and
`action_id` remain ground truth; `observed_action`, `observed_action_id`, and
`noise_applied` record what the learner saw. Learner actions are never corrupted.
`beliefs.csv` records the observed action used for each Bayesian update.

## Metrics and plots

Evaluate one or more saved runs (install `requirements.txt` for Matplotlib):

```sh
python -m experiments.evaluate results/run-al_ypooz --output-dir results/noise-analysis
python -m experiments.evaluate results/run-FIRST results/run-SECOND --output-dir results/comparison --window 25
```

The evaluator writes:

- `summary.csv`: total/mean reward, actual corruption rate, end-of-hand strategy
  accuracy, probability of the true strategy, entropy, and mean/p95 decision time.
- `adaptation.csv`: each strategy segment and its recovery time. By default recovery
  requires five consecutive end-of-hand beliefs assigning at least 50% to the true
  strategy (`--recovery-threshold`, `--recovery-sustain`). Time counts completed
  hands through confirmation. Blank recovery means no recovery before the next
  switch/end, or no applicable beliefs; check `belief_available`. The initial
  segment is acquisition, not post-switch adaptation (`is_switch=false`).
- `trajectories.png`: cumulative reward, rolling reward, probability of the true
  strategy, and reward since switching, averaged over switches within each run.
- `noise_comparison.png`: mean reward by agent/noise rate, with sample standard
  deviation across runs when multiple runs are supplied.
- `report.json`: settings, summaries, per-hand series, segments, and grouped results.

Beliefs carry forward on hands without opponent observations. Strategy metrics
are unavailable for opponents outside the Bayesian model (such as `random_actions`).
Decision timing currently exists only for Bayesian agents. Entropy measures
confidence, not correctness. A low entropy can accompany the wrong strategy.

For noise experiments, run rates such as 0, 0.1, 0.2, and 0.4 with the same set of
several seeds, hand counts, opponent schedules, and calibration model. Pass those
run folders to the evaluator. Grouping uses agent and noise rate only, so supply
comparable runs and avoid duplicate seeds. Error bars describe variability across
runs, not confidence intervals; one run cannot establish an average noise effect.
Changed learner decisions can change later game trajectories even with matched seeds.