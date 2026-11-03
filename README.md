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