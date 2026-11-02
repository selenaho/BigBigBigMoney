# BigBigBigMoney

RLCard Leduc Hold'em, two players. Player 0 only uses the fixed rule-based baseline agent. Player 1 uses a registered opponent policy. 

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

The default number of games is 100 and the default seed is 42:

```sh
python main.py --num-games 100 --seed 42 --opponent random
python main.py --num-games 100 --seed 42 --opponent random rule_based --switch-every 25
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