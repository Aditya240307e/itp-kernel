# Repros

Small scripts that reproduce kernel findings (see FINDINGS.md). Each one prints what it shows and runs against the current build (`python3 repros/<name>.py` from this directory's parent).

| script | finding | what it shows |
|---|---|---|
| `conv_not_transitive.py` | F28 | `conv` answers False for two terms that each convert to their own normal form (stuck `match`es are compared structurally): incomplete, never unsound |
| `conv_exponential_shared.py` | F29 | `conv` costs about 3^n guarded steps on two chains of n links that share every node (n = 18: 3.9e8 steps, 20 s) |
| `wrong_apply_budget.py` | F30 | a wrongly instantiated `apply` against a big goal ends in `FuelExhausted` after the step budget, leaving the proof state unchanged |
