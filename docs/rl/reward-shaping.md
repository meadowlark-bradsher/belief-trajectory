# Reward Shaping

Different reward functions emphasize different failure modes.

## Sparse Reward (Baseline)

```python
reward = +1 if guess_correct else -1 if guessed else 0
```

**Pros**: Simple, clear signal
**Cons**: Sparse, slow learning

## Dense Reward Options

### Information Gain Reward

Rewards informative questions:

```python
reward = information_gain(question, state)
# IG = H(before) - H(after)
```

**Best for**: FM-1.3 (redundant loop), T4 trajectories

### Entropy Penalty

Penalizes stopping with high uncertainty:

```python
if action == STOP or action == GUESS:
    reward = -entropy_at_stop
```

**Best for**: FM-3.1 (premature termination), T3/T7 trajectories

### Calibration Reward

Penalizes confident wrong guesses:

```python
if guess_correct:
    reward = +1
else:
    reward = -confidence  # more confident = worse
```

**Best for**: FM-3.3 (incorrect verification), T8 trajectories

### Efficiency Bonus

Rewards faster solutions:

```python
reward = +1 - 0.1 * num_turns  # if correct
reward = -1                     # if wrong
```

**Best for**: General efficiency, avoiding T4-style waste

## Composite Rewards

Combine signals for multi-objective learning:

```python
reward = (
    +1.0 * correct_guess
    - 1.0 * wrong_guess
    + 0.1 * information_gain
    - 0.05 * per_turn_cost
    - 0.5 * (confidence if wrong_guess else 0)
)
```

## Trajectory-Specific Recommendations

| Type | Primary Failure | Recommended Reward |
|------|-----------------|-------------------|
| T4 | Low IG questions | +IG per turn |
| T3/T7 | Premature stop | -H at termination |
| T8 | Wrong confident guess | -conf if wrong |
| T2/T6 | Miscalibration | Brier score penalty |
