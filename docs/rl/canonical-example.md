# Canonical RL Example

This page illustrates how a single trajectory provides training signal for reinforcement learning.

## The Setup

**State space** = (entropy, feasible set size, turn history)

**Action space** = {ASK(question), GUESS(item)}

**Reward** = +1 correct guess, -1 wrong guess, 0 otherwise

---

## Example Trajectory: T8_e5f3ff15

Secret: **Horse** (index 56)

| Turn | State (H, \|S\|) | Action | Reward |
|------|------------------|--------|--------|
| 1 | (7.0, 128) | ASK("leisure activities?") | 0 |
| 2 | (6.0, 64) | ASK("professional purposes?") | 0 |
| 3 | (5.0, 32) | ASK("found in a store?") | 0 |
| 4 | **(2.3, 5)** | **GUESS("Watermelon")** | **-1** |
| 5 | (1.6, 3) | ASK("smaller than car?") | 0 |
| 6 | (0.0, 1) | GUESS("Horse") | +1 |

---

## The Failure Point (Turn 4)

```json
{
  "turn": 4,
  "entropy_after": 1.58,
  "feasible_set_size_after": 3,
  "model_action": "guess",
  "guess": {
    "secret": "Watermelon",
    "confidence": 0.9,
    "verification_claim": "The secret must be 'Watermelon' because
      the answer to 'Is it something used for leisure-time activities?'
      was Yes, which is consistent with 'Watermelon'."
  },
  "guess_correct": false,
  "stop_accepted": false
}
```

### What went wrong?

1. **Entropy too high**: H = 1.58 bits means ~3 items remain
2. **Overconfident**: Claimed 90% confidence when true probability ≈ 33%
3. **False verification**: Claimed consistency, but 2 other items are also consistent

---

## The Learning Signal

### Negative signal (Turn 4)

```
state:  H=1.58 bits, |S|=3
action: GUESS("Watermelon", conf=0.9)
reward: -1
```

**Lesson**: "At H ≈ 1.5 bits, GUESS yields negative reward. Prefer ASK."

### Positive signal (Turn 6)

```
state:  H=0 bits, |S|=1
action: GUESS("Horse", conf=1.0)
reward: +1
```

**Lesson**: "At H = 0 bits, GUESS yields positive reward."

---

## Reward Shaping Options

The sparse reward (+1/-1) can be augmented:

| Signal | Formula | Effect |
|--------|---------|--------|
| Information gain | +IG per question | Encourages informative questions |
| Entropy penalty | -H at termination | Penalizes early stopping |
| Confidence penalty | -conf if wrong | Penalizes overconfidence |
| Turn cost | -0.1 per turn | Encourages efficiency |

### Example: Calibration-aware reward

```python
if action == GUESS:
    if correct:
        reward = +1
    else:
        # Penalize proportional to confidence
        reward = -confidence  # -0.9 for this example
```

This teaches the agent: "If you're going to be wrong, at least be uncertain about it."

---

## Why T8 is Canonical

T8 trajectories include:

1. **Explicit failure** - `guess_correct=false`
2. **Stated reasoning** - `verification_claim` to critique
3. **Recovery** - Game continues, correct guess follows
4. **Measurable state** - Entropy at failure is recorded

This makes them ideal for teaching because failure and correction are both present in the same trajectory.
