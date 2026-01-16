# FM-2: Belief Failures

Belief failures occur when the agent's internal beliefs are miscalibrated or inconsistent.

## FM-2.2: Multi-modal Ambiguity

**Definition**: Agent fails to maintain appropriate uncertainty over multiple hypotheses.

**In 20 Questions**: Collapsing to a single hypothesis too early when multiple items remain equally plausible.

**Trajectories**: T5

### Example

At turn 5 with |S| = 7:

```
Feasible: {Guitar, Violin, Cello, Drum, Flute, Piano, Harp}

BAD (over-confident):
  "I'm 80% sure it's Guitar"

GOOD (multi-modal):
  "Each of the 7 instruments is roughly equally likely (14% each)"
```

### The Problem

When the feasible set contains multiple items with similar likelihoods, the agent should:

1. Maintain a distribution over items (not point estimate)
2. Acknowledge uncertainty in predictions
3. Continue asking distinguishing questions

### Detection

```python
def detect_fm22(trajectory):
    for turn in trajectory.turns:
        # Mid-game with multiple items
        if 4 <= turn.feasible_set_size_after <= 16:
            # Check if prediction confidence is inappropriately high
            if turn.prediction and turn.prediction.confidence > 0.8:
                # High confidence with many items = FM-2.2
                return True
    return False
```

---

## FM-2.6: Reasoning-Action Mismatch

**Definition**: Agent's stated reasoning contradicts the action outcome.

**In 20 Questions**: Predicting YES (based on majority) when the oracle answers NO.

**Trajectories**: T2, T6

### Example

```json
{
  "turn": 1,
  "question": "Is it alive?",
  "split_ratio": 0.70,
  "prediction": {
    "predicted_answer": true,   // "70% would say YES, so I predict YES"
    "confidence": 0.40
  },
  "answer": false               // Oracle says NO
}
```

**Reasoning**: "Most items are alive, so the answer is probably YES"

**Reality**: The secret is not alive

### Why This Happens

The mismatch isn't necessarily a "bug" - a well-calibrated predictor *should* sometimes be wrong. The issue is when:

1. The agent claims certainty it doesn't have
2. The agent fails to update beliefs after the mismatch
3. The agent doesn't recognize that its predictions are systematically biased

### Detection

```python
def detect_fm26(trajectory):
    mismatch_count = 0
    for turn in trajectory.turns:
        if turn.prediction:
            predicted = turn.prediction.predicted_answer
            actual = turn.answer
            if predicted != actual:
                mismatch_count += 1
    # Multiple mismatches indicate systematic issue
    return mismatch_count >= 3
```

---

## Summary Table

| Mode | Failure | Detection Signal | Trajectory |
|------|---------|------------------|------------|
| FM-2.2 | Over-commitment | High confidence with \|S\| > 4 | T5 |
| FM-2.6 | Predict ≠ Actual | prediction.answer ≠ turn.answer | T2, T6 |

## Training Implications

For FM-2 failures, the reward should penalize:

1. **Overconfidence**: Confidence exceeds 1/|S|
2. **Brier score**: (predicted_prob - actual)²
3. **Calibration error**: Mean difference between confidence and accuracy
