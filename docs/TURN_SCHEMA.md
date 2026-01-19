# Turn Schema

Schema documentation for trajectory turns in the belief trajectory generator.

## TrajectoryTurn

A single turn in a 20 Questions game trajectory.

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `turn` | integer | yes | Turn number (1-indexed) |
| `question_id` | integer | yes | Reference to question in the pool |
| `question` | string | yes | The question text asked |
| `answer` | boolean | yes | Oracle answer: `true` = YES, `false` = NO |

### Feasible Set Tracking

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `feasible_set_size_before` | integer | yes | Number of possible secrets before this turn |
| `feasible_set_size_after` | integer | yes | Number of possible secrets after this turn |

### Entropy Tracking

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entropy_before` | float | yes | Shannon entropy (bits) before this turn |
| `entropy_after` | float | yes | Shannon entropy (bits) after this turn |

**Note:** Entropy is computed as `log2(feasible_set_size)` assuming uniform distribution.

### Split Characteristics

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `split_ratio` | float | yes | Proportion of feasible set answering YES (0.0-1.0) |
| `branch_taken` | string | yes | Which branch was followed: `"yes"` or `"no"` |
| `branch_probability` | float | yes | Probability of the taken branch |

**Split ratio interpretation:**
- `0.5` = balanced split (maximum information gain)
- `0.0` or `1.0` = no-op question (zero information gain)
- `< 0.1` or `> 0.9` = extreme split (potential shock)

### Action Channel (Termination)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_action` | string | yes | Action taken: `"continue"`, `"guess"`, or `"stop"` |
| `guess` | Guess | no | Present if `model_action == "guess"` |
| `guess_correct` | boolean | no | Whether the guess was correct |
| `stop_reason` | string | no | Reason for stopping (if stopped) |
| `stop_accepted` | boolean | no | Whether the stop was valid |

### Belief-Report Channel (Prediction)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prediction` | Prediction | no | Model's prediction for this turn |

### State Tracking (Optional)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `state_before_hex` | string | no | Hex-encoded 128-bit feasible set bitmask |
| `question_bitmask_hex` | string | no | Hex-encoded 128-bit question bitmask |

---

## Guess

A guess action by the model.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `secret_index` | integer | yes | Index of guessed item (0-127) |
| `secret` | string | yes | Name of guessed item |
| `confidence` | float | yes | Model confidence (0.0-1.0) |
| `verification_claim` | string | no | Verification statement (for FM-3.3 wrong verification) |

---

## Prediction

Model's belief-report for a turn.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `predicted_answer` | boolean | yes | Predicted answer: `true` = YES, `false` = NO |
| `confidence` | float | yes | Confidence in prediction (0.0-1.0) |

---

## Example Turn

```json
{
  "turn": 4,
  "question_id": 12847,
  "question": "Is it used for decoration?",
  "answer": false,
  "feasible_set_size_before": 12,
  "feasible_set_size_after": 8,
  "entropy_before": 3.58,
  "entropy_after": 3.0,
  "split_ratio": 0.33,
  "branch_taken": "no",
  "branch_probability": 0.67,
  "model_action": "continue",
  "prediction": {
    "predicted_answer": false,
    "confidence": 0.72
  }
}
```

## Example Turn with Guess

```json
{
  "turn": 12,
  "question_id": 45231,
  "question": "Is it a fruit?",
  "answer": true,
  "feasible_set_size_before": 3,
  "feasible_set_size_after": 1,
  "entropy_before": 1.58,
  "entropy_after": 0.0,
  "split_ratio": 0.33,
  "branch_taken": "yes",
  "branch_probability": 0.33,
  "model_action": "guess",
  "guess": {
    "secret_index": 87,
    "secret": "Watermelon",
    "confidence": 0.95,
    "verification_claim": null
  },
  "guess_correct": true,
  "prediction": {
    "predicted_answer": true,
    "confidence": 0.88
  }
}
```

## Example Turn with Wrong Verification (FM-3.3)

```json
{
  "turn": 8,
  "question_id": 33102,
  "question": "Can you eat it?",
  "answer": true,
  "feasible_set_size_before": 4,
  "feasible_set_size_after": 2,
  "entropy_before": 2.0,
  "entropy_after": 1.0,
  "split_ratio": 0.5,
  "branch_taken": "yes",
  "branch_probability": 0.5,
  "model_action": "guess",
  "guess": {
    "secret_index": 42,
    "secret": "Apple",
    "confidence": 0.92,
    "verification_claim": "Based on the answers, it must be Apple because it's edible, found in nature, and commonly red."
  },
  "guess_correct": false,
  "prediction": {
    "predicted_answer": true,
    "confidence": 0.85
  }
}
```

---

## Derived Metrics

These can be computed from the turn fields:

| Metric | Formula | Description |
|--------|---------|-------------|
| Information Gain | `entropy_before - entropy_after` | Bits of information gained |
| Elimination Ratio | `1 - (size_after / size_before)` | Proportion of items eliminated |
| Split Balance | `1 - abs(split_ratio - 0.5) * 2` | How balanced the split is (1.0 = perfect) |
| Surprise | `-log2(branch_probability)` | Surprisal of the taken branch |

---

## Trajectory Type Patterns

Different trajectory types have characteristic turn patterns:

| Type | Pattern | Typical Split Ratios |
|------|---------|---------------------|
| T1 | Smooth halving | ~0.5 throughout |
| T2 | Early collapse | Extreme early, balanced late |
| T3 | Plateau | Near-plateau, then balanced |
| T4 | Redundant loop | Consecutive ~0.0 or ~1.0 |
| T5 | Multi-modal | Skewed mid-game |
| T6 | Mismatch | Balanced (mismatch in prediction) |
| T7 | Late shock | Balanced early, extreme late |
| T8 | Wrong verification | Balanced + wrong guess |
