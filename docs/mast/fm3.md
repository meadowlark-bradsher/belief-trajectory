# FM-3: Verification Failures

Verification failures occur when the agent incorrectly decides to stop or validates its answer incorrectly.

## FM-3.1: Premature Termination

**Definition**: Agent stops before completing the task.

**In 20 Questions**: Stopping or guessing when entropy is still high (|S| > 1).

**Trajectories**: T3, T7 (with `premature_stop` overlay)

### Example

```json
{
  "turn": 3,
  "entropy_after": 5.13,
  "feasible_set_size_after": 35,
  "model_action": "stop",
  "stop_reason": "Premature stop at entropy 5.13 bits, |S|=35",
  "stop_accepted": false
}
```

The agent attempts to stop when 35 items remain. The environment rejects this.

### Why This Happens

1. **Impatience**: Agent wants to finish quickly
2. **Misestimation**: Agent thinks it has enough information
3. **Plateau fatigue**: Agent gives up during low-IG streaks (T3)

### Detection

```python
def detect_fm31(trajectory):
    for turn in trajectory.turns:
        if turn.model_action in ("stop", "guess"):
            if turn.entropy_after > 1.0:  # More than 1 bit remaining
                return True
    return False
```

---

## FM-3.2: Incomplete Verification

**Definition**: Agent claims completion without checking all requirements.

**In 20 Questions**: Guessing without verifying the guess is consistent with all Q&A history.

**Trajectories**: T7

### Example

Agent guesses "Cat" at turn 6:

```
Q1: "Is it alive?" → YES       ✓ Cat is alive
Q2: "Is it a mammal?" → YES    ✓ Cat is a mammal
Q3: "Is it wild?" → NO         ✓ Cat is domestic
Q4: "Does it fly?" → YES       ✗ Cat doesn't fly!
```

The agent didn't verify Q4 before guessing.

### Detection

```python
def detect_fm32(trajectory):
    for turn in trajectory.turns:
        if turn.guess:
            # Check if guess is consistent with all answers
            # (This requires access to the bitmask)
            if not verify_consistency(turn.guess, trajectory):
                return True
    return False
```

---

## FM-3.3: Incorrect Verification

**Definition**: Agent claims to have verified but the verification is wrong.

**In 20 Questions**: Providing a `verification_claim` that incorrectly asserts consistency.

**Trajectories**: T8 (with `wrong_guess` overlay)

### Example

```json
{
  "turn": 4,
  "guess": {
    "secret": "Watermelon",
    "confidence": 0.9,
    "verification_claim": "The secret must be 'Watermelon' because
      the answer to 'Is it used for leisure?' was Yes,
      which is consistent with 'Watermelon'."
  },
  "guess_correct": false
}
```

**The error**: "Consistent with Watermelon" is true, but so are Horse and Dog. Consistency ≠ uniqueness.

### Why This is the Canonical RL Example

T8/FM-3.3 provides:

1. **Explicit failure**: `guess_correct=false`
2. **Stated reasoning**: `verification_claim` to critique
3. **Recovery**: Game continues, correct guess follows
4. **Training signal**: "When you claim X is verified, check that X is the *only* consistent item"

### Detection

```python
def detect_fm33(trajectory):
    for turn in trajectory.turns:
        if turn.guess:
            has_claim = turn.guess.verification_claim is not None
            is_wrong = not turn.guess_correct
            if has_claim and is_wrong:
                return True
    return False
```

---

## Summary Table

| Mode | Failure | Detection Signal | Trajectory |
|------|---------|------------------|------------|
| FM-3.1 | Early stop | stop at H > 1 bit | T3, T7 |
| FM-3.2 | Incomplete check | Inconsistent guess | T7 |
| FM-3.3 | Wrong verification | claim + wrong guess | T8 |

## Termination Decision Tree

```
Should I guess?
│
├─ |S| = 1? ──────────────── YES → GUESS (rational)
│
├─ |S| > 1 and confident? ── RISKY
│   │
│   ├─ Verified ALL Q&A? ─── YES → GUESS (FM-3.2 if wrong)
│   │
│   └─ Claimed verified? ─── YES → FM-3.3 if wrong
│
└─ |S| > 1 and uncertain? ── ASK another question
```

## Training Implications

For FM-3 failures, reward should penalize:

1. **Premature stop**: -entropy_at_stop
2. **Wrong guess**: -confidence (more confident = worse)
3. **False verification**: -1 if `verification_claim` and `!guess_correct`
