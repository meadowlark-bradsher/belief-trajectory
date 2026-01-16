# FM-1: Specification Failures

Specification failures occur when the agent takes actions that violate task requirements.

## FM-1.1: Disobey Task Specification

**Definition**: Agent takes an action explicitly forbidden by the task.

**In 20 Questions**: Guessing an item that is not in the feasible set.

**Trajectories**: T2, T8 (with `wrong_guess` overlay)

### Example

```json
{
  "turn": 4,
  "feasible_set": ["Horse", "Cow", "Dog"],
  "guess": {
    "secret": "Watermelon",  // NOT in feasible set
    "confidence": 0.9
  },
  "guess_correct": false
}
```

### Detection

```python
def detect_fm11(trajectory):
    for turn in trajectory.turns:
        if turn.guess and not turn.guess_correct:
            # Wrong guess = potential FM-1.1
            return True
    return False
```

---

## FM-1.3: Step Repetition

**Definition**: Agent repeats the same ineffective action multiple times.

**In 20 Questions**: Asking consecutive low-information-gain questions.

**Trajectories**: T4

### Example

```
Turn 1: "Is it a hat?"      → IG = 0.01 bits
Turn 2: "Is it a bracelet?" → IG = 0.01 bits
Turn 3: "Is it a shoe?"     → IG = 0.01 bits
```

Each question only eliminates 1 item when ~64 could be eliminated with a balanced question.

### Detection

```python
def detect_fm13(trajectory):
    low_ig_streak = 0
    for turn in trajectory.turns:
        ig = turn.entropy_before - turn.entropy_after
        if ig < 0.2:
            low_ig_streak += 1
        else:
            low_ig_streak = 0
        if low_ig_streak >= 3:
            return True
    return False
```

---

## FM-1.5: Unaware of Termination Conditions

**Definition**: Agent continues working past when it should stop.

**In 20 Questions**: Asking more questions when |S| = 1.

**Trajectories**: T3 (with `unaware` overlay)

### Example

```
Turn 8: |S| = 2, action = "ask"  ← reasonable
Turn 9: |S| = 1, action = "ask"  ← SHOULD GUESS
Turn 10: |S| = 1, action = "ask" ← FM-1.5
Turn 11: |S| = 1, action = "guess" ← finally
```

### Detection

```python
def detect_fm15(trajectory):
    for turn in trajectory.turns:
        if turn.feasible_set_size_before == 1:
            if turn.model_action == "continue":
                return True
    return False
```

---

## Summary Table

| Mode | Failure | Detection Signal | Trajectory |
|------|---------|------------------|------------|
| FM-1.1 | Wrong action | `guess_correct=false` | T2, T8 |
| FM-1.3 | Repeated waste | IG < 0.2 for 3+ turns | T4 |
| FM-1.5 | Over-continuing | action="ask" when \|S\|=1 | T3 |
