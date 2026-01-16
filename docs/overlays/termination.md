# Termination Overlays

Termination overlays operate on the **action channel** - they determine when the model attempts to stop/guess and whether the environment accepts.

## Available Overlays

### rational (default)

Guesses when feasible set is singleton.

```python
if feasible_set_size == 1:
    action = "guess"
    confidence = 1.0
```

**Tag**: `term:rational`

### premature_stop

Forces stop/guess when entropy is still high.

```python
if entropy >= threshold and turn >= min_turn:
    action = "stop"
    stop_reason = f"Premature stop at H={entropy:.2f} bits"
```

**Tag**: `term:premature_stop`

**Parameters**:

- `entropy_threshold`: Stop if H ≥ this (default: 4.0 bits)
- `feasible_threshold`: Stop if |S| ≥ this (default: 16)
- `min_turn`: Don't stop before this turn (default: 3)

### unaware

Continues questioning past when it should stop.

```python
if feasible_set_size <= threshold:
    # Should stop, but continues for extra_turns more
    action = "continue"
```

**Tag**: `term:unaware`

### wrong_guess

Forces an incorrect guess with false verification claim.

```python
# Pick a wrong secret
wrong_index = random.choice([i for i in range(128) if i != secret_index])

action = "guess"
guess = Guess(
    secret=items[wrong_index],
    confidence=0.9,
    verification_claim="The secret must be X because..."
)
```

**Tag**: `term:wrong_guess`

**Additional tag**: `verify:claim_present` (auto-added when verification_claim exists)

## Usage

```bash
# For FM-3.1 (premature termination)
python run.py single --type T7 --termination premature_stop

# For FM-3.3 (incorrect verification)
python run.py single --type T8 --termination wrong_guess
```

## Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_action` | string | "continue", "guess", or "stop" |
| `guess` | object | Guess details if action="guess" |
| `guess_correct` | bool | Whether guess matched secret |
| `stop_reason` | string | Why the model stopped |
| `stop_accepted` | bool | Whether environment accepted the stop |

## Stop Acceptance Semantics

- `stop_accepted=true`: Terminal action on final turn
- `stop_accepted=false`: Stop attempted but rejected (game continues)
- `stop_accepted=null`: No stop attempted

This allows trajectories to include *rejected* stop attempts that the agent must recover from.
