# Belief Trajectory Generator

Synthetic trajectory generator for 20 Questions games that stress-test specific **MAST failure modes**.

## What is this?

This toolkit generates trajectories for a 20 Questions game where:

- An **oracle** knows a secret item (one of 128 possibilities)
- A **guesser** asks yes/no questions to identify the secret
- Each question partitions the feasible set based on the oracle's answer
- Is designed as a diagnostic tool for 20Q posed as an agentic task

The trajectories are designed to exhibit specific **failure patterns** that can be used to:

1. **Train** models to recognize epistemic uncertainty
2. **Evaluate** model calibration and termination decisions
3. **Generate reward signals** for RL-based correction

## Trajectory Archetypes (T1-T8)

| Type | Pattern | MAST Mode | Use Case |
|------|---------|-----------|----------|
| T1 | Smooth halving | baseline | Control condition |
| T2 | Early collapse | FM-2.6 | Prediction-belief mismatch |
| T3 | Plateau → resolution | FM-3.1 | Premature termination |
| T4 | Redundant loop | FM-1.3 | Step repetition |
| T5 | Multi-modal | FM-2.2 | Ambiguity handling |
| T6 | Prediction mismatch | FM-2.6 | Calibration failure |
| T7 | Late shock | FM-3.1 | Confidence collapse |
| T8 | Wrong verification | FM-3.3 | Incorrect verification |

## Quick Start

```bash
# Generate a single trajectory
python run.py single --type T8 --termination wrong_guess

# Batch generation
python run.py batch --count 800 --distribution uniform --output-dir outputs/

# Validate trajectories
python run.py validate outputs/ --show-failures
```

## Key Concepts

### State Space

At each turn, the state is:

- **H** - Entropy (bits of uncertainty remaining)
- **|S|** - Feasible set size (number of possible secrets)
- **History** - Sequence of (question, answer) pairs

### Action Space

The guesser can:

- **ASK(question)** - Ask a yes/no question
- **GUESS(item)** - Commit to a final answer

### Failure Modes

MAST failure modes categorize how agents fail:

- **FM-1.x** - Specification failures (wrong actions)
- **FM-2.x** - Belief failures (miscalibration)
- **FM-3.x** - Verification failures (bad termination)

See [MAST Failure Modes](mast/index.md) for details.
