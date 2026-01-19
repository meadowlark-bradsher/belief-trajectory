# Belief Trajectory Generator

Synthetic trajectory generator for 20 Questions (20Q) games that stress-test specific **MAST failure modes**.

## What is this?

This toolkit generates trajectories for a 20Q game where:

- An **oracle** knows a secret item (one of 128 possibilities).
- A **guesser** asks yes/no questions to identify the secret.
- Each question partitions the feasible set based on the oracle’s answer (implemented via bitmask updates).
- The system is designed as a diagnostic tool for 20Q-style (iteratively probing) agentic tasks.

We use **MAST (Multi-Agent System Failure Taxonomy)** terminology for failure-mode naming and attribution (Cemri et al., 2025, arXiv:2503.13657). See [docs/mast/index.md](mast/index.md).

The trajectories are designed to exhibit specific failure patterns that can be used to:

1. **Train** models to report epistemic uncertainty that matches the environment’s uncertainty.
2. **Evaluate** calibration, recovery behavior, and termination decisions.
3. **Generate reward signals** for RL-based correction (optional).

## Trajectory Archetypes (T1–T8)

| Type | Pattern | MAST Mode | Use Case |
|------|---------|-----------|----------|
| T1 | Smooth halving | baseline | Control condition |
| T2 | Early collapse + later shock window | FM-2.6 | Prediction–belief mismatch |
| T3 | Plateau → resolution | FM-3.1 / FM-1.5 | Premature stop / unaware termination (overlay) |
| T4 | Redundant / low-IG streak | FM-1.3 | Step repetition |
| T5 | Multi-modal / skewed mid-game regime | FM-2.2 | Ambiguity handling |
| T6 | Induced prediction mismatch | FM-2.6 | Calibration failure |
| T7 | Late shock | FM-3.1 / FM-3.2 | Premature stop / skipped verification (overlay) |
| T8 | Wrong verification claim | FM-3.3 | Incorrect verification |

## Quick Start

```bash
# Generate a single trajectory
python run.py single --type T8 --termination wrong_guess

# Batch generation
python run.py batch --count 800 --distribution uniform --output-dir outputs/

# Validate trajectories
python run.py validate outputs/ --show-failures
````

## Key Concepts

### State space

At each turn, the environment maintains:

* **|S|** — feasible set size (number of possible secrets remaining)
* **H** — entropy proxy, `log2(|S|)`
* **History** — sequence of (question, oracle answer) pairs

### Action space

The guesser can:

* **ASK(question)** — ask a yes/no question
* **GUESS(item)** — commit to a final answer
* *(Optional in overlays)*: **STOP** — attempt to terminate early (may be accepted/rejected)

### Failure modes

MAST failure modes categorize how agents fail:

* **FM-1.x** — specification failures (wrong or rule-violating actions)
* **FM-2.x** — belief failures (miscalibration, mismatch handling)
* **FM-3.x** — verification failures (bad termination or incorrect verification claims)

See [MAST Failure Modes](mast/index.md) for definitions and the 20Q-specific mapping.