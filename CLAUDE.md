# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic trajectory generator for 20 Questions games that stress-test MAST (Model-Agnostic Stress Testing) failure modes. Generates trajectories (T1-T8 archetypes) with specific entropy/information-gain patterns to evaluate how agents handle edge cases in belief updating.

## Commands

```bash
# Generate single trajectory (prints to console)
python run.py single --type T2

# Generate with overlays
python run.py single --type T8 --termination wrong_guess --overlay calibrated

# Batch generation
python run.py batch --count 800 --distribution uniform --output-dir outputs/batch

# Validate trajectories against gate constraints
python run.py validate outputs/batch --show-failures

# List available items/secrets
python run.py list-items
```

No tests currently. No build step (stdlib only, no dependencies).

## Architecture

### Two Generation Modes

**Secret-first** (`SecretFirstGenerator`): Picks secret first, answers derive from secret. Used for T1, T6.

**Path-first** (`PathFirstGenerator`): Builds question-answer sequence following split ratio constraints, samples secret from final feasible set. Used for T2-T5, T7-T8.

### Core Data Flow

1. `CUQDataset` loads questions (122K with bitmasks) and items (128 secrets) from `data/`
2. `QuestionIndex` provides fast lookups by split ratio, information gain, redundancy
3. Generators use `bitmask.py` for 128-bit feasible set operations (entropy, split_ratio, update_state)
4. `archetypes.py` defines per-turn constraints (split ranges, branch policies) for each T1-T8 type
5. `OverlayChain` applies termination + prediction behaviors post-generation
6. `validators.py` checks if trajectories meet their type's gate constraints

### Key Abstractions

- **Bitmask state**: Python int representing 128-bit feasible set. Core operations in `bitmask.py`.
- **SplitConstraint**: Per-turn rules (ratio bounds, branch policy, special requirements like `require_no_op`)
- **Overlays**: Two families - prediction (belief-report channel) and termination (action channel). Applied via `OverlayChain`.
- **Gate validators**: Type-specific checks (e.g., T1 requires all balanced splits, T8 requires wrong guess with verification_claim)

### MAST Mode Derivation

`models.derive_mast_modes()` computes which failure modes a trajectory actually instantiates based on trajectory type + overlay tags. The static `TRAJECTORY_MAST_MAPPING` is legacy; prefer the derived version.

## Data Requirements

Place CUQ dataset in `data/`:
- `questions.jsonl` - 122,913 questions with `question_id`, `question`, `bitmask` (int)
- `items.txt` - 128 item names, one per line
