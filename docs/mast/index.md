# MAST Failure Modes

MAST (Model Alignment and Safety Taxonomy) categorizes how AI agents fail during multi-step tasks.

## Overview

| Category | Focus | Failure Type |
|----------|-------|--------------|
| FM-1.x | Specification | Wrong actions |
| FM-2.x | Belief | Miscalibration |
| FM-3.x | Verification | Bad termination |

## Failure Modes in This Toolkit

### FM-1: Specification Failures

| Mode | Description | Trajectory |
|------|-------------|------------|
| FM-1.1 | Disobey task specification | T2, T8 |
| FM-1.3 | Step repetition / redundancy | T4 |
| FM-1.5 | Unaware of termination conditions | T3 |

### FM-2: Belief Failures

| Mode | Description | Trajectory |
|------|-------------|------------|
| FM-2.2 | Multi-modal ambiguity | T5 |
| FM-2.6 | Reasoning-action mismatch | T2, T6 |

### FM-3: Verification Failures

| Mode | Description | Trajectory |
|------|-------------|------------|
| FM-3.1 | Premature termination | T3, T7 |
| FM-3.2 | Incomplete verification | T7 |
| FM-3.3 | Incorrect verification | T8 |

## Mapping Trajectories to Modes

```
T1 (baseline)     → no failure mode (control)
T2 (collapse)     → FM-2.6 (prediction ≠ reality)
T3 (plateau)      → FM-3.1 (premature stop) or FM-1.5 (unaware)
T4 (redundant)    → FM-1.3 (wasted questions)
T5 (multi-modal)  → FM-2.2 (ambiguous state)
T6 (mismatch)     → FM-2.6 (calibration error)
T7 (late shock)   → FM-3.1 (confidence collapse)
T8 (wrong verify) → FM-3.3 (false verification claim)
```

## Overlay Requirements

Some failure modes require specific overlays:

| Mode | Required Overlay | Tag |
|------|------------------|-----|
| FM-3.1 | `premature_stop` | `term:premature_stop` |
| FM-3.3 | `wrong_guess` | `term:wrong_guess` + `verify:claim_present` |
| FM-1.5 | `unaware` | `term:unaware` |

World-only modes (no overlay needed):

- FM-1.3 (T4 world enforces low-IG)
- FM-2.2 (T5 world enforces ambiguity)
- FM-2.6 (T2/T6 world creates mismatch opportunity)

## Deriving Modes from Trajectories

The `target_mast_modes` field is derived automatically:

```python
def derive_mast_modes(trajectory_type: str, overlay_tags: list[str]) -> list[str]:
    modes = []

    # World-only modes
    if trajectory_type == "T4":
        modes.append("FM-1.3")
    if trajectory_type == "T5":
        modes.append("FM-2.2")

    # Overlay-dependent modes
    if trajectory_type in ("T2", "T6") and "pred:calibrated_argmax" in overlay_tags:
        modes.append("FM-2.6")
    if trajectory_type in ("T3", "T7") and "term:premature_stop" in overlay_tags:
        modes.append("FM-3.1")
    if trajectory_type == "T8" and "term:wrong_guess" in overlay_tags:
        modes.append("FM-3.3")

    return modes
```
