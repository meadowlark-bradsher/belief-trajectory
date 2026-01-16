# Trajectory Archetypes

The generator produces 8 distinct trajectory types (T1-T8), each designed to stress-test specific failure modes.

## Overview

| Type | Name | Entropy Pattern | MAST Mode |
|------|------|-----------------|-----------|
| T1 | Smooth halving | 7→6→5→4→3→2→1→0 | baseline |
| T2 | Early collapse | 7→3→2→...→0 | FM-2.6 |
| T3 | Plateau | 7→6.9→6.8→...→4→2→0 | FM-3.1 |
| T4 | Redundant loop | 7→6.99→6.98→... | FM-1.3 |
| T5 | Multi-modal | irregular | FM-2.2 |
| T6 | Prediction mismatch | varies | FM-2.6 |
| T7 | Late shock | 7→4→2→1→0.5→0 | FM-3.1 |
| T8 | Wrong verification | varies | FM-3.3 |

## Generation Modes

### Secret-First (T1, T6)

1. Choose secret item
2. Select questions matching target entropy curve
3. Answers determined by secret

Best for: Controlled entropy curves where the secret constrains answers.

### Path-First (T2, T3, T4, T5, T7, T8)

1. Build question-answer sequence following constraints
2. Sample secret from final feasible set
3. Verify answer consistency

Best for: Specific branch patterns (rare branches, plateaus) that require answer control.

## Entropy Curves

```
T1 (baseline):     ████████████████████████  (smooth descent)
T2 (early collapse): ████████░░░░░░████████  (sharp drop, recovery)
T3 (plateau):      ████████████████░░░░░░░░  (flat, then drop)
T4 (redundant):    ████████████████████████  (nearly flat)
T5 (multi-modal):  ████████░░██░░████░░░░░░  (irregular)
T7 (late shock):   ████████████████████░░░░  (smooth, then sharp)
```

## Gate Validators

Each type has an automated validator:

```bash
python run.py validate outputs/ --show-failures
```

| Type | Gate Constraint |
|------|-----------------|
| T1 | All splits balanced (\|p - 0.5\| ≤ 0.15) |
| T2 | Rare branch in turns 1-3 AND after turn 3 |
| T3 | ≥3 consecutive low-IG turns, then resolution |
| T4 | ≥3 consecutive low-IG turns |
| T5 | Not all balanced + skewed mid-game |
| T6 | (none) |
| T7 | Rare branch in last 2 turns |
| T8 | Wrong guess + verification_claim |
