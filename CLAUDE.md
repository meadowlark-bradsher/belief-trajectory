# Belief Trajectory Generator

Synthetic trajectory generator for 20 Questions games that stress-test specific failure modes (T1-T8 archetypes targeting MAST failure modes).

## Quick Start

```bash
# Generate a single T2 trajectory
python belief-trajectory/run.py single --type T2

# Generate with specific secret and overlays
python belief-trajectory/run.py single --type T7 --secret Bear --overlay overconfident sticky

# Batch generation
python belief-trajectory/run.py batch --count 500 --output-dir outputs/batch

# Focus on specific types
python belief-trajectory/run.py batch --count 200 --types T6 T7 T8 --output-dir outputs/fc3_focus
```

## Project Structure

```
belief-trajectory/
├── src/
│   ├── models.py           # Trajectory and turn dataclasses
│   ├── config.py           # Configuration dataclasses
│   ├── index.py            # Question indexing by split characteristics
│   ├── loader.py           # Load CUQ dataset
│   ├── bitmask.py          # Integer bitmask operations
│   ├── storage.py          # JSON output persistence
│   ├── generators/
│   │   ├── base.py         # Abstract TrajectoryGenerator
│   │   ├── secret_first.py # Secret-first mode (T1, T6)
│   │   ├── path_first.py   # Path-first mode (T2, T3, T4, T5, T7, T8)
│   │   └── archetypes.py   # T1-T8 constraint functions
│   └── overlays/
│       ├── base.py         # Abstract PredictionOverlay
│       ├── calibrated.py   # Rational baseline
│       ├── overconfident.py
│       ├── sticky.py
│       ├── commit_early.py
│       ├── refuses_revise.py
│       └── chain.py        # Overlay composition
├── run.py                  # CLI entry point
└── tests/
```

## Trajectory Archetypes (T1-T8)

| ID | Archetype | MAST Modes | Generation Mode |
|----|-----------|------------|-----------------|
| T1 | Smooth halving (control) | baseline | secret_first |
| T2 | Early collapse → rare-branch reversal | FM-1.1, FM-2.6, FM-3.3 | path_first |
| T3 | Plateau → forced resolution | FM-1.5, FM-3.1 | path_first |
| T4 | Redundant loop (low IG questions) | FM-1.3 | path_first |
| T5 | Multi-modal ambiguity | FM-2.2 | path_first |
| T6 | Prediction-belief mismatch | FM-2.6 | secret_first |
| T7 | Late shock after confidence | FM-3.1, FM-3.2 | path_first |
| T8 | Wrong-way update | FM-3.3 | path_first |

## Generation Modes

### Secret-First (T1, T6)
1. Choose secret
2. Select questions matching target entropy curve
3. Answers determined by secret

### Path-First (T2, T3, T4, T5, T7, T8)
1. Build question-answer sequence following constraints
2. Sample secret from final feasible set
3. Answers verified consistent with secret

## Prediction Overlays

Overlays generate model predictions for each turn:

- **calibrated**: Predict YES if p_yes >= 0.5, confidence = |p_yes - 0.5| * 2
- **overconfident**: Same argmax, always 95% confidence
- **sticky**: Persist previous high-confidence predictions
- **commit_early**: Lock to MAP after entropy threshold
- **refuses_revise**: Keep wrong answer K turns after contradiction

Overlays compose via priority chain. Higher priority overlays are tried first.

## Data Dependencies

Uses CUQ dataset:
- `flan-oracle/cuq/release/questions.jsonl` - 122,913 questions with bitmasks
- `flan-oracle/cuq/release/items.txt` - 128 items

## Output Format

```json
{
  "trajectory_id": "abc123",
  "trajectory_type": "T2",
  "target_mast_modes": ["FM-1.1", "FM-2.6", "FM-3.3"],
  "generation_mode": "path_first",
  "secret": "Bear",
  "secret_index": 5,
  "turns": [
    {
      "turn": 1,
      "question_id": 42,
      "question": "Is it alive?",
      "answer": true,
      "feasible_set_size_before": 128,
      "feasible_set_size_after": 64,
      "entropy_before": 7.0,
      "entropy_after": 6.0,
      "split_ratio": 0.5,
      "branch_taken": "yes",
      "branch_probability": 0.5,
      "prediction": {
        "predicted_answer": true,
        "confidence": 0.0
      }
    }
  ]
}
```

## Key Implementation Details

### Bitmask Operations
Uses 128-bit integers (Python ints) for feasible sets. Each bit i represents item i.
- `popcount(state)` - Count remaining items
- `split_ratio(state, mask)` - Proportion answering YES
- `update_state(state, mask, answer)` - Apply question result

### Question Index
Efficient lookup by split characteristics:
- `find_near_balanced()` - 0.4-0.6 split
- `find_extreme_split()` - <0.1 or >0.9
- `find_redundant_with()` - High bitmask overlap

### Archetype Constraints
Each archetype defines per-turn constraints:
- Split ratio range
- Branch policy (LIKELY, UNLIKELY, RANDOM, SECRET)
- Special conditions (require_no_op for T3/T4, require_high_ig for resolution phases)
