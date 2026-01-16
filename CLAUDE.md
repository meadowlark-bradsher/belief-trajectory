# Belief Trajectory Generator

Synthetic trajectory generator for 20 Questions games that stress-test specific MAST failure modes (T1-T8 archetypes).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt  # (none currently - stdlib only)
```

### 2. Download CUQ dataset

Place the CUQ dataset files in `data/`:

```
data/
├── questions.jsonl   # 122,913 questions with bitmasks
└── items.txt         # 128 items (secrets)
```

### 3. Verify installation

```bash
python run.py list-items  # Should list 128 items
```

## Quick Start

```bash
# Generate a single trajectory
python run.py single --type T2

# Generate with overlays
python run.py single --type T8 --termination wrong_guess

# Batch generation (100 per type)
python run.py batch --count 800 --distribution uniform --output-dir outputs/batch

# Validate existing trajectories
python run.py validate outputs/batch --show-failures
```

## Trajectory Archetypes

| Type | Archetype | MAST Mode | Key Property |
|------|-----------|-----------|--------------|
| T1 | Smooth halving | baseline | Balanced splits throughout |
| T2 | Early collapse | FM-2.6 | Rare branch early + late |
| T3 | Plateau | FM-3.1 | Low-IG streak, then resolution |
| T4 | Redundant loop | FM-1.3 | Consecutive low-IG questions |
| T5 | Multi-modal | FM-2.2 | Skewed mid-game ambiguity |
| T6 | Prediction mismatch | FM-2.6 | Calibrated vs oracle mismatch |
| T7 | Late shock | FM-3.1 | Rare branch at end |
| T8 | Wrong verification | FM-3.3 | Incorrect guess with claim |

## Overlays

### Prediction (belief-report channel)
- `calibrated` - Argmax with calibrated confidence
- `overconfident` - Always 95% confidence
- `always_yes` - Always predicts YES
- `sticky` - Persists high-confidence predictions
- `commit_early` - Locks to MAP after threshold
- `refuses_revise` - Keeps wrong answer K turns

### Termination (action channel)
- `rational` - Guess when |S|=1
- `premature_stop` - Stop at high entropy
- `unaware` - Continue past termination point
- `wrong_guess` - Incorrect guess with verification claim

## Output Format

```json
{
  "trajectory_id": "abc123",
  "trajectory_type": "T8",
  "target_mast_modes": ["FM-3.3"],
  "overlay_tags": ["pred:calibrated_argmax", "term:wrong_guess", "verify:claim_present"],
  "secret": "Horse",
  "turns": [
    {
      "turn": 4,
      "question": "Is it used for decoration?",
      "answer": false,
      "entropy_before": 2.32,
      "entropy_after": 1.58,
      "model_action": "guess",
      "guess": {
        "secret": "Watermelon",
        "confidence": 0.9,
        "verification_claim": "The secret must be 'Watermelon' because..."
      },
      "guess_correct": false,
      "stop_accepted": false
    }
  ]
}
```

## Gate Validators

Each trajectory type has an automated gate validator:

```bash
python run.py validate outputs/ --show-failures
```

| Type | Gate Checks |
|------|-------------|
| T1 | All splits balanced |
| T2 | Early + late rare branches |
| T3 | Plateau + resolution |
| T4 | Consecutive low-IG |
| T5 | Skewed mid-game |
| T6 | (none) |
| T7 | Late rare branch |
| T8 | Wrong guess + verification_claim |

## Project Structure

```
├── run.py              # CLI entry point
├── data/               # CUQ dataset (not committed)
├── outputs/            # Generated trajectories
└── src/
    ├── models.py       # Trajectory dataclasses
    ├── loader.py       # Dataset loading
    ├── index.py        # Question indexing
    ├── bitmask.py      # 128-bit operations
    ├── validators.py   # Gate validators
    ├── generators/     # T1-T8 generators
    └── overlays/       # Prediction + termination
```
