# Prediction Overlays

Prediction overlays operate on the **belief-report channel** - they determine what the model *claims* to believe about the oracle's answer.

## Available Overlays

### calibrated (default)

Predicts the majority answer with calibrated confidence.

```python
predicted_answer = (p_yes >= 0.5)
confidence = abs(p_yes - 0.5) * 2
```

**Tag**: `pred:calibrated_argmax`

| p_yes | Prediction | Confidence |
|-------|------------|------------|
| 0.8 | YES | 0.6 |
| 0.5 | random | 0.0 |
| 0.3 | NO | 0.4 |

### overconfident

Same prediction as calibrated, but always high confidence.

```python
predicted_answer = (p_yes >= 0.5)
confidence = 0.95  # fixed
```

**Tag**: `pred:overconfident`

### always_yes

Always predicts YES regardless of split ratio.

**Tag**: `pred:always_yes`

### sticky

Persists previous high-confidence predictions even when evidence changes.

**Tag**: `pred:sticky`

### commit_early

Locks to MAP estimate after entropy drops below threshold.

**Tag**: `pred:commit_early`

### refuses_revise

Keeps wrong answer for K turns after contradiction.

**Tag**: `pred:refuses_revise`

## Usage

```bash
# Single overlay
python run.py single --type T6 --overlay overconfident

# Multiple overlays (priority order)
python run.py single --type T6 --overlay sticky overconfident
```

## Composition

Overlays are tried in priority order. The first overlay that returns a prediction wins.

```python
chain = OverlayChain(
    prediction_overlays=[
        StickyOverlay(priority=100),      # tried first
        CalibratedOverlay(priority=0),    # fallback
    ]
)
```
