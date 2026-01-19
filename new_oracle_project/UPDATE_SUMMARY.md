# Coverage Analysis Toolkit - Update Summary

## Changes Made

### 1. Added Training Parameters Support

**Problem**: Original tool only calculated requirements for generating a single trajectory of each type. Didn't account for needing 100-1000+ diverse training examples.

**Solution**: Added configurable parameters:
- `num_trajectories_per_type`: How many examples you need (default: 200)
- `diversity_factor`: How much variety in questions (1.0-10.0x, recommended: 3.0-5.0x)
- `include_training_requirements`: Scale requirements for training vs single trajectory

### 2. Core Algorithm Updates

**`QuestionCoverageAnalyzer.__init__()`**
- Now accepts training parameters
- Defines base requirements (single trajectory)
- Scales to training requirements using diversity factor
- Formula: `required = base × diversity_factor`

**Example**: T4 needs 7 very_rare questions
- 1.0x (single): 7 questions
- 3.0x (training min): 21 questions
- 5.0x (training preferred): 35 questions

### 3. Enhanced Output

**Analysis now shows**:
- Training parameters configuration
- Side-by-side comparison of base vs scaled requirements
- Diversity factor comparison table
- Warnings if diversity too low (<3.0x)

**Example output**:
```
TRAINING PARAMETERS:
  Trajectories per type: 200
  Diversity factor: 3.0x

Trajectory   Base (1x)              Scaled (3.0x)
T4           very_rare≥7            very_rare≥21
```

### 4. New Documentation

**DIVERSITY_GUIDE.md** (8KB)
- Comprehensive explanation of diversity factor
- Why you need it for training
- How to choose the right factor
- Cost-quality tradeoffs
- Practical recommendations
- Decision matrix by use case

### 5. Updated Test Suite

**test_synthetic.py**
- Now tests both training mode and single trajectory mode
- Shows comparison to illustrate difference
- Validates that 3.0x diversity needs ~3x more questions

### 6. Configuration in run_analysis.py

Users can now adjust at top of file:
```python
NUM_TRAJECTORIES_PER_TYPE = 200  # train + val + test
DIVERSITY_FACTOR = 3.0  # 3.0 minimum, 5.0 preferred
```

## Key Insights

### The Math

**Without diversity factor** (wrong):
- 200 T4 trajectories × 7 questions = 1,400 questions needed ❌

**With diversity factor** (correct):
- Base requirement: 7 questions
- Diversity factor: 3.0x
- Scaled requirement: 21 questions ✓
- Combinatorics: C(21, 7) = 116,280 possible subsets >> 200 trajectories

### Why This Matters

**1.0x diversity (no scaling)**:
- All 200 trajectories nearly identical
- Model learns the specific questions, not the failure pattern
- Catastrophic overfitting

**3.0x diversity (minimum)**:
- Moderate variety, some overlap okay
- Model sees multiple question patterns
- Acceptable generalization

**5.0x diversity (preferred)**:
- High variety, minimal overlap
- Model learns failure patterns robustly
- Good generalization

## Impact on Your Analysis

### Before (wrong estimation):
```
Optimal sample size: ~500 questions
Cost: ~$1-2 with Gemini Flash
```
**Problem**: Would generate identical trajectories!

### After (correct estimation):
```
With 3.0x diversity:
  Optimal sample size: ~1,500 questions
  Cost: ~$3-5 with Gemini Flash

With 5.0x diversity:
  Optimal sample size: ~2,500 questions
  Cost: ~$5-10 with Gemini Flash
```
**Result**: Diverse training set, good generalization!

## Validation

Tested on synthetic data (2000 questions, 128 items):

| Mode | Diversity | Min Required | Optimal Size | Redundancy |
|------|-----------|--------------|--------------|------------|
| Single trajectory | 1.0x | 30 | 90 | 66.67x |
| Training (200/type) | 3.0x | 90 | 270 | 22.22x |
| Training (200/type) | 5.0x | 150 | 450 | 13.33x |

## Files Modified

1. ✓ `question_coverage_analysis.py` - Core algorithm updates
2. ✓ `run_analysis.py` - Added configuration parameters
3. ✓ `test_synthetic.py` - Added training mode testing
4. ✓ `README.md` - Updated with training parameters
5. ✓ `SUMMARY.md` - Updated with diversity factor info

## Files Added

1. ✓ `DIVERSITY_GUIDE.md` - Comprehensive guide on diversity factor (8KB)
2. ✓ `UPDATE_SUMMARY.md` - This file

## How to Use

### Step 1: Configure Parameters

Edit `run_analysis.py`:
```python
NUM_TRAJECTORIES_PER_TYPE = 200  # Your need
DIVERSITY_FACTOR = 3.0  # 3.0 min, 5.0 preferred
```

### Step 2: Implement Data Loader

Implement `load_questions()` function in `run_analysis.py`

### Step 3: Run Analysis

```bash
python run_analysis.py
```

### Step 4: Review Results

You'll get:
- Recommended sample size for your diversity factor
- Cost estimates across different models
- Diversity factor comparison showing impact
- Coverage probabilities

### Step 5: Make Decision

Based on results:
- Cost acceptable? → Generate with recommended size
- Cost too high? → Reduce diversity factor (but not below 3.0x)
- Need more trajectories? → Increase diversity factor

## Recommendations for Bridge Experiment

**For 200 trajectories per type (train/val/test split):**

### Conservative (minimum viable):
- Diversity: 3.0x
- Questions: ~1,500
- Cost: ~$3-5 (Gemini Flash)
- Quality: Acceptable, some overfitting risk

### Recommended (balanced):
- Diversity: 5.0x
- Questions: ~2,500
- Cost: ~$5-10 (Gemini Flash)
- Quality: Good generalization

### Premium (if budget allows):
- Diversity: 5.0x
- Questions: ~2,500
- Model: Sonnet 4.5
- Cost: ~$120
- Quality: Excellent + highest oracle quality

## Common Questions

**Q: Why not use diversity_factor=1.0?**
A: All trajectories would be nearly identical. Model overfits to specific questions, not failure patterns.

**Q: Is 10.0x diversity necessary?**
A: Only if you need >1000 trajectories per type. For 200, 5.0x is sufficient.

**Q: Can I start low and regenerate later?**
A: Yes, but you'll need to regenerate all trajectories. Better to do it right once.

**Q: How much does diversity factor affect cost?**
A: Roughly linear. 5.0x diversity ≈ 5x more questions ≈ 5x cost compared to 1.0x.

**Q: What if my analysis shows I need 10,000 questions?**
A: Either:
- Reduce diversity factor (3.0x instead of 5.0x)
- Use cheaper model (Gemini Flash vs Sonnet)
- Generate fewer trajectories (100 instead of 200)
- Accept the cost if dataset quality is critical

## Next Steps

1. Read `DIVERSITY_GUIDE.md` for detailed explanation
2. Test with synthetic data: `python test_synthetic.py`
3. Configure parameters in `run_analysis.py`
4. Implement your data loader
5. Run analysis on your 122k questions
6. Review results and make decision
7. Generate questions with chosen model and sample size

## Validation Checklist

Before regenerating oracle:
- [ ] Analyzed your 122k questions with training parameters
- [ ] Chosen appropriate diversity factor (3.0-5.0x)
- [ ] Calculated optimal sample size for 95% confidence
- [ ] Reviewed cost estimates
- [ ] Selected oracle model (Gemini Flash / Sonnet / etc)
- [ ] Decided on number of trajectories per type
- [ ] Understood tradeoffs between diversity and cost

## Contact

If unexpected results:
1. Run `python test_synthetic.py` to verify code works
2. Check your `load_questions()` implementation
3. Review `DIVERSITY_GUIDE.md` for conceptual understanding
4. Verify `diversity_factor` is 3.0-5.0 for training

**Critical**: diversity_factor < 3.0 → overfitting risk!
