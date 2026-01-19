# Question Coverage Analysis Toolkit - File Index

## Files You Need to Edit

### `run_analysis.py` (4.5K)
**Action Required:** Implement the `load_questions()` function

This is the main entry point. You need to write code to load your 122,913 questions
into the format: `List[Tuple[str, np.ndarray]]` where each tuple is (question_id, bitmask).

Once implemented, run: `python run_analysis.py`

## Core Analysis Engine

### `question_coverage_analysis.py` (14K)
**No editing needed** - The main analysis engine

Contains:
- `QuestionCoverageAnalyzer` - Main analysis class
- `CoverageAnalysis` - Results dataclass
- Monte Carlo simulation for coverage probability
- Optimal sample size finder
- All bucket definitions and trajectory requirements

## Testing & Validation

### `test_synthetic.py` (4.2K)
**No editing needed** - Validation script

Run: `python test_synthetic.py`

Generates 2000 synthetic questions and runs the full analysis pipeline.
Use this to verify everything works before implementing your real data loader.

Expected output:
- Analysis of 2000 questions across 4 buckets
- Coverage probability estimates
- Saves `test_coverage_analysis.json`

## Documentation

### `README.md` (5.3K)
**Reference** - Quick start guide and API documentation

Covers:
- Installation and setup
- Step-by-step workflow
- Understanding the output
- Bucket definitions
- Trajectory requirements
- Cost estimates

### `SUMMARY.md` (5.4K)
**Reference** - Conceptual overview

Explains:
- The problem being solved
- Why you need this analysis
- Key insights (split ratio symmetry)
- Decision tree for regeneration
- Technical details
- What questions this answers

### `FILE_INDEX.md` (this file)
**Reference** - What each file does

## Generated Output

### `test_coverage_analysis.json` (959 bytes)
**Test output** - Results from running `test_synthetic.py`

### `coverage_analysis.json`
**Will be created** - Results from running `run_analysis.py` on your real data

Contains:
```json
{
  "total_questions": 122913,
  "split_ratio_histogram": {...},
  "trajectory_requirements": {...},
  "recommended_sample_size": 1500,
  "redundancy_factor": 81.94,
  "coverage_gaps": []
}
```

## Typical Workflow

```
1. python test_synthetic.py          # Verify pipeline works
   ↓
2. Edit run_analysis.py              # Implement load_questions()
   ↓
3. python run_analysis.py            # Analyze your 122k questions
   ↓
4. Read coverage_analysis.json       # Get optimal sample size
   ↓
5. Decide on regeneration strategy   # Based on cost and coverage
```

## Dependencies

```python
import numpy as np           # Array operations
from dataclasses import dataclass  # Data structures
from typing import List, Dict, Tuple, Callable  # Type hints
from collections import defaultdict  # Counting
import json                  # Save results
```

All standard library except `numpy`.

## File Sizes

- **Total toolkit**: ~33 KB (very lightweight)
- **No large dependencies**: Just numpy
- **Fast execution**: <10 seconds for 122k questions

## What to Commit to Git

If sharing this analysis:
```
✓ question_coverage_analysis.py
✓ run_analysis.py (with your load_questions implemented)
✓ test_synthetic.py
✓ README.md
✓ SUMMARY.md
✗ test_coverage_analysis.json (regenerable)
✓ coverage_analysis.json (your results)
```

## Troubleshooting

**Error: "Implement your data loading logic"**
→ You need to edit `run_analysis.py` and implement `load_questions()`

**Error: Shape mismatch**
→ Each bitmask should be shape (128,) boolean array

**Error: ModuleNotFoundError: numpy**
→ Run: `pip install numpy`

**Unexpected coverage gaps**
→ Run `test_synthetic.py` first to verify code works
→ Then check your `load_questions()` implementation

**Split ratios look wrong**
→ Verify: `split_ratio = min(yes_count, no_count) / total`
→ Should be between 0.0 and 0.5

## Quick Reference

**To test:** `python test_synthetic.py`
**To analyze:** `python run_analysis.py`
**To understand buckets:** See README.md
**To understand problem:** See SUMMARY.md
**To see what file does what:** This file
