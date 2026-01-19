# Question Coverage Analysis for Trajectory Generation

This toolkit analyzes your existing question set to determine:
1. How questions are distributed across split ratio buckets
2. Whether you have sufficient coverage for all trajectory types (T1-T8)
3. How many questions you need to regenerate with a new oracle
4. Cost estimates for oracle regeneration

## Quick Start

### Step 1: Test with Synthetic Data

First, verify the pipeline works:

```bash
python test_synthetic.py
```

This will:
- Generate 2000 synthetic questions
- Run the complete analysis
- Show you what the output looks like
- Save results to `test_coverage_analysis.json`

### Step 2: Implement Your Data Loader

Edit `run_analysis.py` and implement the `load_questions()` function.

Your function should return: `List[Tuple[str, np.ndarray]]`

Where each tuple is:
- `question_id` (str): Unique identifier (e.g., "q_12345")
- `bitmask` (np.ndarray): Boolean array of shape (128,)
  - `True` = item answers "yes" to this question
  - `False` = item answers "no" to this question

**Example implementations:**

```python
# If you have a matrix where rows = questions, cols = items
def load_questions():
    bitmask_matrix = load_your_bitmask_matrix()  # shape (num_questions, 128)
    question_ids = load_your_question_ids()  # list of str
    
    return [
        (qid, bitmask_matrix[i].astype(bool))
        for i, qid in enumerate(question_ids)
    ]

# If you have a dictionary
def load_questions():
    question_dict = load_your_dict()  # {question_id: bitmask}
    return list(question_dict.items())

# If you compute bitmasks on-the-fly from items
def load_questions():
    items = load_items()  # your 128 items
    questions = load_question_list()
    
    results = []
    for q_id, question_text in questions:
        bitmask = np.array([
            item_matches_question(item, question_text)
            for item in items
        ], dtype=bool)
        results.append((q_id, bitmask))
    
    return results
```

### Step 3: Run Analysis on Your Data

```bash
python run_analysis.py
```

This will output:
- Distribution of questions across split ratio buckets
- Coverage analysis per trajectory type
- Any coverage gaps
- Recommended sample size for regeneration
- Probability of sufficient coverage for different sample sizes
- Optimal sample size for 95% confidence
- Cost estimates for different models

## Understanding the Output

### Split Ratio Buckets

Questions are classified into 4 buckets based on split ratio (min(yes, no) / total):

**Note:** Split ratio is symmetric - a 25% yes and 75% yes both give split_ratio=0.25

| Bucket | Range | Used For |
|--------|-------|----------|
| very_rare | [0.00, 0.05) | T4 redundant loops, T2/T7 rare branches |
| rare | [0.05, 0.15) | T2/T7 rare branches, transitions |
| skewed | [0.15, 0.35) | T5 multi-modal |
| balanced | [0.35, 0.50] | T1 baseline, general use |

### Trajectory Requirements

Each trajectory type has minimum requirements:

- **T1 (Smooth halving)**: 15 balanced questions
- **T2 (Early collapse)**: 2 rare + 1 very_rare + 10 balanced
- **T3 (Plateau)**: 5 very_rare + 8 balanced
- **T4 (Redundant loop)**: 7 very_rare
- **T5 (Multi-modal)**: 6 skewed + 5 balanced
- **T6 (Prediction mismatch)**: 10 balanced
- **T7 (Late shock)**: 8 balanced + 2 rare
- **T8 (Wrong verification)**: 10 balanced

### Sample Size Recommendations

The analysis provides:

1. **Recommended sample size**: Conservative estimate (3-5x redundancy)
2. **Coverage probability**: Monte Carlo simulation showing probability of meeting all requirements
3. **Optimal size**: Minimum questions needed for 95% confidence

### Cost Estimates

Based on optimal sample size, shows costs for:
- Gemini Flash 1.5 (cheapest)
- Gemini Flash 2.0
- GPT-4o mini
- Sonnet 4.5 (highest quality)

## Key Insights

### Why You Need Redundancy

Even if you only need ~200 questions minimum across all trajectory types, you need 3-5x that number because:

1. **Unknown distribution**: You don't know split ratios until after generating
2. **Rare buckets**: Very rare splits (<5%) occur infrequently
3. **Safety margin**: Want high confidence (>95%) of coverage

### The Math Problem

This is a variant of the **Coupon Collector Problem**:
- You have 4 buckets (split ratio ranges)
- Each generated question falls into one bucket randomly
- You need ≥k questions in each specific bucket
- What's P(success) for N generated questions?

The analyzer solves this via Monte Carlo simulation using your empirical distribution.

## Output Files

- `coverage_analysis.json`: Full analysis results
- Console output: Formatted tables and recommendations

## Next Steps After Analysis

Based on the recommended sample size and cost:

1. **If cost is acceptable**: Generate that many questions with new oracle
2. **If cost is too high**: 
   - Consider keeping some existing questions
   - Use cheaper model (Gemini Flash)
   - Accept lower coverage probability

3. **If coverage gaps exist**: Current question set insufficient for some trajectories

## Questions?

The key question this answers: **How many questions must I generate to be 95% confident I can construct all 8 trajectory types?**

The answer depends on:
- Current distribution of split ratios
- Minimum requirements per trajectory type
- Desired confidence level
