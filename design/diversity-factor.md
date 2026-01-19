# Understanding Diversity Factor for Training

## The Problem

You need to train a model on trajectories, which means you need many examples:
- **Training set**: 150 trajectories per type
- **Validation set**: 25 trajectories per type  
- **Test set**: 25 trajectories per type
- **Total**: 200 trajectories per type × 8 types = 1,600 trajectories

**But how many questions do you need to generate 1,600 diverse trajectories?**

## Naive Calculation (WRONG)

```
T4 trajectory requires 7 very_rare questions
200 T4 trajectories × 7 questions = 1,400 very_rare questions needed
```

**This is wrong because:**
1. Trajectories can **share questions**
2. Different trajectories can use **overlapping subsets**
3. You don't need complete uniqueness, just **diversity**

## Correct Approach: Diversity Factor

**Diversity factor** determines how many unique questions you want available for sampling diverse trajectory subsets.

### Formula

```
Required questions = base_requirement × diversity_factor
```

For T4 (needs 7 very_rare):
- 1.0x diversity: 7 questions (minimal)
- 3.0x diversity: 21 questions (standard)
- 5.0x diversity: 35 questions (preferred)
- 10.0x diversity: 70 questions (maximum)

## What Each Factor Means

### 1.0x Diversity (NOT RECOMMENDED for training)

**Available**: 7 very_rare questions

**200 T4 Trajectories look like:**
```
Trajectory 1: [A, B, C, D, E, F, G]
Trajectory 2: [A, B, C, D, E, F, G]  ← identical
Trajectory 3: [A, B, C, D, E, F, G]  ← identical
...
Trajectory 200: [A, B, C, D, E, F, G]  ← identical
```

**Result**: Model overfits to these specific 7 questions. Learns the questions, not the failure pattern.

### 3.0x Diversity (MINIMUM for training)

**Available**: 21 very_rare questions

**200 T4 Trajectories look like:**
```
Trajectory 1: [A, B, C, D, E, F, G]
Trajectory 2: [B, C, D, E, F, G, H]  ← 6/7 overlap
Trajectory 3: [C, D, E, F, G, H, I]  ← different but related
Trajectory 4: [D, E, F, G, H, I, J]
...
Trajectory 200: [O, P, Q, R, S, T, U]  ← mostly different
```

**Result**: Some diversity. Moderate overlap is okay. Model sees multiple question patterns.

**Combinatorics**: C(21, 7) = 116,280 possible subsets >> 200 trajectories needed

### 5.0x Diversity (PREFERRED for training)

**Available**: 35 very_rare questions

**200 T4 Trajectories look like:**
```
Trajectory 1: [A, B, C, D, E, F, G]
Trajectory 2: [H, I, J, K, L, M, N]  ← completely different
Trajectory 3: [O, P, Q, R, S, T, U]  ← completely different
...
Trajectory 200: [Mix of 35 available questions with good variety]
```

**Result**: High diversity. Model learns the failure pattern, not specific questions.

**Combinatorics**: C(35, 7) = 6,724,520 possible subsets >>> 200 trajectories needed

### 10.0x Diversity (MAXIMUM)

**Available**: 70 very_rare questions

**Result**: Extreme diversity but expensive. Probably overkill unless you need >10,000 trajectories.

**Combinatorics**: C(70, 7) ≈ 548 million subsets

## How to Choose

### Decision Matrix

| Your Use Case | Recommended Factor | Rationale |
|---------------|-------------------|-----------|
| Single trajectory generation | 1.0x | No training, just need one example |
| Proof of concept (10-50 trajectories) | 2.0x | Small dataset, some variety |
| Small training set (50-100 per type) | 3.0x | Minimum for generalization |
| Standard training (100-300 per type) | 3.0-5.0x | Good balance cost/quality |
| Large training (300-1000 per type) | 5.0x | Need robust diversity |
| Massive training (1000+ per type) | 5.0-10.0x | High diversity required |

### Cost-Quality Tradeoff

Higher diversity = More questions = Higher cost but better model

**Example for T4 (7 very_rare base):**

| Factor | Questions | Unique Subsets | Cost @$2/1000q | Quality |
|--------|-----------|----------------|----------------|---------|
| 1.0x | 7 | 1 | $0.02 | ⚠️ Poor |
| 3.0x | 21 | 116k | $0.05 | ✓ Acceptable |
| 5.0x | 35 | 6.7M | $0.09 | ✓✓ Good |
| 10.0x | 70 | 548M | $0.18 | ✓✓✓ Excellent |

(This is just for T4; total cost depends on all trajectory types)

## Interaction with Trajectory Count

If you need **more trajectories**, you might need higher diversity:

```
50 trajectories × 3.0x diversity = Good
200 trajectories × 3.0x diversity = Acceptable  
500 trajectories × 3.0x diversity = Risky (marginal)
500 trajectories × 5.0x diversity = Good

1000 trajectories × 5.0x diversity = Good
1000 trajectories × 10.0x diversity = Better
```

**Rule of thumb**: If `num_trajectories × diversity_factor < 100`, you probably have enough variety.

For 200 trajectories:
- 3.0x: 200 × 3 = 600 (marginal)
- 5.0x: 200 × 5 = 1000 (good)

## Practical Recommendations

### For Your Bridge Experiment (200 trajectories per type):

**Minimum acceptable**: 3.0x diversity
- Total questions needed: ~1,000-2,000
- Cost with Gemini Flash: ~$3-6
- Quality: Acceptable but not ideal

**Recommended**: 5.0x diversity
- Total questions needed: ~1,500-3,000
- Cost with Gemini Flash: ~$5-10
- Quality: Good generalization

**Overkill**: 10.0x diversity
- Total questions needed: ~3,000-5,000
- Cost with Gemini Flash: ~$10-15
- Quality: Excellent but diminishing returns

### Start Conservative, Scale Up

1. **First experiment**: 3.0x diversity, 200 trajectories per type
2. **Analyze results**: Does model overfit? Are trajectories too similar?
3. **If needed**: Regenerate with 5.0x diversity
4. **Cost difference**: Only a few dollars more

## Configuration in Code

```python
# In run_analysis.py, adjust these lines:

NUM_TRAJECTORIES_PER_TYPE = 200  # train + val + test
DIVERSITY_FACTOR = 3.0  # 3.0 minimum, 5.0 preferred

# The analyzer will calculate:
# T4 very_rare requirement = 7 × 3.0 = 21 questions
# T1 balanced requirement = 15 × 3.0 = 45 questions
# etc.
```

## FAQ

**Q: Why not just generate 10,000 questions and be safe?**
A: Cost! At $2-10 per 1000 questions, 10k = $20-100. Plus processing time.

**Q: Can I start with 1.0x and add more later?**
A: Yes, but you'll need to regenerate trajectories. Better to do it right once.

**Q: What if I use different factors per trajectory type?**
A: Advanced. T4 might need 5.0x (rare questions critical) while T1 needs 3.0x (balanced questions abundant).

**Q: How do I know if my diversity is sufficient?**
A: After training, check:
- Does model generalize to held-out questions?
- Are validation trajectories sufficiently different from training?
- Plot question usage histogram - should be relatively uniform

**Q: Is there a mathematical optimal factor?**
A: Depends on your probability distribution. The tool estimates this via Monte Carlo.

## Bottom Line

**For the Bridge Experiment with 200 trajectories per type:**

- **Use 3.0x minimum** (~1,500 questions, ~$5)
- **Prefer 5.0x** (~2,500 questions, ~$8)
- **Don't use 1.0x** (catastrophic overfitting)
- **10.0x is overkill** unless you need >1000 trajectories

The tool will calculate exact requirements based on your question distribution.
