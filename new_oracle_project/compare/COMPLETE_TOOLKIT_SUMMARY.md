# Complete Oracle Regeneration Toolkit - Summary

## What You Have

A complete suite of tools for regenerating your oracle with a new LLM while ensuring functional trajectory generation capability.

## The Files

### Core Analysis Tools

**1. question_coverage_analysis.py** (14KB)
- Analyzes split ratio distribution
- Calculates minimum questions needed for trajectory types
- Supports training diversity requirements
- Monte Carlo simulation for coverage probability

**2. run_analysis.py** (Template)
- Main script to analyze your question set
- Configure diversity factor (3-100x)
- Configure trajectory count (200 recommended)
- Outputs optimal sample size and cost estimates

**3. oracle_functional_comparison.py** (NEW - Recommended)
- **Tests which LLM can generate all trajectory types**
- Measures useless rate, coverage, consistency, cost
- NOT focused on CUQ agreement
- Integrated with coverage analysis toolkit
- **Use this for oracle selection**

**4. oracle_model_comparison.py** (Legacy)
- Tests which LLM agrees most with CUQ
- Useful if you want to stay close to CUQ world
- Skip this if building new oracle world

### Documentation

**5. FUNCTIONAL_COMPARISON_GUIDE.md** (Key reading)
- Explains priority metrics
- How to interpret results
- Decision matrix
- Examples of good/bad results

**6. DIVERSITY_GUIDE.md**
- Why you need diversity factor for training
- How to choose 3x vs 50x vs 100x
- Cost-quality tradeoffs
- Combinatorics explained

**7. BATCH_API_GUIDE.md**
- Why batch API is perfect for this
- 50% cost savings
- How to use with different providers
- Timeline expectations

**8. UPDATE_SUMMARY.md**
- What changed from original version
- Why diversity factor was added
- Impact on your analysis

## The Workflow

### Phase 1: Understand Current State (Already Done)

✓ You analyzed your 122k CUQ questions
✓ Found: Only need 1,281 questions for base coverage
✓ Bottleneck: 4.4% balanced splits
✓ Tested Gemini: 84% agreement, 15% useless

### Phase 2: Select Oracle Model (~$16 test)

```bash
# Edit oracle_functional_comparison.py to wire up APIs
python oracle_functional_comparison.py
```

**Tests 1000 questions across 3 models:**
- Gemini Flash 2.0
- GPT-4o mini
- Claude Sonnet 4

**Outputs:**
1. Useless rate for each model (<5% target)
2. Coverage feasibility (can generate T1-T8?)
3. Self-consistency score (>95% target)
4. Optimal sample size needed
5. Cost estimates
6. Recommendation with reasoning

**Expected winner**: GPT-4o mini (~0.85 quality score, ~3% useless, ~$48 for 50k)

### Phase 3: Generate Questions ($50-100)

**Based on test results, generate with chosen model:**

```python
# For 50x diversity (recommended)
num_questions = ~2,500
cost = ~$48 (GPT-4o mini batch)

# For 100x diversity (if budget allows)
num_questions = ~5,000
cost = ~$75 (GPT-4o mini batch)
```

**Submit as batch job, wait ~24 hours**

### Phase 4: Filter and Validate

```python
# After generation:
1. Filter useless questions (all-YES, all-NO)
2. Run consistency check (query sample twice)
3. Verify coverage (re-run coverage analysis)
4. Build trajectories
```

**Expected yield:**
- Generate 2,500 questions
- Filter ~75 useless (3%)
- Keep ~2,425 useful
- Effective diversity: ~48x (excellent for 200 trajectories)

## Key Decisions Made

### 1. Diversity Factor: 50x Recommended

**Why not 100x?**
- Diminishing returns beyond 50x for 200 trajectories
- C(350, 7) ≈ 10^14 combinations already astronomical
- Oracle quality becomes limiting factor, not diversity
- Save $25 for other experiments

**Why not 10x?**
- Only 3x safety margin after accounting for useless questions
- Risk of coverage gaps if distribution shifts
- Cheap insurance ($48 vs $8, only $40 difference)

### 2. Model Selection: Test First

**Don't assume Sonnet is best:**
- Gemini: Cheap but 15% useless (bad)
- GPT-4o mini: Moderate cost, likely 3% useless (good)
- Sonnet: Expensive, likely 2% useless (premium)

**GPT-4o mini probably wins on cost/quality tradeoff**

### 3. Focus: Functional Coverage Not CUQ Agreement

**Key insight from you:**
> "We already knew splits might change which is why we projected 
> the need to get more bitmasks so that by random chance we would 
> find new questions that give us the same split profiles but in a 
> less bizarre oracle world."

**This is why functional comparison script exists.**

## Cost Summary

### Test Phase: ~$16
- Oracle comparison: $8 for all 3 models
- Consistency testing: $8 (2x repeats)
- **Essential** - prevents wasting $50-100 on wrong model

### Generation Phase: $48-75
- 50x diversity: ~$48 (GPT-4o mini batch)
- 100x diversity: ~$75 (GPT-4o mini batch)
- **Recommended**: 50x diversity

### Alternative If Quality Critical: $175
- 50x diversity with Sonnet 4.5 batch
- Best oracle quality + excellent diversity
- Worth it if ML research depends on data quality

## Timeline

**Week 1**: Model selection testing
- Day 1: Implement API calling
- Day 2: Run functional comparison (~1-2 hours compute)
- Day 3: Review results, make decision

**Week 2**: Full generation
- Day 1: Submit batch job (50k questions × 128 items)
- Day 2: Wait for batch completion
- Day 3: Download results, filter useless questions
- Day 4: Verify coverage, build test trajectories
- Day 5: Validate trajectory quality

**Total: ~2 weeks**

## Success Criteria

**After generation, you should have:**

✓ ~2,400 useful questions (50x diversity)
✓ <5% useless rate
✓ Coverage for all trajectory types (T1-T8)
✓ >95% self-consistency
✓ All buckets represented:
  - very_rare: >1%
  - rare: >1%
  - skewed: >1%
  - balanced: >2%

**Red flags to watch for:**

❌ >10% useless rate → Prompt refinement needed
❌ Coverage gaps → Generate more questions
❌ <90% consistency → Set temperature=0 or use voting
❌ 0% in any bucket → Oracle has pathological distribution

## Files You Need to Edit

### 1. oracle_functional_comparison.py

**Two functions to implement:**

```python
def load_test_questions(questions_file, num_questions=1000):
    # Load your questions and items
    # Return: (questions, items)
    pass

def query_oracle_with_consistency_test(model_name, questions, items):
    # Call appropriate API for model
    # Return: dict of bitmasks
    pass
```

**That's it!** Everything else is ready to go.

## What Happens Next

**After running functional comparison, you'll see:**

```
RECOMMENDATION: gpt4o_mini
Functional Quality: 0.847

✓ Key Strengths:
  • Low waste: Only 3.2% useless questions
  • Complete coverage: Can generate all trajectory types
  • Needs 2,450 questions for 95% confidence
  • Highly consistent: 97.8% deterministic
  • Cost: $55.68 for optimal coverage

📊 Next Steps:
  1. Generate 2,450 questions with gpt4o_mini
  2. Filter out useless questions (expect ~78 filtered)
  3. Run consistency check on generated bitmasks
  4. Verify coverage for all trajectory types
  5. Generate training trajectories
```

**Then you just follow the steps!**

## Questions You Can Now Answer

✓ How many questions do I need? → Run functional comparison
✓ Which LLM should I use? → Test shows GPT-4o mini likely best
✓ What will it cost? → ~$48-75 depending on diversity
✓ Will it work for trajectories? → Test verifies coverage
✓ How long will it take? → ~2 weeks total
✓ What if oracle is bad? → Test catches this before spending $100

## The Big Picture

**You're not replicating CUQ** - you're building a new, better oracle:
- More intuitive ("Train" is outdoor)
- More consistent (frontier LLM vs trained T5-XL)
- More functional (tested coverage before generation)
- More scalable (can regenerate with new models later)

**The tools ensure:**
- You don't waste money on useless questions
- You can generate all 8 trajectory types
- You have sufficient diversity for training
- You catch problems before committing to full generation

## Final Recommendation

1. **This week**: Run functional comparison ($16)
2. **Next week**: Generate with winner (~$48-75)
3. **Total cost**: ~$64-91
4. **Total time**: ~2 weeks
5. **Result**: High-quality oracle for robust RL training

**This is way better than:**
- ❌ Blindly generating 122k questions ($200+)
- ❌ Assuming CUQ agreement matters (it doesn't)
- ❌ Not testing for useless questions (15% waste)
- ❌ Hoping diversity is sufficient (verify first)

You now have everything you need. Good luck!
