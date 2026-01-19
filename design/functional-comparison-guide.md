# Functional Oracle Comparison Guide

## Key Paradigm Shift

### OLD APPROACH (oracle_model_comparison.py)
❌ "Which model agrees most with CUQ?"
❌ Measures CUQ similarity as primary metric
❌ Assumes we want to replicate CUQ's world

### NEW APPROACH (oracle_functional_comparison.py)
✓ "Which model can generate all 8 trajectory types?"
✓ Measures functional coverage as primary metric
✓ Accepts we're building a NEW oracle world

## Why This Matters

**CUQ world**: "Train" is indoor (weird but consistent)
**LLM world**: "Train" is outdoor (intuitive and consistent)

**We don't care which world - we just need functional coverage!**

## Priority Metrics (in order)

### 1. Useless Question Rate (<5% CRITICAL)

**What it measures**: % of questions that return all-YES or all-NO (no information)

**Why it matters**: Direct waste of generation budget

**Example**:
- Gemini: 15% useless → Generate 50k, only 42.5k useful → Waste $10
- GPT-4o mini: 3% useless → Generate 50k, 48.5k useful → Waste $2
- Claude Sonnet: 2% useless → Generate 50k, 49k useful → Waste $3

**Thresholds**:
- <5%: ✓✓✓ Excellent
- 5-10%: ✓ Acceptable
- 10-15%: ⚠️ Concerning (but workable)
- >15%: ❌ Disqualifying

**Your Gemini result (15%)**: Concerning but not disqualifying if other metrics are good

### 2. Coverage Feasibility (Binary: YES/NO)

**What it measures**: Can you generate all trajectory types (T1-T8) with this oracle?

**How it's tested**:
1. Generate 1000 questions with oracle
2. Run coverage analysis on those bitmasks
3. Check: Do you have enough very_rare, rare, skewed, balanced?
4. Report: Coverage gaps (if any)

**Why it matters**: If you can't generate T4 (needs very_rare), the oracle is useless

**Thresholds**:
- No gaps: ✓✓✓ Perfect
- 1-2 gaps: ⚠️ Workable (generate more questions)
- 3+ gaps: ❌ Problematic distribution

**Example bad result**:
```
Coverage gaps:
  • very_rare: have 15, need 21
  • balanced: have 30, need 45
```
→ Need to generate 1.5x more questions

**Example good result**:
```
✓ All trajectory requirements satisfied
Optimal sample size: 1,281 questions
```
→ Ready to proceed

### 3. Self-Consistency (>95% target)

**What it measures**: If you ask the same question twice, do you get the same answer?

**How it's tested**:
1. Sample 50 questions
2. Query oracle 2-3 times for each
3. Measure: % of question-item pairs with consistent answers

**Why it matters**: Inconsistent oracle means noisy training data

**Thresholds**:
- >95%: ✓✓✓ Highly deterministic
- 90-95%: ✓✓ Good (mostly consistent)
- 85-90%: ✓ Acceptable (some variance)
- <85%: ⚠️ Problematic (set temperature=0 or use voting)

**Example**:
- Question: "Is it edible?"
- Item: "Banana"
- Query 1: YES
- Query 2: YES
- Query 3: YES
- → 100% consistent ✓

**If inconsistent (70%)**: Oracle has high sampling variance → need deterministic generation

### 4. Distribution Shape (Not absolute values!)

**What it measures**: Does oracle have SOME of each bucket?

**What we DON'T care about**:
- ❌ CUQ has 4.4% balanced, oracle has 8% → IRRELEVANT
- ❌ CUQ has 20% very_rare, oracle has 5% → IRRELEVANT

**What we DO care about**:
- ✓ Oracle has >1% very_rare (can generate T4)
- ✓ Oracle has >2% balanced (can generate T1, T6, T7, T8)
- ✓ Oracle has >1% rare (can generate T2, T7)
- ✓ Oracle has >1% skewed (can generate T5)

**Pathological distributions**:
- 0% balanced → Can't generate T1/T6/T7/T8 ❌
- 90% very_rare → Hard to find balanced questions ⚠️
- 100% all-YES → Oracle is broken ❌

**Example good result**:
```
very_rare: 8.2%   ✓ (>1%)
rare: 12.1%       ✓ (>1%)
skewed: 35.4%     ✓ (>1%)
balanced: 44.3%   ✓ (>2%)
```
→ Has all buckets, good shape

**Example bad result**:
```
very_rare: 0.1%   ⚠️ (too few)
balanced: 0.8%    ⚠️ (too few)
```
→ Will struggle with T4 and T1 trajectories

### 5. Cost per Useful Question

**What it measures**: $/1000 questions after accounting for waste

**Formula**: `cost_per_1k / (1 - useless_rate)`

**Why it matters**: True cost accounting

**Example**:
- Gemini: $1.41/1k raw, 15% useless → $1.66/1k useful (+18% hidden cost)
- GPT-4o mini: $1.92/1k raw, 3% useless → $1.98/1k useful (+3% hidden cost)
- Claude Sonnet: $3.50/1k raw, 2% useless → $3.57/1k useful (+2% hidden cost)

**This is why useless rate matters**: High waste increases effective cost

## Functional Quality Score

Composite metric (0-1 scale):

```
Quality = 0.30 × low_useless_rate
        + 0.30 × coverage_feasible
        + 0.20 × self_consistency
        + 0.10 × has_balanced
        + 0.10 × has_very_rare
```

**Interpretation**:
- >0.85: ✓✓✓ Excellent for trajectory generation
- 0.75-0.85: ✓✓ Good, will work well
- 0.65-0.75: ✓ Acceptable, may need adjustments
- 0.50-0.65: ⚠️ Marginal, consider alternatives
- <0.50: ❌ Poor, not recommended

## Expected Results

Based on your preliminary findings:

| Model | Useless Rate | Coverage | Consistency | Quality | Verdict |
|-------|--------------|----------|-------------|---------|---------|
| Gemini Flash | 15% | ? | ? | ~0.70 | ⚠️ Marginal |
| GPT-4o mini | 3%* | ? | ? | ~0.85 | ✓✓ Likely best |
| Claude Sonnet | 2%* | ? | ? | ~0.88 | ✓✓✓ Premium |

*Estimated - needs testing

## How to Interpret Output

### Example Output 1: Clear Winner

```
RECOMMENDATION: gpt4o_mini
Functional Quality: 0.847

✓ Key Strengths:
  • Low waste: Only 3.2% useless questions
  • Complete coverage: Can generate all trajectory types
  • Needs 1,450 questions for 95% confidence
  • Highly consistent: 97.8% deterministic
  • Cost: $55.68 for optimal coverage
```

**Action**: Use GPT-4o mini, generate 1,450 questions

### Example Output 2: High Useless Rate

```
RECOMMENDATION: gemini_flash_2.0
Functional Quality: 0.723

⚠️ Watch Out For:
  • 15.3% useless questions (~$8.55 wasted)
    → Consider prompt refinement or post-generation filtering
  
📋 Examples of Problematic Questions:
  • q_044: 'can be used to sculpt' (128/128 YES)
  • q_014: 'able to be used for art' (128/128 YES)
  • q_030: 'relates to historical figure' (91/128 YES)
```

**Action**: 
1. Refine prompt: "Answer NO unless property clearly and literally applies"
2. Re-test with refined prompt
3. If still >10% useless, switch to GPT-4o mini

### Example Output 3: Coverage Gaps

```
RECOMMENDATION: claude_sonnet_4
Functional Quality: 0.812

⚠️ Watch Out For:
  • Coverage gaps detected:
      balanced: have 18, need 45
      very_rare: have 15, need 21
    → May need to generate more questions or bias toward missing buckets
```

**Action**:
1. Generate 2x more questions (2,820 instead of 1,410)
2. Or bias generation toward balanced/very_rare questions
3. Re-run coverage analysis to verify

### Example Output 4: Low Consistency

```
RECOMMENDATION: gemini_flash_2.0
Functional Quality: 0.759

⚠️ Watch Out For:
  • Moderate consistency: 87.4%
    → May need to set temperature=0 or use majority voting
```

**Action**:
1. Set temperature=0 in API calls
2. Or use best-of-3 voting for each question
3. Re-test consistency

## Comparison to Old Script

### Old Script (oracle_model_comparison.py)

**Primary metric**: Agreement with CUQ (84%)
**Secondary**: Split distribution match to CUQ
**Output**: "Gemini has 84% agreement with CUQ"
**Conclusion**: ❓ Is 84% good enough? Unclear.

### New Script (oracle_functional_comparison.py)

**Primary metric**: Useless rate (15%)
**Secondary**: Coverage feasibility (YES/NO)
**Output**: "Gemini wastes 15% of budget but covers all trajectory types"
**Conclusion**: ✓ Actionable - reduce waste or accept cost

## When to Use Each Script

### Use OLD script if:
- You want to stay close to CUQ oracle world
- You have specific reasons to preserve CUQ's world model
- You're debugging CUQ oracle issues

### Use NEW script if:
- You're building a new oracle (most cases)
- You care about trajectory generation capability
- You want to minimize wasted generation budget
- You're deciding between multiple LLM oracles

## Implementation Checklist

- [ ] Prepare 1000 test questions (diverse sample from your full set)
- [ ] Implement `load_test_questions()` function
- [ ] Implement `query_oracle_with_consistency_test()` for each model
- [ ] Run script: `python oracle_functional_comparison.py`
- [ ] Review useless rate (target: <5%)
- [ ] Check coverage feasibility (target: no gaps)
- [ ] Verify self-consistency (target: >95%)
- [ ] Compare costs
- [ ] Make decision
- [ ] If needed: refine prompts and re-test
- [ ] Generate full dataset with chosen model

## Cost Estimate

**Test run (1000 questions × 3 models)**:
- Gemini Flash: $0.22
- GPT-4o mini: $0.38
- Claude Sonnet: $7.68
- **Total: ~$8.30**

Plus 2x for consistency testing = **~$16.60 total**

Worth it to avoid wasting $50-100 on wrong model!

## Quick Decision Matrix

| Useless Rate | Coverage | Consistency | → Decision |
|--------------|----------|-------------|------------|
| <5% | ✓ | >95% | ✓✓✓ USE IT |
| <5% | ✓ | 85-95% | ✓✓ Use with temp=0 |
| <10% | ✓ | >95% | ✓ Acceptable |
| 10-15% | ✓ | >95% | ⚠️ Try prompt refinement first |
| >15% | Any | Any | ❌ Switch models |
| Any | ✗ | Any | ❌ Switch models |
| <5% | ✓ | <85% | ⚠️ Need voting/temperature fix |

## Bottom Line

**Old question**: "Does this match CUQ?"
**New question**: "Can I build trajectories with this?"

The new script answers the question you actually care about.
