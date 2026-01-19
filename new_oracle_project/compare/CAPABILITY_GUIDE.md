# Capability-Based Oracle Selection Guide

## The Paradigm Shift

### What I Built Initially (WRONG)
❌ "Do you have 4.4% balanced questions globally?"
❌ "Can you generate ONE trajectory of each type?"
❌ Simple bucket counting

### What You Actually Need (RIGHT)
✓ "Do you have balanced questions at |S|=16? At |S|=32?"
✓ "Can you generate 2,000 trajectories of each type reliably?"
✓ **Conditional capability testing at runtime feasible set sizes**

## The Core Insight

**Global distribution is misleading.**

Example question: "Is it edible?"
- Global: 50/128 items edible → 39% yes rate → "skewed" bucket
- At |S|=16 (late game): Maybe 12/16 are edible → 75% yes rate → "extreme_high" bucket
- **Same question, different capability depending on |S|**

## What the Capability Analyzer Does

### 1. Conditional Coverage Analysis

For each |S| ∈ {128, 64, 32, 16, 8, 4, 2}:
1. Sample 100 random feasible sets of size |S|
2. For each question, compute p(q|S) = |S ∩ m_q| / |S|
3. Average across samples
4. Classify into bins:
   - **no_op** [0.99, 1.0]: Zero IG (for T4/T8 degenerate segments)
   - **near_no_op** [0.95, 0.99]: Near-zero IG (for T3/T4 plateaus)
   - **extreme_high** [0.80, 0.95]: Shock candidates (for T2/T7)
   - **balanced** [0.40, 0.60]: Good splits (for T1/T6)
   - etc.

**Output**: "At |S|=16, you have 450 balanced questions, 120 no-op questions, etc."

### 2. Trajectory Feasibility Assessment

For each trajectory type (T1-T8):
1. Define phases with |S| ranges and bin needs
   - Example T1: needs 5 balanced at |S|≈96, 4 balanced at |S|≈48, etc.
2. Check if pool has enough questions in each phase
3. Compute pass rate (product of phase success probabilities)
4. Estimate: "Need ~3,500 attempts to generate 2,000 T4 trajectories"

**Output**: Pass rates, bottlenecks, production feasibility

### 3. Per-Secret Separability

For each secret:
- Count YES questions (include it)
- Count NO questions (exclude it)
- Ensure no secret is "impossible to isolate"

**Output**: Separability ranges, problematic secrets

### 4. Gap Identification

Automatically identifies:
- "T4 needs 1,050 near_no_op at |S|≈64, only have 320"
- "T7 needs 100 extreme_low at |S|≈12, only have 45"

**Output**: What to generate more of

## Example Output

```
TRAJECTORY GENERATION FEASIBILITY
Type   Pass Rate    Attempts for 2k  Status
T1     68.3%        2,929           ✓✓ Good
T2     45.2%        4,425           ✓✓ Good
T3     32.1%        6,231           ✓ Workable
T4     12.4%        16,129          ⚠️ Problematic
T5     51.8%        3,861           ✓✓ Good
T6     72.1%        2,774           ✓✓✓ Excellent
T7     38.7%        5,168           ✓✓ Good
T8     28.3%        7,067           ✓ Workable

⚠️ CRITICAL COVERAGE GAPS
  • T4: near_no_op at |S|≈64: have 320, need 1050
  • T4: no_op at |S|≈48: have 180, need 700

📋 RECOMMENDED GENERATION TARGETS
  → Generate more near_no_op questions
  → Generate more no_op questions
```

## How This Changes Oracle Selection

### Old Workflow
1. Test Gemini/GPT-4o/Sonnet on 200 questions
2. Measure CUQ agreement
3. Pick winner
4. Generate 50k questions
5. Hope it works

### New Workflow
1. Test each oracle on 1000 questions
2. **Run capability analyzer** on each
3. Compare:
   - Pool quality scores
   - Pass rates per trajectory type
   - Coverage gaps
   - Production costs (attempts × API cost)
4. Pick oracle with:
   - Best trajectory feasibility (not CUQ agreement)
   - Fewest critical gaps
   - Acceptable production cost
5. **Targeted generation** to fill gaps
6. Re-analyze capability
7. Generate trajectories

## Oracle Comparison Priorities (Revised)

### Priority 1: Trajectory Feasibility (40%)
- Can you generate 2k+ of each type?
- What are the pass rates?
- **This is make-or-break**

### Priority 2: Coverage Completeness (30%)
- Do you have all needed bins at all needed |S| sizes?
- Number of critical gaps
- **Fewer gaps = less rework**

### Priority 3: Self-Consistency (20%)
- Are answers deterministic?
- **Affects training data quality**

### Priority 4: Cost (10%)
- API cost × attempts needed
- **Secondary to functionality**

## Targeted Question Generation

Once you know the gaps, generate questions INTO bins:

### For "near_no_op" at |S|≈64

**Strategy**: Generate questions that are extreme globally
1. Sample candidate texts like:
   - "Is it a vehicle?" (should be ~10-20% yes)
   - "Is it food?" (should be ~30% yes)
2. Query oracle → get bitmask
3. Compute global yes rate
4. **Accept if**: 0.10 < yes_rate < 0.30 (will be near_no_op at |S|=64)

### For "balanced" at |S|≈32

**Strategy**: Generate questions with ~50% global yes rate
1. Sample candidate texts
2. Query oracle
3. **Accept if**: 0.45 < yes_rate < 0.55

### For "no_op" questions (surgical use)

**Strategy**: Find truly degenerate questions
1. Generate questions like "Is it made of matter?"
2. Query oracle
3. **Accept if**: yes_rate > 0.98 or yes_rate < 0.02

**Budget**: Only need ~100-200 of these total (T4/T8 use 2-6 per trajectory)

## Production Workflow

### Phase 1: Test Oracles (~$16)
```bash
# For each oracle (Gemini, GPT-4o, Sonnet)
python oracle_functional_comparison.py --model gemini_flash_2.0 --test-size 1000

# Outputs:
# - 1000 questions with bitmasks
# - Self-consistency score
```

### Phase 2: Capability Analysis (~$0, local compute)
```bash
# For each oracle's 1000 test questions
python question_capability_analyzer.py --questions gemini_test.json

# Outputs:
# - Conditional coverage at all |S| sizes
# - Trajectory feasibility (pass rates)
# - Coverage gaps
# - Recommended targets
```

### Phase 3: Oracle Selection

Compare capability reports:

**Gemini**:
- Pool quality: 0.72
- T4 pass rate: 12% (problematic)
- Gaps: near_no_op, no_op
- Cost for 2k per type: ~$65

**GPT-4o mini**:
- Pool quality: 0.84
- T4 pass rate: 35% (good)
- Gaps: few
- Cost for 2k per type: ~$85

**Sonnet**:
- Pool quality: 0.89
- T4 pass rate: 42% (excellent)
- Gaps: minimal
- Cost for 2k per type: ~$320

**Decision matrix**:
```
If T4 critical & budget >$300: Use Sonnet
If balanced needs & budget ~$100: Use GPT-4o mini
If cost-constrained but willing to do targeted gen: Use Gemini + fill gaps
```

### Phase 4: Targeted Generation

If using Gemini (has gaps):
1. Generate 500 near_no_op questions (target yes_rate 0.10-0.25)
2. Generate 200 no_op questions (target yes_rate >0.98)
3. Re-analyze capability
4. Verify T4 pass rate improved

### Phase 5: Production Generation

Generate final pool:
- Base size from capability analysis
- Plus oversampling for diversity
- Filter & validate

## Key Metrics to Track

### Pool Quality Score
```
Quality = 0.40 × avg_pass_rate
        + 0.40 × min_pass_rate  
        + 0.20 × coverage_completeness
```

**Thresholds**:
- >0.85: ✓✓✓ Excellent - ready for production
- 0.70-0.85: ✓✓ Good - minor gaps acceptable
- 0.50-0.70: ✓ Workable - targeted generation needed
- <0.50: ⚠️ Poor - reconsider oracle choice

### Pass Rates

**Per trajectory type**:
- >50%: Fast generation (3-4k attempts for 2k trajectories)
- 20-50%: Moderate (4-10k attempts)
- 10-20%: Slow (10-20k attempts)
- <10%: Problematic (>20k attempts or infeasible)

**Production budgets**:
- 2k trajectories × 8 types = 16k trajectories total
- If avg pass rate = 40% → need ~40k attempts
- At $1.50 per attempt (API cost) → $60k total
- **This is why pass rates matter**

## What About Zero-IG Questions?

**They're not waste - they're surgical tools:**

From your document:
- T4: 3-6 zero-IG turns for step repetition
- T8: 1-2 zero-IG turns for false verification
- T5: 1-3 zero-IG turns for menu failures

**Capability analyzer treats them correctly:**
- Counts them in "no_op" bin
- Tests availability at relevant |S| sizes
- Ensures you have enough (but not too many)

**Ideal distribution**:
```
no_op (zero-IG): ~5-10% of pool
  → ~500-1000 questions if pool is 10k
  → Enough for T4/T5/T8 degenerate segments
  → Not so many they dominate dataset
```

## Example Capability Report Interpretation

```
CONDITIONAL COVERAGE AT KEY SIZES
Bin             |S|=64    |S|=32    |S|=16    |S|=8     
no_op           145       178       203       245      
near_no_op      287       312       334       356      
extreme_low     423       456       478       489      
balanced        1,245     1,187     1,098     892      
extreme_high    401       445       467       482      
```

**Analysis**:
- ✓ Balanced questions available at all sizes (good for T1/T6)
- ✓ no_op increasing at smaller |S| (good for late-game T4/T8)
- ✓ Extreme splits available (good for T2/T7 shocks)
- **This pool has good shape**

**Bad example**:
```
Bin             |S|=64    |S|=32    |S|=16    |S|=8     
no_op           12        15        18        22       ⚠️ Too few
balanced        45        38        31        21       ⚠️ Declining
```
→ Can't generate T4 (needs many no_op)
→ T1 will struggle at smaller |S| (needs balanced throughout)

## Bottom Line

**Old question**: "Does this match CUQ?"

**New question**: "Can I generate 2,000 trajectories of each type with <5,000 attempts each?"

The capability analyzer answers the question you actually care about.

---

## Quick Start

1. Test oracles on 1000 questions each (~$16 total)
2. Run capability analyzer on each test set
3. Compare pool quality scores and pass rates
4. Pick oracle or decide to do targeted generation
5. Generate final pool
6. Validate with capability analyzer
7. Generate trajectories

**Total time**: ~1 week
**Total cost**: $16 test + $50-300 production
**Result**: High-confidence production-ready question pool
