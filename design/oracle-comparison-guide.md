# Oracle Model Comparison Script - Usage Guide

## Purpose

This script tests Gemini Flash, GPT-4o mini, and Claude Sonnet against your CUQ (T5-XL) oracle to determine which LLM produces the most compatible bitmasks for trajectory generation.

## What It Measures

### 1. Agreement with CUQ
- **Average agreement**: % of bits that match CUQ oracle
- **Range**: Min/max agreement across questions
- **Jaccard similarity**: Set overlap metric

**Good**: >85% average agreement
**Warning**: <80% average agreement

### 2. Split Ratio Distribution
Compares how questions split across buckets:
- **very_rare**: [0, 0.05) - Critical for T4 trajectories
- **rare**: [0.05, 0.15) - Needed for T2, T7
- **skewed**: [0.15, 0.35) - Needed for T5
- **balanced**: [0.35, 0.50] - Critical for T1, T6, T7, T8

**Your CUQ baseline**: 4.4% balanced (this is the bottleneck!)

### 3. Problematic Patterns
- **All YES (128/128)**: Question provides no information
- **All NO (0/128)**: Question provides no information
- **Useless rate**: % of questions that are uninformative

**Good**: <5% useless
**Warning**: 5-15% useless
**Critical**: >15% useless

### 4. Cost Analysis
- Per-test cost (200 questions, no batch)
- Per 1k questions (with batch API)
- Per 50k questions (for full generation)

### 5. Quality Score
Composite metric (0-1 scale) combining:
- 40%: Agreement with CUQ
- 20%: Balanced split availability
- 20%: Low useless question rate
- 20%: Jaccard similarity

Higher = better

## Implementation Steps

### Step 1: Prepare Test Data

You need three inputs:

**1. Items file** (already have: items.txt)
```
Airplane
Banana
Baseball
...
```

**2. Test questions file**
Create `test_questions.json`:
```json
[
  {"id": "q_001", "text": "Is it edible?"},
  {"id": "q_002", "text": "Is it living?"},
  ...
]
```

Sample 200-500 diverse questions from your CUQ dataset.

**3. CUQ bitmasks file**
Create `cuq_bitmasks.npy`:
```python
import numpy as np

# Load your CUQ bitmasks
cuq_bitmasks = {
    'q_001': np.array([True, True, False, ...]),  # 128 booleans
    'q_002': np.array([False, True, True, ...]),
    ...
}

# Save
np.save('cuq_bitmasks.npy', cuq_bitmasks)
```

### Step 2: Implement Data Loading

Edit `oracle_model_comparison.py`, find `load_test_data()` function:

```python
def load_test_data(items_file, questions_file, cuq_bitmasks_file):
    # Load items
    with open(items_file, 'r') as f:
        items = [line.strip() for line in f if line.strip()]
    
    # Load questions
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    # Load CUQ bitmasks
    cuq_bitmasks = np.load(cuq_bitmasks_file, allow_pickle=True).item()
    
    return items, questions, cuq_bitmasks
```

### Step 3: Implement Oracle Queries

Edit `query_oracle()` function for each model:

```python
def query_oracle(model_name, questions, items):
    bitmasks = {}
    
    if model_name == 'gemini_flash_2.0':
        # Use Google's Gemini API
        import google.generativeai as genai
        genai.configure(api_key='YOUR_API_KEY')
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        for q in questions:
            bitmask = np.zeros(len(items), dtype=bool)
            for i, item in enumerate(items):
                prompt = f"Question: '{q['text']}' Item: '{item}'. Answer only 'yes' or 'no'."
                response = model.generate_content(prompt)
                bitmask[i] = 'yes' in response.text.lower()
            bitmasks[q['id']] = bitmask
    
    elif model_name == 'gpt4o_mini':
        # Use OpenAI API
        from openai import OpenAI
        client = OpenAI()
        
        for q in questions:
            bitmask = np.zeros(len(items), dtype=bool)
            for i, item in enumerate(items):
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{
                        'role': 'user',
                        'content': f"Question: '{q['text']}' Item: '{item}'. Answer only 'yes' or 'no'."
                    }]
                )
                bitmask[i] = 'yes' in response.choices[0].message.content.lower()
            bitmasks[q['id']] = bitmask
    
    elif model_name == 'claude_sonnet_4':
        # Use Anthropic API
        import anthropic
        client = anthropic.Anthropic()
        
        for q in questions:
            bitmask = np.zeros(len(items), dtype=bool)
            for i, item in enumerate(items):
                response = client.messages.create(
                    model='claude-sonnet-4-20250514',
                    max_tokens=10,
                    messages=[{
                        'role': 'user',
                        'content': f"Question: '{q['text']}' Item: '{item}'. Answer only 'yes' or 'no'."
                    }]
                )
                bitmask[i] = 'yes' in response.content[0].text.lower()
            bitmasks[q['id']] = bitmask
    
    return bitmasks
```

### Step 4: Run Comparison

```bash
python oracle_model_comparison.py
```

### Step 5: Review Results

The script outputs:

**Console**:
- Agreement metrics for each model
- Split distribution comparison
- Problematic pattern detection
- Cost analysis
- Overall ranking
- Recommendation with reasoning

**File**: `oracle_comparison_results.json`
```json
{
  "recommendation": "claude_sonnet_4",
  "models": [
    {
      "model_name": "claude_sonnet_4",
      "agreement_with_cuq": 0.89,
      "balanced_pct": 5.2,
      "useless_pct": 2.1,
      "quality_score": 0.847,
      ...
    },
    ...
  ]
}
```

## Interpreting Results

### Scenario 1: Clear Winner

```
Model              Quality Score  Assessment
claude_sonnet_4    0.850          ✓✓✓ BEST CHOICE
gpt4o_mini         0.720          ✓ GOOD - Acceptable
gemini_flash_2.0   0.650          ⚠️ SUBOPTIMAL
```

**Action**: Use claude_sonnet_4

### Scenario 2: Close Race

```
Model              Quality Score  Assessment
gpt4o_mini         0.845          ✓✓✓ BEST CHOICE
claude_sonnet_4    0.840          ✓✓ EXCELLENT - Very close
```

**Action**: Consider cost - if GPT-4o mini is much cheaper and only 0.5% behind, use it

### Scenario 3: High Useless Rate

```
gemini_flash_2.0:
  Agreement: 84.2%
  Useless questions: 15.3% ⚠️
  
⚠️ ACTION REQUIRED:
   15.3% of questions return all-YES or all-NO
   1. Refine prompt: Add 'Answer NO unless clearly YES'
   2. Filter post-generation
   3. Consider alternative model
```

**Action**: 
1. Try prompt refinement first
2. Re-run test with refined prompt
3. If still high, switch models

### Scenario 4: Low Balanced Split Coverage

```
Model              Balanced Splits
CUQ Baseline       4.4%
gemini_flash_2.0   2.1% ⚠️

⚠️ WARNING:
   Only 2.1% balanced splits (CUQ: 4.4%)
   May cause issues with T1/T6/T7/T8 trajectories
```

**Action**: 
- Generate 2x more questions to compensate
- Or switch to model with better balanced coverage

## Expected Costs

For 200-question test (no batch):

| Model | Est. Cost |
|-------|-----------|
| Gemini Flash 2.0 | $0.22 |
| GPT-4o mini | $0.38 |
| Claude Sonnet 4 | $7.68 |

**Total for all three: ~$8.30**

## Prompt Refinement Examples

If a model shows high useless rate, test these prompts:

**Conservative prompt**:
```
Question: {question}
Item: {item}

Answer "yes" ONLY if this property clearly and literally applies to the item.
When in doubt, answer "no".
Respond with only "yes" or "no".
```

**Few-shot prompt**:
```
Question: "Can be used for art?"
Item: "Hammer"
Correct: "no" (hammer is not primarily for art)

Question: "Can be used for art?"
Item: "Paintbrush"
Correct: "yes" (designed for art)

Question: {question}
Item: {item}
Answer: 
```

## Next Steps After Comparison

### If Results are Good (>85% agreement, <5% useless)

1. Use recommended model
2. Proceed with full generation (50-100k questions)
3. Apply same consistency filtering you used with CUQ

### If Results Need Improvement (80-85% agreement, 5-15% useless)

1. Test prompt refinements
2. Re-run comparison with new prompts
3. Consider model with better scores

### If Results are Poor (<80% agreement, >15% useless)

1. Stick with CUQ oracle
2. Or invest in extensive prompt engineering
3. Or accept that LLM oracle != CUQ oracle and rebuild trajectories from scratch

## Common Issues

**Issue**: "All models show >50% useless questions"
**Cause**: Prompt is too permissive
**Fix**: Add "Answer NO unless clearly YES" instruction

**Issue**: "Agreement is 60% across all models"
**Cause**: CUQ oracle world is very different from LLM world
**Fix**: Either stick with CUQ or accept you're building a new oracle world

**Issue**: "Script crashes on API rate limits"
**Cause**: Too many requests too fast
**Fix**: Add rate limiting or use batch API

**Issue**: "Costs are way higher than estimated"
**Cause**: Responses are longer than 10 tokens
**Fix**: Add "Respond with only yes or no" to prompt

## Files Generated

1. `oracle_comparison_results.json` - Full results
2. Console output - Human-readable comparison
3. Recommendation with reasoning

## Support

If you encounter issues:
1. Check that test data loads correctly
2. Verify API keys are set
3. Test with just 10 questions first
4. Check rate limits for each API

The script is designed to be conservative - it will tell you if models have issues rather than silently accepting poor quality.
