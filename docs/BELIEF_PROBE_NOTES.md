# Belief Probe Baseline Notes

**Date:** 2026-01-18

## Overview

Built a belief probe system to measure divergence between behavioral belief (token distribution at decision point) and declarative belief (self-reported probabilities) under four probe orderings.

## Probes

| Probe | Order | Purpose |
|-------|-------|---------|
| A | Act → Report | Post-hoc honesty |
| B | Report → Act | Belief anchoring |
| C | CoT → Act | Reasoning propagation |
| D | Act → Introspect | Retrospective report |

## Key Findings

### 1. Uniform Report Attractor

Without constraints, models output uniform `{"A":25,"B":25,"C":25,"D":25}` regardless of action distribution. This is a learned "safe" policy under RLHF.

**Fix:** Anti-uniform constraint (5-80% per option) breaks the attractor without inducing boundary gaming.

### 2. CoT Effectiveness is Question-Dependent

| Category | Best Probe | CoT (C) Performance |
|----------|------------|---------------------|
| Easy | C (CoT) | JS=0.072 — best |
| Ambiguous | D (Introspect) | JS=0.334 — worst |
| Adversarial | D (Introspect) | JS=0.316 — worst |

CoT acts as a **variance amplifier**: good when there's a crisp answer, bad when multiple answers are valid.

### 3. Logprobs vs Sampling Show Opposite Correlations

| Method | Corr(JS, H_token) | Meaning |
|--------|-------------------|---------|
| Sampling (Ollama) | +0.33 | JS↑ when uncertain |
| Logprobs (OpenAI/Gemini) | -0.50 | JS↑ when confident |

With true logprobs, divergence is highest when the model is most confident — revealing "confident action, hedged report" pattern.

### 4. Probe B Wins Across Providers

Report-first (B) consistently shows lowest JS divergence, confirming the anchoring effect works across model families.

## Files

```
belief_probe_baseline.py   # Main probe harness (Ollama)
providers.py               # Provider-agnostic interface
compare_providers.py       # Cross-provider comparison
generate_mc_questions.py   # Question generator
mc_questions.json          # 40 test questions
experiment2_results.json   # Ollama multi-question results
provider_comparison.json   # OpenAI/Gemini comparison
```

## Next Steps (Not Yet Implemented)

1. **Experiment 1:** Sweep constraint bounds [1%,95%], [5%,80%], [10%,70%]
2. **Experiment 3:** Add Probe E (Report → CoT → Act)
3. **Calibration metrics:** Report accuracy vs action agreement
4. **20Q integration:** Use probes at trajectory inflection points

## Usage

```bash
# Single probe run with anti-uniform
python belief_probe_baseline.py llama3.1:latest --anti-uniform -n 50

# Multi-question experiment
python belief_probe_baseline.py --experiment2 --anti-uniform -o results.json

# Cross-provider comparison
python compare_providers.py --providers openai gemini --quick
```
