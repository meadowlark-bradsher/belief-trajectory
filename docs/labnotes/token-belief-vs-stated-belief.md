# Belief–Action–Report Decoupling in LLMs

**Findings That Survived + Open Questions**

## Scope and framing

We studied how large language models express, act on, and report uncertainty under controlled multiple-choice (MC) probes. The core objective was to distinguish:

* **Behavioral belief**: implied by the model’s action distribution (token softmax / logprobs)
* **Reported belief**: explicit probability reports from the model
* **Interventions**: ordering and framing choices (Act, Report, CoT, Introspection)

We used:

* Multiple probe orderings (A/B/C/D and extensions)
* Anti-uniform reporting constraints
* Category stratification (Easy / Ambiguous / Adversarial)
* Both **sampling-based** and **true logprob-based** action distributions
* Cross-provider comparisons (OpenAI, Gemini, Ollama)

This document summarizes what has held up across variants and what remains unresolved.

---

## Robust Findings (Survived Across Conditions)

### F1. **Uniform probability reporting is a stable learned attractor**

**Claim**
Absent constraints, models overwhelmingly report uniform distributions (e.g., 25/25/25/25), regardless of how confident their actions are.

**Evidence**

* 100% uniform reports across probes (A/B/C/D) without constraint
* Holds across providers and prompt orderings
* JSON compliance does not break the attractor

**Method(s)**

* Probes A–D
* MC probability reporting
* Both sampling and logprob action estimation

**Interpretation**

* This is not ignorance or incapacity
* It is a *policy equilibrium* induced by training and evaluation pressure
* Self-report ≠ introspection by default

---

### F2. **A mild anti-uniform constraint reliably breaks the attractor**

**Claim**
Adding a gentle constraint (e.g., min 5%, max 80% unless logically certain) forces informative reports without inducing boundary gaming.

**Evidence**

* Uniform reports drop from ~100% → ~0–2%
* No piling at min/max bounds
* Report entropy drops modestly (≈1.85–1.95), not collapse

**Method(s)**

* Anti-uniform constraint on reporting
* JSON-only output
* Boundary hit tracking

**Interpretation**

* Models *can* express graded uncertainty
* The uniform attractor is optional, not fundamental

---

### F3. **Belief–action decoupling is real and largest when action is most confident (with true logprobs)**

**Claim**
With true logprobs, mismatch (JS divergence) is **highest when the action distribution is most peaked**.

**Evidence**

* OpenAI / Gemini: Corr(JS, H_action) ≈ −0.5
* Divergence increases as entropy decreases
* Sampling-based estimates previously masked this effect

**Method(s)**

* True logprobs over next-token MC options
* JS divergence between action vs report
* Correlation analysis

**Interpretation**

* Models often act decisively while reporting broad uncertainty
* This is classic belief–action decoupling, not noise

---

### F4. **Probe order matters; interventions have regime-dependent effects**

#### F4a. **Report→Act (Probe B) acts as a coupling/anchoring intervention**

**Claim**
Forcing belief declaration *before* action generally reduces mismatch.

**Evidence**

* Best or near-best mean JS across providers
* Effect persists even when report is imperfect

**Method(s)**

* Probe B
* Anti-uniform reporting

**Interpretation**

* Declared belief becomes an upstream latent variable
* Acts as a commitment or anchoring device

---

#### F4b. **Chain-of-Thought (CoT) is a variance amplifier on ambiguous inputs**

**Claim**
CoT improves alignment on easy questions but **worsens** alignment on ambiguous/adversarial ones.

**Evidence**

* Easy category: CoT lowest JS
* Ambiguous/adversarial: CoT highest JS, most extreme failures
* Near-zero Corr(JS, H_action) for CoT → structural mismatch

**Method(s)**

* Probe C
* Category-stratified analysis
* Extreme-case counts (JS > 0.5)

**Interpretation**

* CoT collapses narratives, not uncertainty
* When multiple plausible stories exist, CoT commits to one without propagating belief coherently

---

#### F4c. **Introspective framing stabilizes across heterogeneous tasks**

**Claim**
Introspective framing (Probe D) yields the most stable belief–action coupling on ambiguous/adversarial sets.

**Evidence**

* Lowest mean JS on ambiguous and adversarial categories
* Lower variance than CoT
* Best overall probe in multi-question runs

**Method(s)**

* Probe D
* Anti-uniform reporting
* Category analysis

**Interpretation**

* Introspection is not a neutral readout
* It *regularizes* behavior, dampening narrative collapse
* Useful as a stabilizer, not as “truth access”

---

### F5. **Sampling-based action estimation can invert conclusions**

**Claim**
Sampling with smoothing can reverse correlations and obscure true effects.

**Evidence**

* Ollama (sampling): Corr(JS, H_action) positive
* OpenAI/Gemini (logprobs): Corr(JS, H_action) negative
* Difference explained by smoothing bias on peaked distributions

**Method(s)**

* Side-by-side sampling vs logprob comparisons
* Correlation sign checks

**Interpretation**

* True logprobs are essential for confidence-related claims
* Sampling is acceptable for exploratory work but not final attribution

---

## Open Questions (Not Yet Settled)

### O1. **What exactly is the “reporting policy”?**

* Is it a distinct learned head?
* Is it a conditional generation mode triggered by probability language?
* How provider-specific is it?

**Next methods**

* Report→Act→Report loops
* Act→Report→Act₂ with flip analysis
* Cross-model fine-grained prompt ablations

---

### O2. **Can belief revision be *controlled* rather than just observed?**

* When allowed to “change your mind,” what causes *justified* flips vs confabulation?
* Does Report→Act₂ anchor, or does it sometimes mislead?

**Next methods**

* Act→Report→Act₂
* Act→CoT→Act₂
* Flip quality metrics (Δentropy, TV distance)

---

### O3. **Can a probe-selection policy generalize?**

* Simple heuristic: use CoT if “easy,” Introspect otherwise
* Does a model-internal “crispness” signal work without oracle labels?

**Next methods**

* Pre-act entropy thresholds
* Automatic probe routing
* Compare Always-C, Always-D, and Adaptive policies

---

### O4. **How does belief–action–report alignment interact with environment truth?**

(Current work only compares belief vs belief.)

**Next step**

* Reintroduce environments with known posteriors (e.g., 20Q)
* Measure:

  * action ↔ environment
  * report ↔ environment
  * whether stabilizing probes preserve correctness

---

### O5. **Provider-specific effects**

* Gemini’s higher residual uniform rate under constraint
* Differences in logprob sharpness
* Possible tokenization or decoding biases

**Next methods**

* Strict token canonicalization
* Per-provider calibration curves
* Excluding uniform cases vs treating as signal

---

## Summary (One-Paragraph Takeaway)

We find that LLMs exhibit a robust, learned decoupling between action confidence and reported uncertainty. Absent constraints, probability reporting collapses to a uniform “safe” policy regardless of behavior. A mild anti-uniform constraint reveals that models *can* express graded uncertainty, but belief–action alignment depends strongly on intervention order and task ambiguity. Chain-of-thought amplifies commitment and helps on easy problems, but destabilizes belief reporting on ambiguous ones; introspective framing acts as a stabilizer across regimes. True logprobs are essential: with them, divergence is highest precisely when models act most confidently. These results suggest belief expression, reasoning, and action are partially independent control surfaces—measurable, intervenable, and not interchangeable.

---

If you want, next we can:

* turn this into a 2–3 page technical note
* map each finding to a simple causal diagram
* or write a “methods appendix” that exactly specifies probes A–D (+ extensions) so this can be replicated cleanly
