# Trajectory Construction Notes

## Scope

This document explains:

* How our 20Q questions and secrets were derived
* How 20Q questions are converted into **bitmasks** over a fixed secret set.
* How bitmasks define a **deterministic posterior / feasible set** update.
* How we compute **environment epistemic uncertainty** and **calibration targets** from masks.

---

## Secrets

We use a fixed universe of **N = 128** secrets.

This is a reduction from an earlier 158-secret setting, originally derived from LMRL_Gym, to simplify downstream math and indexing:

* Entropy bounds become \(\log_2 128 = 7\) bits.
* Masks fit cleanly in power-of-two representations.
* Coverage binning and gating logic is simpler.

Secrets are assigned stable integer IDs \(\{1,\dots,128\}\).

---

## Question sources and canonicalization

Question text originates from the **LMRL-Gym** 20Q question corpus.

To reduce redundancy from paraphrases / near-duplicates, we canonicalize questions using embedding-based clustering and select a **medoid** (most central question) per cluster.

Operationally:

* Compute embeddings for questions.
* Cluster questions in embedding space.
* Choose the medoid question per cluster as the canonical representative.

(Implementation lives in `find_question_medoids.py`.)

---

## Oracle-derived bitmasks

For each canonical question \(q\) and each secret \(s_i \in \Omega\) (\(|\Omega|=128\)), we query an oracle for a binary answer:

\[
a(q,s_i) \in \{0,1\},\quad 1=\text{YES},\;0=\text{NO}.
\]

We stack these answers into a per-question bitmask:

\[
m_q = [a(q,s_1), a(q,s_2), \dots, a(q,s_{128})]^\top \in \{0,1\}^{128}.
\]

Masks are stored as bitsets for fast intersection and popcount.

---

## Posterior update via bitmask algebra

We represent the environment's belief state as a **feasible set** over secrets rather than a free-form probability distribution.

### Feasible-set (posterior) state

Let \(S_t \in \{0,1\}^{128}\) be the feasible-set mask after turn \(t\), with \(S_0=\mathbf{1}\) (all secrets initially feasible).

When we ask question \(q_t\) and observe oracle answer \(y_t \in \{\text{YES},\text{NO}\}\), we update deterministically:

* If \(y_t=\text{YES}\):
  \[
  S_{t+1} = S_t \wedge m_{q_t}
  \]

* If \(y_t=\text{NO}\):
  \[
  S_{t+1} = S_t \wedge \neg m_{q_t}
  \]

This update is equivalent to Bayesian filtering under a uniform prior over \(S_t\) and a deterministic observation model: secrets inconsistent with the observation are eliminated.

---

## Derived quantities for calibration

From \(S_t\) we compute:

* **Feasible set size:** \(|S_t| = \|S_t\|_1\)

* **Environment epistemic entropy proxy:**
  \[
  H^{\text{env}}_t = \log_2 |S_t|
  \]

* **Environment probability of YES for a candidate question \(q\):**
  \[
  p^{\text{env}}_t(\text{YES}\mid q) = \frac{\|S_t \wedge m_q\|_1}{\|S_t\|_1}
  \]

These quantities provide ground truth targets for:

* next-answer probability calibration (match \(\hat p\) to \(p^{\text{env}}\))
* state uncertainty calibration (match \(\hat H\) to \(H^{\text{env}}\))

---

## Trajectory construction (world + overlays)

We synthesize belief trajectories by selecting question sequences whose induced split ratios / information gain profiles match desired archetypes (T1–T8).

### World layer

The world layer consists of:

* question IDs (and text)
* oracle answers
* feasible-set updates \(S_t\)
* derived metrics (\(|S_t|\), \(H_t\), split ratios)

World generation is validated by per-type gates (e.g., plateau→resolution, late shock).

### Overlay layer

Overlays decorate the world trace to create explicit, auditable behaviors:

* **Prediction overlay:** model predicts next oracle answer + confidence.
* **Termination overlay:** model_action in {continue, guess, stop} with stop_accepted semantics.
* **Verification overlay:** explicit verification_claim fields for FM-3.3-style incorrect verification events.

Overlays are tagged (e.g., `overlay_tags`) to enable automated audits.

---

## Oracle Choice

The bitmask for each question requires an **oracle** to answer whether each of the 128 secrets satisfies the question. Oracle choice affects both the semantic quality and consistency of the resulting trajectories.

### Original CUQ Oracle (T5-XL)

The original CUQ (Calibrated Uncertainty Quantification) dataset from LMRL-Gym used **T5-XL** as the oracle model. This encoder-decoder model was queried for each (question, item) pair to produce binary YES/NO answers.

Characteristics:

* Reasonable semantic accuracy for common-sense questions
* Some inconsistencies for edge-case items or ambiguous phrasings
* ~122K questions with precomputed bitmasks

### GPT-4o-mini Oracle

For trajectory generation requiring higher oracle fidelity, we provide an alternative oracle based on **GPT-4o-mini** via OpenAI's Batch API.

Regeneration procedure:

1. Sample questions from the canonical set
2. For each question, query GPT-4o-mini for all 128 items in a single batched prompt
3. Parse YES/NO responses and construct the 128-bit bitmask
4. Store results in `questions_gpt4o_mini.jsonl`

Prompt template (simplified):

```
For each item, answer YES if the item satisfies the question, NO otherwise.
Reply with ONLY numbered answers (1. YES or 1. NO), one per line.

Question: {question}

Items:
1. {item_1}
2. {item_2}
...
128. {item_128}
```

Configuration:

* `temperature=0` for deterministic responses
* Batch API for cost efficiency (~50% discount vs. synchronous)

### Gemini Oracle (Alternative)

For users without OpenAI API access, we also support **Gemini 2.0 Flash** as an oracle. The implementation follows the same batched-prompt strategy.

See `new_oracle_project/generate_bitmasks.py` for the Gemini implementation.

### Oracle Agreement

Empirically, GPT-4o-mini and Gemini achieve **~85-95% agreement** with the original T5-XL oracle on a per-bit basis. Disagreements typically occur on:

* Ambiguous category boundaries (e.g., "Is it a tool?" for multi-use objects)
* Regional or cultural knowledge differences
* Edge-case items that are uncommon or polysemous

For trajectory stress-testing purposes, **internal consistency** (same oracle throughout a trajectory) matters more than absolute accuracy.

### Choosing an Oracle

| Use Case | Recommended Oracle |
|----------|-------------------|
| Reproducing LMRL-Gym baselines | T5-XL (original `questions.jsonl`) |
| Higher semantic fidelity | GPT-4o-mini (`questions_gpt4o_mini.jsonl`) |
| Cost-sensitive / API diversity | Gemini 2.0 Flash |

The trajectory generator defaults to the original CUQ bitmasks but can be configured to use alternative oracle files.

---

## References

1. **LMRL-Gym**: Abdulhai, M., White, I., Snell, C., Sun, C., Hong, J., Zhai, Y., Xu, K., & Levine, S. (2023). *LMRL-Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models*. arXiv:2311.18232. [https://lmrl-gym.github.io/](https://lmrl-gym.github.io/)

2. **20 Questions Task**: The 20 Questions environment in LMRL-Gym tests strategic information gathering through yes/no questions. Our secret set and question corpus derive from this benchmark.

3. **GPT-4o-mini**: OpenAI (2024). GPT-4o mini model card. [https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)

4. **Gemini**: Google DeepMind (2024). Gemini 2.0 Flash. [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)
