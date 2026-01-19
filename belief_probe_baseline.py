#!/usr/bin/env python3
"""
Belief Probe Baseline - Ollama Compatible (v2)

Measures divergence between:
- Behavioral belief: token distribution at decision point
- Declarative belief: model-reported probabilities

Under four probe orderings:
- A: Act → Report
- B: Report → Freeze → Act
- C: CoT → Freeze → Act
- D: Act → Report (introspective prompt)

Improvements in v2:
- Parse health logging
- Configurable n_samples
- Jensen-Shannon divergence
- Introspective report probe
"""

import json
import math
import re
import requests
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import argparse

# -------------------------
# Configuration
# -------------------------

OLLAMA_URL = "http://localhost:11434"
OPTIONS = ["A", "B", "C", "D"]

PROMPT_BASE = """You are answering a multiple choice question.

Question:
Which option is most plausible given incomplete information?

Options:
A) Alpha
B) Beta
C) Gamma
D) Delta"""

# -------------------------
# Math Utilities
# -------------------------

def entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    """KL(P || Q) - how much P diverges from Q."""
    return sum(
        p[k] * math.log2((p[k] + eps) / (q[k] + eps))
        for k in p
        if p[k] > 0
    )


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence - symmetric, bounded [0, 1]."""
    m = {k: (p[k] + q[k]) / 2 for k in p}
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total variation distance - max probability difference."""
    return sum(abs(p[k] - q[k]) for k in p) / 2


# -------------------------
# Ollama Interface
# -------------------------

class OllamaModel:
    def __init__(self, model_name: str = "llama3.2", n_samples: int = 50):
        self.model_name = model_name
        self.base_url = OLLAMA_URL
        self.n_samples = n_samples

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate text completion."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    def get_option_probs(self, prompt: str, temperature: float = 1.0) -> Tuple[Dict[str, float], List[str]]:
        """Get probability distribution over A, B, C, D via sampling.

        Returns (probs, raw_samples) for diagnostics.
        """
        counts = {opt: 0 for opt in OPTIONS}
        raw_samples = []

        for _ in range(self.n_samples):
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": temperature,
                    }
                }
            )
            text = response.json().get("response", "").strip()
            raw_samples.append(text)

            # Strict first-character parsing (Fix 1)
            # Only count if first non-whitespace char is A/B/C/D
            if text:
                first_char = text[0].upper()
                if first_char in OPTIONS:
                    counts[first_char] += 1
                # else: invalid sample, not counted

        # Add Laplace smoothing (add-1)
        total = sum(counts.values()) + len(OPTIONS)
        probs = {opt: (counts[opt] + 1) / total for opt in OPTIONS}

        return probs, raw_samples

    def self_report(
        self,
        prompt: str,
        introspective: bool = False,
        anti_uniform: bool = False,
    ) -> Tuple[Dict[str, float], str, bool]:
        """Ask model to report its probability distribution.

        Returns (probs, raw_response, parse_success).

        Args:
            introspective: Use introspective framing ("what governed your choice")
            anti_uniform: Add constraint forcing non-uniform distribution (5-80% per option)
        """
        # Anti-uniform constraint text (intervention probe)
        constraint = ""
        if anti_uniform:
            constraint = """
CONSTRAINT: You must allocate at least 5% and at most 80% to any option unless logically certain.
Do not use 25/25/25/25 — express your actual confidence distribution."""

        if introspective:
            report_prompt = prompt + f"""

Report the probabilities that governed your choice *immediately before you answered*.
What was your internal confidence for each option at the moment of decision?
{constraint}
Return EXACTLY this JSON format, no other text:
{{"A":25,"B":25,"C":25,"D":25}}"""
        else:
            report_prompt = prompt + f"""

Report your probability for each option as percentages that sum to 100.
{constraint}
Return EXACTLY this JSON format, no other text:
{{"A":25,"B":25,"C":25,"D":25}}"""

        raw_response = self.generate(report_prompt, max_tokens=100)
        probs, parse_success = parse_report(raw_response)

        return probs, raw_response, parse_success


def parse_report(text: str) -> Tuple[Dict[str, float], bool]:
    """Parse model's self-reported probabilities.

    Returns (probs, parse_success).

    Fix 2: Try JSON parsing first, then fall back to regex.
    """
    probs = {}
    parse_success = False

    # Try JSON parsing first (Fix 2)
    # Look for JSON object in the response
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Extract A, B, C, D values
            for opt in OPTIONS:
                if opt in data:
                    val = float(data[opt])
                    # Normalize: if values are 0-100, divide by 100
                    if val > 1:
                        val = val / 100.0
                    probs[opt] = val
            if len(probs) == 4:
                parse_success = True
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fall back to regex patterns if JSON failed
    if not parse_success:
        probs = {}
        for opt in OPTIONS:
            patterns = [
                rf"{opt}\s*[:\=]\s*(\d+(?:\.\d+)?)\s*%?",
                rf"{opt}\)\s*[:\=]?\s*(\d+(?:\.\d+)?)\s*%?",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    probs[opt] = float(match.group(1)) / 100.0
                    break

        # Check parse success for regex fallback
        parse_success = len(probs) >= 3  # At least 3 options parsed

    # Fill missing with uniform
    for opt in OPTIONS:
        if opt not in probs:
            probs[opt] = 0.25

    # Renormalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs, parse_success


# -------------------------
# Uniformity Pressure Metrics
# -------------------------

@dataclass
class UniformityPressure:
    """Metrics for analyzing report distribution under anti-uniform constraint."""
    min_enforced: float = 0.05
    max_enforced: float = 0.80
    binding: bool = False  # True if constraint was active

    # Distance metrics
    options_at_min: int = 0  # Count of options at exactly min (5%)
    options_at_max: int = 0  # Count of options at exactly max (80%)
    avg_distance_to_bounds: float = 0.0  # How far from 5%/80% edges on average

    # Gaming detection
    is_boundary_piling: bool = False  # Mass piled at exactly 5% or 80%
    is_uniform: bool = False  # All options within 2% of 25%


def compute_uniformity_pressure(
    probs: Dict[str, float],
    min_bound: float = 0.05,
    max_bound: float = 0.80,
    binding: bool = False,
) -> UniformityPressure:
    """Compute uniformity pressure metrics for a reported distribution."""
    tolerance = 0.02  # 2% tolerance for boundary detection

    options_at_min = sum(1 for p in probs.values() if abs(p - min_bound) < tolerance)
    options_at_max = sum(1 for p in probs.values() if abs(p - max_bound) < tolerance)

    # Distance to nearest bound
    distances = []
    for p in probs.values():
        dist_to_min = abs(p - min_bound)
        dist_to_max = abs(p - max_bound)
        distances.append(min(dist_to_min, dist_to_max))
    avg_distance = sum(distances) / len(distances) if distances else 0.0

    # Gaming detection: 3+ options at boundary
    is_boundary_piling = (options_at_min >= 3) or (options_at_max >= 2)

    # Uniform detection: all within 2% of 0.25
    is_uniform = all(abs(p - 0.25) < tolerance for p in probs.values())

    return UniformityPressure(
        min_enforced=min_bound,
        max_enforced=max_bound,
        binding=binding,
        options_at_min=options_at_min,
        options_at_max=options_at_max,
        avg_distance_to_bounds=avg_distance,
        is_boundary_piling=is_boundary_piling,
        is_uniform=is_uniform,
    )


# -------------------------
# Probe Results
# -------------------------

@dataclass
class ProbeResult:
    name: str
    token_probs: Dict[str, float]
    report_probs: Dict[str, float]
    token_entropy: float
    report_entropy: float
    kl_token_report: float
    kl_report_token: float
    js_divergence: float
    total_variation: float
    # Diagnostics
    report_parse_ok: bool = True
    report_raw: str = ""
    report_sum: float = 1.0
    token_samples: List[str] = field(default_factory=list)
    n_samples: int = 0
    # Uniformity pressure (anti-uniform intervention)
    uniformity_pressure: Optional[UniformityPressure] = None

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"Token probs:  {self._fmt_probs(self.token_probs)}",
            f"Report probs: {self._fmt_probs(self.report_probs)} {'[PARSE OK]' if self.report_parse_ok else '[PARSE FAILED]'}",
            f"Token H:  {self.token_entropy:.3f} bits | Report H: {self.report_entropy:.3f} bits",
            f"KL(T||R): {self.kl_token_report:.4f} | KL(R||T): {self.kl_report_token:.4f}",
            f"JS div:   {self.js_divergence:.4f} | TV dist:  {self.total_variation:.4f}",
            f"Samples:  n={self.n_samples}",
        ]
        # Add uniformity pressure if present
        if self.uniformity_pressure is not None:
            up = self.uniformity_pressure
            constraint_status = "ACTIVE" if up.binding else "inactive"
            uniform_flag = " [UNIFORM]" if up.is_uniform else ""
            gaming_flag = " [BOUNDARY PILING]" if up.is_boundary_piling else ""
            lines.append(
                f"Constraint: {constraint_status} | @min:{up.options_at_min} @max:{up.options_at_max} | "
                f"avg_dist:{up.avg_distance_to_bounds:.3f}{uniform_flag}{gaming_flag}"
            )
        return "\n".join(lines)

    def _fmt_probs(self, probs: Dict[str, float]) -> str:
        return " ".join(f"{k}:{v:.2f}" for k, v in sorted(probs.items()))


def analyze(
    name: str,
    token_probs: Dict[str, float],
    report_probs: Dict[str, float],
    report_parse_ok: bool = True,
    report_raw: str = "",
    token_samples: List[str] = None,
    n_samples: int = 0,
    anti_uniform: bool = False,
) -> ProbeResult:
    # Compute uniformity pressure metrics
    uniformity_pressure = compute_uniformity_pressure(
        report_probs,
        binding=anti_uniform,
    )

    return ProbeResult(
        name=name,
        token_probs=token_probs,
        report_probs=report_probs,
        token_entropy=entropy(token_probs),
        report_entropy=entropy(report_probs),
        kl_token_report=kl_divergence(token_probs, report_probs),
        kl_report_token=kl_divergence(report_probs, token_probs),
        js_divergence=js_divergence(token_probs, report_probs),
        total_variation=total_variation(token_probs, report_probs),
        report_parse_ok=report_parse_ok,
        report_raw=report_raw,
        report_sum=sum(report_probs.values()),
        token_samples=token_samples or [],
        n_samples=n_samples,
        uniformity_pressure=uniformity_pressure,
    )


# -------------------------
# Probes
# -------------------------

def probe_A_act_then_report(
    model: OllamaModel,
    temperature: float = 1.0,
    anti_uniform: bool = False,
    prompt_base: str = None,
) -> ProbeResult:
    """Probe A: Act first, then report belief."""
    base = prompt_base if prompt_base is not None else PROMPT_BASE
    prompt = base + "\n\nAnswer with A, B, C, or D:\n"
    token_probs, samples = model.get_option_probs(prompt, temperature=temperature)
    report_probs, report_raw, parse_ok = model.self_report(
        base, anti_uniform=anti_uniform
    )

    return analyze(
        "A: Act → Report",
        token_probs, report_probs,
        report_parse_ok=parse_ok,
        report_raw=report_raw,
        token_samples=samples,
        n_samples=model.n_samples,
        anti_uniform=anti_uniform,
    )


def probe_B_report_then_act(
    model: OllamaModel,
    temperature: float = 1.0,
    anti_uniform: bool = False,
    prompt_base: str = None,
) -> ProbeResult:
    """Probe B: Report belief first, then act."""
    base = prompt_base if prompt_base is not None else PROMPT_BASE
    report_probs, report_raw, parse_ok = model.self_report(
        base, anti_uniform=anti_uniform
    )

    prompt = base + """

--- BELIEF RECORDED ---
Now answer with A, B, C, or D:
"""
    token_probs, samples = model.get_option_probs(prompt, temperature=temperature)

    return analyze(
        "B: Report → Act",
        token_probs, report_probs,
        report_parse_ok=parse_ok,
        report_raw=report_raw,
        token_samples=samples,
        n_samples=model.n_samples,
        anti_uniform=anti_uniform,
    )


def probe_C_cot_then_act(
    model: OllamaModel,
    temperature: float = 1.0,
    anti_uniform: bool = False,
    prompt_base: str = None,
) -> ProbeResult:
    """Probe C: Chain-of-thought first, then act."""
    base = prompt_base if prompt_base is not None else PROMPT_BASE
    cot_prompt = base + "\n\nThink through the options carefully:\n"
    cot_output = model.generate(cot_prompt, max_tokens=200, temperature=temperature)

    prompt = base + f"""

--- THINKING ---
{cot_output}
--- END THINKING ---

Now answer with A, B, C, or D:
"""
    token_probs, samples = model.get_option_probs(prompt, temperature=temperature)
    report_probs, report_raw, parse_ok = model.self_report(
        base, anti_uniform=anti_uniform
    )

    return analyze(
        "C: CoT → Act",
        token_probs, report_probs,
        report_parse_ok=parse_ok,
        report_raw=report_raw,
        token_samples=samples,
        n_samples=model.n_samples,
        anti_uniform=anti_uniform,
    )


def probe_D_act_then_introspect(
    model: OllamaModel,
    temperature: float = 1.0,
    anti_uniform: bool = False,
    prompt_base: str = None,
) -> ProbeResult:
    """Probe D: Act first, then introspective report."""
    base = prompt_base if prompt_base is not None else PROMPT_BASE
    prompt = base + "\n\nAnswer with A, B, C, or D:\n"
    token_probs, samples = model.get_option_probs(prompt, temperature=temperature)

    # Use introspective prompt
    report_probs, report_raw, parse_ok = model.self_report(
        base, introspective=True, anti_uniform=anti_uniform
    )

    return analyze(
        "D: Act → Introspect",
        token_probs, report_probs,
        report_parse_ok=parse_ok,
        report_raw=report_raw,
        token_samples=samples,
        n_samples=model.n_samples,
        anti_uniform=anti_uniform,
    )


# -------------------------
# Main
# -------------------------

def run_probes(
    model_name: str = "llama3.1:latest",
    n_samples: int = 50,
    temperature: float = 1.0,
    verbose: bool = False,
    anti_uniform: bool = False,
):
    """Run all probes and print results."""
    print(f"Belief Probe Baseline v2")
    constraint_status = "ACTIVE" if anti_uniform else "off"
    print(f"Model: {model_name} | Samples: {n_samples} | Temperature: {temperature} | Anti-uniform: {constraint_status}")
    print("=" * 70)

    model = OllamaModel(model_name, n_samples=n_samples)

    results = []

    print("\nRunning Probe A (Act → Report)...", end=" ", flush=True)
    results.append(probe_A_act_then_report(model, temperature, anti_uniform=anti_uniform))
    print("done")

    print("Running Probe B (Report → Act)...", end=" ", flush=True)
    results.append(probe_B_report_then_act(model, temperature, anti_uniform=anti_uniform))
    print("done")

    print("Running Probe C (CoT → Act)...", end=" ", flush=True)
    results.append(probe_C_cot_then_act(model, temperature, anti_uniform=anti_uniform))
    print("done")

    print("Running Probe D (Act → Introspect)...", end=" ", flush=True)
    results.append(probe_D_act_then_introspect(model, temperature, anti_uniform=anti_uniform))
    print("done")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for result in results:
        print()
        print(result.summary())
        if verbose:
            print(f"Report raw: {result.report_raw[:200]}...")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Probe':<22} {'Token H':>8} {'Report H':>9} {'JS Div':>8} {'TV Dist':>8} {'Parse':>6}")
    print("-" * 70)
    for r in results:
        parse_status = "OK" if r.report_parse_ok else "FAIL"
        print(f"{r.name:<22} {r.token_entropy:>8.3f} {r.report_entropy:>9.3f} {r.js_divergence:>8.4f} {r.total_variation:>8.4f} {parse_status:>6}")

    # Parse health summary
    n_parse_ok = sum(1 for r in results if r.report_parse_ok)
    print(f"\nParse success: {n_parse_ok}/{len(results)}")

    return results


def run_sweep(
    model_name: str = "llama3.1:latest",
    n_samples_list: List[int] = [10, 50],
    temperature_list: List[float] = [0.5, 1.0],
):
    """Run probe sweep over n_samples and temperature."""
    print(f"Running parameter sweep on {model_name}")
    print("=" * 70)

    all_results = []

    for n_samples in n_samples_list:
        for temp in temperature_list:
            print(f"\n--- n_samples={n_samples}, temperature={temp} ---")
            results = run_probes(model_name, n_samples=n_samples, temperature=temp)
            all_results.append({
                "n_samples": n_samples,
                "temperature": temp,
                "results": results,
            })

    # Final comparison
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY: JS Divergence by (n_samples, temp)")
    print("=" * 70)
    print(f"{'Config':<20} {'A:Act→Rep':>12} {'B:Rep→Act':>12} {'C:CoT→Act':>12} {'D:Introsp':>12}")
    print("-" * 70)

    for run in all_results:
        config = f"n={run['n_samples']}, t={run['temperature']}"
        js_vals = [f"{r.js_divergence:.4f}" for r in run['results']]
        print(f"{config:<20} {js_vals[0]:>12} {js_vals[1]:>12} {js_vals[2]:>12} {js_vals[3]:>12}")

    return all_results


def format_question_prompt(question: dict) -> str:
    """Format a question dict as a prompt string."""
    lines = [
        "You are answering a multiple choice question.",
        "",
        "Question:",
        question["question"],
        "",
        "Options:",
    ]
    for opt in ["A", "B", "C", "D"]:
        lines.append(f"{opt}) {question['options'][opt]}")

    return "\n".join(lines)


@dataclass
class QuestionResult:
    """Results for a single question across all probes."""
    question_id: int
    category: str
    question_text: str
    probe_results: Dict[str, ProbeResult]


def run_multi_question_experiment(
    model_name: str = "llama3.1:latest",
    questions_file: str = "mc_questions.json",
    n_samples: int = 20,
    temperature: float = 1.0,
    anti_uniform: bool = True,
    output_file: str = None,
):
    """Run Experiment 2: probes across multiple questions.

    Tests stability of belief-action alignment patterns across:
    - Ambiguous questions
    - Easy questions
    - Adversarially symmetric questions
    """
    print("=" * 70)
    print("EXPERIMENT 2: Multi-Question Stability Test")
    print("=" * 70)
    constraint_status = "ACTIVE" if anti_uniform else "off"
    print(f"Model: {model_name} | Samples: {n_samples} | Anti-uniform: {constraint_status}")
    print(f"Questions: {questions_file}")
    print("=" * 70)

    # Load questions
    with open(questions_file) as f:
        data = json.load(f)

    questions = data["questions"]
    print(f"Loaded {len(questions)} questions")

    model = OllamaModel(model_name, n_samples=n_samples)

    # Results by category
    results_by_category = {"ambiguous": [], "easy": [], "adversarial": []}

    # Aggregate stats
    probe_stats = {
        "A": {"js_sum": 0, "js_sq_sum": 0, "uniform_count": 0, "count": 0},
        "B": {"js_sum": 0, "js_sq_sum": 0, "uniform_count": 0, "count": 0},
        "C": {"js_sum": 0, "js_sq_sum": 0, "uniform_count": 0, "count": 0},
        "D": {"js_sum": 0, "js_sq_sum": 0, "uniform_count": 0, "count": 0},
    }

    all_results = []

    for i, q in enumerate(questions):
        category = q.get("category", "unknown")
        prompt_base = format_question_prompt(q)

        print(f"\n[{i+1}/{len(questions)}] {category}: {q['question'][:50]}...")

        probe_results = {}

        # Run all 4 probes (use reduced samples for speed)
        for probe_name, probe_fn in [
            ("A", probe_A_act_then_report),
            ("B", probe_B_report_then_act),
            ("C", probe_C_cot_then_act),
            ("D", probe_D_act_then_introspect),
        ]:
            result = probe_fn(
                model,
                temperature=temperature,
                anti_uniform=anti_uniform,
                prompt_base=prompt_base,
            )
            probe_results[probe_name] = result

            # Update stats
            stats = probe_stats[probe_name]
            stats["js_sum"] += result.js_divergence
            stats["js_sq_sum"] += result.js_divergence ** 2
            stats["count"] += 1
            if result.uniformity_pressure and result.uniformity_pressure.is_uniform:
                stats["uniform_count"] += 1

            print(f"  {probe_name}: JS={result.js_divergence:.4f}", end="")
        print()

        qr = QuestionResult(
            question_id=i,
            category=category,
            question_text=q["question"],
            probe_results=probe_results,
        )
        all_results.append(qr)
        if category in results_by_category:
            results_by_category[category].append(qr)

    # Print summary by category
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)

    for category, results in results_by_category.items():
        if not results:
            continue

        print(f"\n--- {category.upper()} ({len(results)} questions) ---")
        print(f"{'Probe':<12} {'Mean JS':>10} {'Std JS':>10} {'Uniform%':>10}")
        print("-" * 45)

        for probe_name in ["A", "B", "C", "D"]:
            js_values = [r.probe_results[probe_name].js_divergence for r in results]
            uniform_count = sum(
                1 for r in results
                if r.probe_results[probe_name].uniformity_pressure
                and r.probe_results[probe_name].uniformity_pressure.is_uniform
            )
            mean_js = sum(js_values) / len(js_values)
            std_js = (sum((x - mean_js) ** 2 for x in js_values) / len(js_values)) ** 0.5
            uniform_pct = 100 * uniform_count / len(js_values)
            print(f"{probe_name:<12} {mean_js:>10.4f} {std_js:>10.4f} {uniform_pct:>9.1f}%")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY (all questions)")
    print("=" * 70)
    print(f"{'Probe':<12} {'Mean JS':>10} {'Std JS':>10} {'Uniform%':>10}")
    print("-" * 45)

    for probe_name in ["A", "B", "C", "D"]:
        stats = probe_stats[probe_name]
        n = stats["count"]
        if n == 0:
            continue
        mean_js = stats["js_sum"] / n
        variance = (stats["js_sq_sum"] / n) - (mean_js ** 2)
        std_js = variance ** 0.5 if variance > 0 else 0
        uniform_pct = 100 * stats["uniform_count"] / n
        print(f"{probe_name:<12} {mean_js:>10.4f} {std_js:>10.4f} {uniform_pct:>9.1f}%")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best probe
    mean_js = {p: probe_stats[p]["js_sum"] / max(probe_stats[p]["count"], 1) for p in "ABCD"}
    best_probe = min(mean_js, key=mean_js.get)
    worst_probe = max(mean_js, key=mean_js.get)

    print(f"Best alignment:  Probe {best_probe} (mean JS = {mean_js[best_probe]:.4f})")
    print(f"Worst alignment: Probe {worst_probe} (mean JS = {mean_js[worst_probe]:.4f})")

    # Check if uniform attractor is broken
    total_uniform = sum(probe_stats[p]["uniform_count"] for p in "ABCD")
    total_reports = sum(probe_stats[p]["count"] for p in "ABCD")
    uniform_rate = 100 * total_uniform / max(total_reports, 1)
    print(f"Uniform reports: {uniform_rate:.1f}% (attractor {'ACTIVE' if uniform_rate > 50 else 'BROKEN'})")

    # Save detailed results if requested
    if output_file:
        output_data = {
            "config": {
                "model": model_name,
                "n_samples": n_samples,
                "temperature": temperature,
                "anti_uniform": anti_uniform,
                "questions_file": questions_file,
            },
            "summary": {
                "by_probe": {
                    p: {
                        "mean_js": probe_stats[p]["js_sum"] / max(probe_stats[p]["count"], 1),
                        "uniform_rate": probe_stats[p]["uniform_count"] / max(probe_stats[p]["count"], 1),
                        "n": probe_stats[p]["count"],
                    }
                    for p in "ABCD"
                },
            },
            "questions": [
                {
                    "id": qr.question_id,
                    "category": qr.category,
                    "question": qr.question_text,
                    "probes": {
                        name: {
                            "js_divergence": pr.js_divergence,
                            "token_entropy": pr.token_entropy,
                            "report_entropy": pr.report_entropy,
                            "is_uniform": pr.uniformity_pressure.is_uniform if pr.uniformity_pressure else None,
                        }
                        for name, pr in qr.probe_results.items()
                    }
                }
                for qr in all_results
            ]
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belief Probe Baseline")
    parser.add_argument("model", nargs="?", default="llama3.1:latest", help="Ollama model name")
    parser.add_argument("-n", "--n-samples", type=int, default=50, help="Number of samples for token estimation")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show raw report text")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--anti-uniform", action="store_true",
                        help="Add constraint forcing non-uniform distribution (5-80%% per option)")
    parser.add_argument("--experiment2", action="store_true",
                        help="Run Experiment 2: multi-question stability test")
    parser.add_argument("--questions", default="mc_questions.json",
                        help="Questions JSON file for Experiment 2")
    parser.add_argument("-o", "--output", help="Output file for detailed results (JSON)")

    args = parser.parse_args()

    if args.experiment2:
        run_multi_question_experiment(
            args.model,
            questions_file=args.questions,
            n_samples=args.n_samples,
            temperature=args.temperature,
            anti_uniform=args.anti_uniform,
            output_file=args.output,
        )
    elif args.sweep:
        run_sweep(args.model)
    else:
        run_probes(args.model, args.n_samples, args.temperature, args.verbose, args.anti_uniform)
