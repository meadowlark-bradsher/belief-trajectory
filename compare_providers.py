#!/usr/bin/env python3
"""Cross-provider comparison for belief probes.

Runs the same probes across multiple providers (with true logprobs where available)
and compares belief-action alignment patterns.

Usage:
    python compare_providers.py --questions mc_questions.json --providers openai gemini ollama
    python compare_providers.py --quick  # Quick test with 5 questions per category
"""

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

from providers import get_provider, BaseProvider, OPTIONS


@dataclass
class ProbeMetrics:
    """Metrics for a single probe on a single question."""
    js_divergence: float
    token_entropy: float
    report_entropy: float
    kl_token_report: float
    total_variation: float
    report_parse_ok: bool
    report_is_uniform: bool
    action_method: str  # "logprobs" or "sampling"


def entropy(probs: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    return sum(p[k] * math.log2((p[k] + eps) / (q[k] + eps)) for k in p if p[k] > 0)


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    m = {k: (p[k] + q[k]) / 2 for k in p}
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    return sum(abs(p[k] - q[k]) for k in p) / 2


def format_question_prompt(question: dict) -> str:
    lines = [
        "You are answering a multiple choice question.",
        "",
        "Question:",
        question["question"],
        "",
        "Options:",
    ]
    for opt in OPTIONS:
        lines.append(f"{opt}) {question['options'][opt]}")
    return "\n".join(lines)


def run_probe_a(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> ProbeMetrics:
    """Probe A: Act → Report"""
    prompt = prompt_base + "\n\nAnswer with A, B, C, or D:\n"
    action = provider.get_action_probs(prompt)
    report = provider.get_report(prompt_base, anti_uniform=anti_uniform)

    return ProbeMetrics(
        js_divergence=js_divergence(action.probs, report.probs),
        token_entropy=action.entropy,
        report_entropy=entropy(report.probs),
        kl_token_report=kl_divergence(action.probs, report.probs),
        total_variation=total_variation(action.probs, report.probs),
        report_parse_ok=report.parse_ok,
        report_is_uniform=report.is_uniform,
        action_method=action.method,
    )


def run_probe_b(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> ProbeMetrics:
    """Probe B: Report → Act"""
    report = provider.get_report(prompt_base, anti_uniform=anti_uniform)

    prompt = prompt_base + """

--- BELIEF RECORDED ---
Now answer with A, B, C, or D:
"""
    action = provider.get_action_probs(prompt)

    return ProbeMetrics(
        js_divergence=js_divergence(action.probs, report.probs),
        token_entropy=action.entropy,
        report_entropy=entropy(report.probs),
        kl_token_report=kl_divergence(action.probs, report.probs),
        total_variation=total_variation(action.probs, report.probs),
        report_parse_ok=report.parse_ok,
        report_is_uniform=report.is_uniform,
        action_method=action.method,
    )


def run_probe_c(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> ProbeMetrics:
    """Probe C: CoT → Act"""
    cot_prompt = prompt_base + "\n\nThink through the options carefully:\n"
    cot_output = provider.generate(cot_prompt, max_tokens=200, temperature=1.0)

    prompt = prompt_base + f"""

--- THINKING ---
{cot_output}
--- END THINKING ---

Now answer with A, B, C, or D:
"""
    action = provider.get_action_probs(prompt)
    report = provider.get_report(prompt_base, anti_uniform=anti_uniform)

    return ProbeMetrics(
        js_divergence=js_divergence(action.probs, report.probs),
        token_entropy=action.entropy,
        report_entropy=entropy(report.probs),
        kl_token_report=kl_divergence(action.probs, report.probs),
        total_variation=total_variation(action.probs, report.probs),
        report_parse_ok=report.parse_ok,
        report_is_uniform=report.is_uniform,
        action_method=action.method,
    )


def run_probe_d(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> ProbeMetrics:
    """Probe D: Act → Introspect"""
    prompt = prompt_base + "\n\nAnswer with A, B, C, or D:\n"
    action = provider.get_action_probs(prompt)
    report = provider.get_report(prompt_base, introspective=True, anti_uniform=anti_uniform)

    return ProbeMetrics(
        js_divergence=js_divergence(action.probs, report.probs),
        token_entropy=action.entropy,
        report_entropy=entropy(report.probs),
        kl_token_report=kl_divergence(action.probs, report.probs),
        total_variation=total_variation(action.probs, report.probs),
        report_parse_ok=report.parse_ok,
        report_is_uniform=report.is_uniform,
        action_method=action.method,
    )


PROBES = {
    "A": run_probe_a,
    "B": run_probe_b,
    "C": run_probe_c,
    "D": run_probe_d,
}


def run_comparison(
    providers_config: List[dict],
    questions_file: str,
    output_file: str = None,
    max_per_category: int = None,
    anti_uniform: bool = True,
):
    """Run cross-provider comparison."""
    print("=" * 70)
    print("CROSS-PROVIDER BELIEF PROBE COMPARISON")
    print("=" * 70)

    # Load questions
    with open(questions_file) as f:
        data = json.load(f)

    questions = data["questions"]

    # Optionally limit per category
    if max_per_category:
        filtered = []
        counts = {}
        for q in questions:
            cat = q.get("category", "unknown")
            if counts.get(cat, 0) < max_per_category:
                filtered.append(q)
                counts[cat] = counts.get(cat, 0) + 1
        questions = filtered

    print(f"Questions: {len(questions)}")
    print(f"Anti-uniform constraint: {'ACTIVE' if anti_uniform else 'off'}")
    print()

    # Initialize providers
    providers = {}
    for cfg in providers_config:
        name = cfg.pop("name")
        try:
            providers[name] = get_provider(name, **cfg)
            print(f"✓ {name}: {providers[name].model_name} (logprobs: {providers[name].supports_logprobs})")
        except Exception as e:
            print(f"✗ {name}: {e}")
        finally:
            cfg["name"] = name  # Restore for later use

    if not providers:
        print("No providers available!")
        return

    print()

    # Results storage
    all_results = []

    # Run probes
    for qi, q in enumerate(questions):
        category = q.get("category", "unknown")
        prompt_base = format_question_prompt(q)

        print(f"[{qi+1}/{len(questions)}] {category}: {q['question'][:40]}...")

        q_results = {
            "question_id": qi,
            "category": category,
            "question": q["question"],
            "providers": {},
        }

        for prov_name, provider in providers.items():
            prov_results = {}
            print(f"  {prov_name}: ", end="", flush=True)

            for probe_name, probe_fn in PROBES.items():
                try:
                    metrics = probe_fn(provider, prompt_base, anti_uniform=anti_uniform)
                    prov_results[probe_name] = asdict(metrics)
                    print(f"{probe_name}={metrics.js_divergence:.3f} ", end="", flush=True)
                except Exception as e:
                    prov_results[probe_name] = {"error": str(e)}
                    print(f"{probe_name}=ERR ", end="", flush=True)

            print()
            q_results["providers"][prov_name] = prov_results

        all_results.append(q_results)

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY BY PROVIDER")
    print("=" * 70)

    for prov_name in providers:
        print(f"\n--- {prov_name.upper()} ({providers[prov_name].model_name}) ---")
        print(f"Method: {providers[prov_name].supports_logprobs and 'logprobs' or 'sampling'}")
        print(f"{'Probe':<8} {'Mean JS':>10} {'Std JS':>10} {'Uniform%':>10}")
        print("-" * 45)

        for probe_name in PROBES:
            js_values = []
            uniform_count = 0
            for r in all_results:
                pdata = r["providers"].get(prov_name, {}).get(probe_name, {})
                if "js_divergence" in pdata:
                    js_values.append(pdata["js_divergence"])
                    if pdata.get("report_is_uniform"):
                        uniform_count += 1

            if js_values:
                mean_js = sum(js_values) / len(js_values)
                std_js = (sum((x - mean_js) ** 2 for x in js_values) / len(js_values)) ** 0.5
                uniform_pct = 100 * uniform_count / len(js_values)
                print(f"{probe_name:<8} {mean_js:>10.4f} {std_js:>10.4f} {uniform_pct:>9.1f}%")

    # Cross-provider comparison
    print("\n" + "=" * 70)
    print("CROSS-PROVIDER COMPARISON (Mean JS by Probe)")
    print("=" * 70)

    header = f"{'Provider':<15}"
    for probe_name in PROBES:
        header += f" {probe_name:>10}"
    header += f" {'Best':>8}"
    print(header)
    print("-" * 70)

    for prov_name in providers:
        row = f"{prov_name:<15}"
        probe_means = {}
        for probe_name in PROBES:
            js_values = [
                r["providers"].get(prov_name, {}).get(probe_name, {}).get("js_divergence")
                for r in all_results
            ]
            js_values = [v for v in js_values if v is not None]
            if js_values:
                mean_js = sum(js_values) / len(js_values)
                probe_means[probe_name] = mean_js
                row += f" {mean_js:>10.4f}"
            else:
                row += f" {'N/A':>10}"

        if probe_means:
            best = min(probe_means, key=probe_means.get)
            row += f" {best:>8}"
        print(row)

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION: JS vs Token Entropy (per provider)")
    print("=" * 70)

    def pearson_corr(x, y):
        n = len(x)
        if n < 3:
            return float('nan')
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        if den_x * den_y == 0:
            return 0
        return num / (den_x * den_y)

    for prov_name in providers:
        print(f"\n{prov_name}:")
        for probe_name in PROBES:
            js_vals = []
            h_vals = []
            for r in all_results:
                pdata = r["providers"].get(prov_name, {}).get(probe_name, {})
                if "js_divergence" in pdata and "token_entropy" in pdata:
                    js_vals.append(pdata["js_divergence"])
                    h_vals.append(pdata["token_entropy"])

            if len(js_vals) >= 3:
                corr = pearson_corr(js_vals, h_vals)
                print(f"  {probe_name}: r={corr:+.3f}")

    # Save results
    if output_file:
        output_data = {
            "config": {
                "questions_file": questions_file,
                "n_questions": len(questions),
                "anti_uniform": anti_uniform,
                "providers": [
                    {
                        "name": name,
                        "model": providers[name].model_name,
                        "supports_logprobs": providers[name].supports_logprobs,
                    }
                    for name in providers
                ],
            },
            "results": all_results,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-provider belief probe comparison")
    parser.add_argument("--questions", default="mc_questions.json", help="Questions JSON file")
    parser.add_argument("--providers", nargs="+", default=["openai", "gemini"],
                        help="Providers to compare")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 questions per category)")
    parser.add_argument("--no-anti-uniform", action="store_true", help="Disable anti-uniform constraint")

    args = parser.parse_args()

    # Build provider configs
    provider_configs = []
    for p in args.providers:
        if p == "ollama":
            provider_configs.append({"name": "ollama", "model_name": "llama3.1:latest", "n_samples": 20})
        elif p == "openai":
            provider_configs.append({"name": "openai", "model_name": "gpt-4o-mini"})
        elif p == "gemini":
            provider_configs.append({"name": "gemini", "model_name": "gemini-2.0-flash"})
        else:
            provider_configs.append({"name": p})

    run_comparison(
        providers_config=provider_configs,
        questions_file=args.questions,
        output_file=args.output,
        max_per_category=5 if args.quick else None,
        anti_uniform=not args.no_anti_uniform,
    )
