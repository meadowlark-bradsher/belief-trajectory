#!/usr/bin/env python3
"""
Run functional oracle comparison - executable version.

Tests which LLM can generate all 8 trajectory types.
Focus: useless rate, coverage, consistency, cost.
"""

import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass, asdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
NEW_ORACLE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NEW_ORACLE_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

from question_coverage_analysis import QuestionCoverageAnalyzer

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR


# =============================================================================
# DATA LOADING
# =============================================================================

def load_items() -> List[str]:
    with open(DATA_DIR / "items.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_test_questions(num_questions: int = 100, seed: int = 42) -> List[Dict]:
    questions = []
    with open(DATA_DIR / "questions.jsonl", "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append({"id": str(data["question_id"]), "text": data["question"]})

    random.seed(seed)
    return random.sample(questions, min(num_questions, len(questions)))


# =============================================================================
# MODEL QUERIES
# =============================================================================

def query_gemini(questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    from google import genai
    client = genai.Client()
    bitmasks = {}

    for q_idx, q in enumerate(questions):
        if (q_idx + 1) % 10 == 0 or q_idx == 0:
            print(f"    Gemini [{q_idx+1}/{len(questions)}]")

        items_list = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        prompt = f"""For each item, answer YES or NO to the question.
Reply with numbered answers only (1. YES, 2. NO, etc).

Question: {q['text']}

Items:
{items_list}

Answers:"""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 1000}
            )

            bitmask = np.zeros(len(items), dtype=bool)
            for line in response.text.strip().split("\n"):
                for i in range(len(items)):
                    if line.strip().startswith(f"{i+1}.") or line.strip().startswith(f"{i+1}:"):
                        if "YES" in line.upper():
                            bitmask[i] = True
                        break

            bitmasks[q["id"]] = bitmask
        except Exception as e:
            print(f"      Error: {e}")
            bitmasks[q["id"]] = np.zeros(len(items), dtype=bool)

        time.sleep(1.0)
    return bitmasks


def query_openai(questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    from openai import OpenAI
    client = OpenAI()
    bitmasks = {}

    for q_idx, q in enumerate(questions):
        if (q_idx + 1) % 10 == 0 or q_idx == 0:
            print(f"    GPT-4o-mini [{q_idx+1}/{len(questions)}]")

        items_list = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        prompt = f"""For each item, answer YES or NO to the question.
Reply with numbered answers only (1. YES, 2. NO, etc).

Question: {q['text']}

Items:
{items_list}

Answers:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )

            bitmask = np.zeros(len(items), dtype=bool)
            for line in response.choices[0].message.content.strip().split("\n"):
                for i in range(len(items)):
                    if line.strip().startswith(f"{i+1}.") or line.strip().startswith(f"{i+1}:"):
                        if "YES" in line.upper():
                            bitmask[i] = True
                        break

            bitmasks[q["id"]] = bitmask
        except Exception as e:
            print(f"      Error: {e}")
            bitmasks[q["id"]] = np.zeros(len(items), dtype=bool)

        time.sleep(0.3)
    return bitmasks


def query_claude(questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    bitmasks = {}

    for q_idx, q in enumerate(questions):
        if (q_idx + 1) % 10 == 0 or q_idx == 0:
            print(f"    Claude [{q_idx+1}/{len(questions)}]")

        items_list = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        prompt = f"""For each item, answer YES or NO to the question.
Reply with numbered answers only (1. YES, 2. NO, etc).

Question: {q['text']}

Items:
{items_list}

Answers:"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            bitmask = np.zeros(len(items), dtype=bool)
            for line in response.content[0].text.strip().split("\n"):
                for i in range(len(items)):
                    if line.strip().startswith(f"{i+1}.") or line.strip().startswith(f"{i+1}:"):
                        if "YES" in line.upper():
                            bitmask[i] = True
                        break

            bitmasks[q["id"]] = bitmask
        except Exception as e:
            print(f"      Error: {e}")
            bitmasks[q["id"]] = np.zeros(len(items), dtype=bool)

        time.sleep(0.3)
    return bitmasks


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_split_ratio(bitmask: np.ndarray) -> float:
    yes_count = np.sum(bitmask)
    no_count = len(bitmask) - yes_count
    return min(yes_count, no_count) / len(bitmask)


def classify_split(ratio: float) -> str:
    if ratio < 0.05: return 'very_rare'
    elif ratio < 0.15: return 'rare'
    elif ratio < 0.35: return 'skewed'
    else: return 'balanced'


def analyze_model(name: str, bitmasks: Dict[str, np.ndarray], questions: List[Dict], num_items: int = 128) -> dict:
    """Analyze for functional trajectory generation capability"""

    splits = []
    all_yes = 0
    all_no = 0
    pathological = []

    for qid, mask in bitmasks.items():
        yes_count = np.sum(mask)
        splits.append(compute_split_ratio(mask))

        if yes_count == num_items:
            all_yes += 1
            q_text = next((q['text'] for q in questions if q['id'] == qid), 'unknown')
            pathological.append(f"{qid}: '{q_text[:40]}...' (128/128 YES)")
        elif yes_count == 0:
            all_no += 1
            q_text = next((q['text'] for q in questions if q['id'] == qid), 'unknown')
            pathological.append(f"{qid}: '{q_text[:40]}...' (0/128 YES)")

    buckets = Counter([classify_split(s) for s in splits])
    total = len(splits)
    useless_rate = 100 * (all_yes + all_no) / total

    # Run coverage analysis
    question_data = [(qid, mask) for qid, mask in bitmasks.items()]
    analyzer = QuestionCoverageAnalyzer(
        num_trajectories_per_type=200,
        diversity_factor=3.0,
        include_training_requirements=True
    )
    coverage = analyzer.analyze_questions(lambda: question_data)

    return {
        "model": name,
        "questions": total,
        "very_rare_pct": 100 * buckets.get('very_rare', 0) / total,
        "rare_pct": 100 * buckets.get('rare', 0) / total,
        "skewed_pct": 100 * buckets.get('skewed', 0) / total,
        "balanced_pct": 100 * buckets.get('balanced', 0) / total,
        "all_yes": all_yes,
        "all_no": all_no,
        "useless_pct": useless_rate,
        "avg_yes_count": np.mean([np.sum(m) for m in bitmasks.values()]),
        "coverage_feasible": len(coverage.coverage_gaps) == 0,
        "coverage_gaps": coverage.coverage_gaps,
        "optimal_sample_size": coverage.recommended_sample_size,
        "pathological": pathological[:5],
    }


def test_consistency(query_func, questions: List[Dict], items: List[str]) -> float:
    """Test self-consistency by querying twice"""
    print("    Testing consistency (2 runs on 10 questions)...")

    sample = questions[:10]  # Small sample for consistency

    results1 = query_func(sample, items)
    results2 = query_func(sample, items)

    agreements = []
    for qid in results1:
        if qid in results2:
            agreement = np.mean(results1[qid] == results2[qid])
            agreements.append(agreement)

    return np.mean(agreements) if agreements else 0.0


def print_results(results: List[dict], consistencies: dict):
    print("\n" + "=" * 80)
    print("FUNCTIONAL ORACLE COMPARISON")
    print("=" * 80)
    print("Focus: Can this oracle generate all 8 trajectory types?")

    # 1. Useless Rate
    print("\n1. USELESS QUESTION RATE (CRITICAL)")
    print("-" * 80)
    print(f"{'Model':<20} {'Useless %':<12} {'All YES':<10} {'All NO':<10} {'Status'}")
    print("-" * 80)

    for r in results:
        status = "✓✓✓" if r['useless_pct'] < 5 else "✓" if r['useless_pct'] < 10 else "⚠️" if r['useless_pct'] < 15 else "❌"
        print(f"{r['model']:<20} {r['useless_pct']:>6.1f}%     {r['all_yes']:<10} {r['all_no']:<10} {status}")

    # 2. Coverage
    print("\n2. TRAJECTORY COVERAGE")
    print("-" * 80)
    print(f"{'Model':<20} {'Can Gen T1-T8?':<18} {'Sample Size':<15} {'Gaps'}")
    print("-" * 80)

    for r in results:
        feasible = "✓ YES" if r['coverage_feasible'] else "❌ NO"
        gaps = len(r['coverage_gaps'])
        print(f"{r['model']:<20} {feasible:<18} {r['optimal_sample_size']:<15} {gaps}")
        if r['coverage_gaps']:
            for gap in r['coverage_gaps'][:2]:
                print(f"{'':20}   • {gap}")

    # 3. Consistency
    print("\n3. SELF-CONSISTENCY")
    print("-" * 80)
    for model, score in consistencies.items():
        status = "✓✓✓" if score > 0.95 else "✓✓" if score > 0.90 else "✓" if score > 0.85 else "⚠️"
        print(f"{model:<20} {score:>6.1%}  {status}")

    # 4. Distribution
    print("\n4. SPLIT DISTRIBUTION")
    print("-" * 80)
    print(f"{'Model':<20} {'Very Rare':<12} {'Rare':<12} {'Skewed':<12} {'Balanced':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<20} {r['very_rare_pct']:>6.1f}%     {r['rare_pct']:>6.1f}%     "
              f"{r['skewed_pct']:>6.1f}%     {r['balanced_pct']:>6.1f}%")

    # 5. Ranking
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    for r in results:
        useless_score = max(0, 1.0 - r['useless_pct'] / 20)
        coverage_score = 1.0 if r['coverage_feasible'] else 0.5
        consistency_score = consistencies.get(r['model'], 0.9)
        balanced_score = min(r['balanced_pct'] / 5, 1.0)

        r['score'] = 0.35 * useless_score + 0.30 * coverage_score + 0.20 * consistency_score + 0.15 * balanced_score

    results.sort(key=lambda r: r['score'], reverse=True)

    for i, r in enumerate(results):
        marker = "🏆 BEST" if i == 0 else ""
        print(f"{r['model']:<20} Score: {r['score']:.3f}  {marker}")

    winner = results[0]
    print(f"\n→ Use {winner['model']}")
    print(f"  • {winner['useless_pct']:.1f}% useless")
    print(f"  • Coverage: {'✓' if winner['coverage_feasible'] else '❌'}")
    print(f"  • Consistency: {consistencies.get(winner['model'], 0):.1%}")

    if winner['pathological']:
        print(f"\n  Problematic questions:")
        for p in winner['pathological'][:3]:
            print(f"    • {p}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", "-n", type=int, default=50)
    parser.add_argument("--models", "-m", nargs="+", default=["gemini", "openai", "claude"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-consistency", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("FUNCTIONAL ORACLE COMPARISON")
    print("=" * 80)
    print(f"Testing {args.num_questions} questions across: {args.models}")

    items = load_items()
    questions = load_test_questions(args.num_questions, args.seed)
    print(f"✓ Loaded {len(items)} items, {len(questions)} questions")

    model_funcs = {
        "gemini": ("Gemini Flash 2.0", query_gemini),
        "openai": ("GPT-4o-mini", query_openai),
        "claude": ("Claude Sonnet 4", query_claude),
    }

    all_bitmasks = {}
    results = []
    consistencies = {}

    for model_key in args.models:
        if model_key not in model_funcs:
            continue

        name, query_func = model_funcs[model_key]
        print(f"\n{'='*80}\n{name}\n{'='*80}")

        # Test consistency first (small sample)
        if not args.skip_consistency:
            consistency = test_consistency(query_func, questions, items)
            consistencies[name] = consistency
            print(f"  Consistency: {consistency:.1%}")

        # Full query
        print(f"  Querying {len(questions)} questions...")
        start = time.time()
        bitmasks = query_func(questions, items)
        elapsed = time.time() - start

        all_bitmasks[name] = bitmasks
        result = analyze_model(name, bitmasks, questions, len(items))
        results.append(result)

        print(f"  Done in {elapsed:.1f}s")
        print(f"  Useless: {result['useless_pct']:.1f}%")
        print(f"  Coverage: {'✓' if result['coverage_feasible'] else '❌ gaps'}")

    print_results(results, consistencies)

    # Save
    output = {"results": results, "consistencies": consistencies}
    output_file = OUTPUT_DIR / "functional_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
