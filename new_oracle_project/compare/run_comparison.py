#!/usr/bin/env python3
"""
Compare oracle models for trajectory generation suitability.

Focus: Split distribution, useless rate, inter-model agreement, cost.
NOT focused on matching CUQ oracle.
"""

import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR


def load_items() -> List[str]:
    with open(DATA_DIR / "items.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_questions_sample(n: int = 50, seed: int = 42) -> List[Dict]:
    questions = []
    with open(DATA_DIR / "questions.jsonl", "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append({"id": str(data["question_id"]), "text": data["question"]})

    random.seed(seed)
    return random.sample(questions, min(n, len(questions)))


def compute_split_ratio(bitmask: np.ndarray) -> float:
    """min(yes, no) / total"""
    yes_count = np.sum(bitmask)
    no_count = len(bitmask) - yes_count
    return min(yes_count, no_count) / len(bitmask)


def classify_split(ratio: float) -> str:
    if ratio < 0.05: return 'very_rare'
    elif ratio < 0.15: return 'rare'
    elif ratio < 0.35: return 'skewed'
    else: return 'balanced'


# =============================================================================
# MODEL QUERIES
# =============================================================================

def query_gemini(questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    from google import genai
    client = genai.Client()
    bitmasks = {}

    for q_idx, q in enumerate(questions):
        print(f"  Gemini [{q_idx+1}/{len(questions)}] {q['text'][:50]}...")

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
                line = line.strip()
                for i in range(len(items)):
                    if line.startswith(f"{i+1}.") or line.startswith(f"{i+1}:") or line.startswith(f"{i+1})"):
                        if "YES" in line.upper():
                            bitmask[i] = True
                        break

            bitmasks[q["id"]] = bitmask
        except Exception as e:
            print(f"    Error: {e}")
            bitmasks[q["id"]] = np.zeros(len(items), dtype=bool)

        time.sleep(1.0)
    return bitmasks


def query_openai(questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    from openai import OpenAI
    client = OpenAI()
    bitmasks = {}

    for q_idx, q in enumerate(questions):
        print(f"  GPT-4o-mini [{q_idx+1}/{len(questions)}] {q['text'][:50]}...")

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
                line = line.strip()
                for i in range(len(items)):
                    if line.startswith(f"{i+1}.") or line.startswith(f"{i+1}:") or line.startswith(f"{i+1})"):
                        if "YES" in line.upper():
                            bitmask[i] = True
                        break

            bitmasks[q["id"]] = bitmask
        except Exception as e:
            print(f"    Error: {e}")
            bitmasks[q["id"]] = np.zeros(len(items), dtype=bool)

        time.sleep(0.5)
    return bitmasks


def query_claude(questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    bitmasks = {}

    for q_idx, q in enumerate(questions):
        print(f"  Claude [{q_idx+1}/{len(questions)}] {q['text'][:50]}...")

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
                line = line.strip()
                for i in range(len(items)):
                    if line.startswith(f"{i+1}.") or line.startswith(f"{i+1}:") or line.startswith(f"{i+1})"):
                        if "YES" in line.upper():
                            bitmask[i] = True
                        break

            bitmasks[q["id"]] = bitmask
        except Exception as e:
            print(f"    Error: {e}")
            bitmasks[q["id"]] = np.zeros(len(items), dtype=bool)

        time.sleep(0.5)
    return bitmasks


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_model(name: str, bitmasks: Dict[str, np.ndarray], num_items: int = 128) -> dict:
    """Analyze one model's bitmasks for trajectory suitability"""
    splits = []
    all_yes = 0
    all_no = 0

    for qid, mask in bitmasks.items():
        yes_count = np.sum(mask)
        splits.append(compute_split_ratio(mask))

        if yes_count == num_items:
            all_yes += 1
        elif yes_count == 0:
            all_no += 1

    buckets = Counter([classify_split(s) for s in splits])
    total = len(splits)

    return {
        "model": name,
        "questions": total,
        "very_rare_pct": 100 * buckets.get('very_rare', 0) / total,
        "rare_pct": 100 * buckets.get('rare', 0) / total,
        "skewed_pct": 100 * buckets.get('skewed', 0) / total,
        "balanced_pct": 100 * buckets.get('balanced', 0) / total,
        "all_yes": all_yes,
        "all_no": all_no,
        "useless_pct": 100 * (all_yes + all_no) / total,
        "avg_yes_count": np.mean([np.sum(m) for m in bitmasks.values()]),
    }


def compute_inter_model_agreement(bitmasks_a: Dict, bitmasks_b: Dict) -> float:
    """Average agreement between two models on shared questions"""
    agreements = []
    for qid in bitmasks_a:
        if qid in bitmasks_b:
            agreement = np.mean(bitmasks_a[qid] == bitmasks_b[qid])
            agreements.append(agreement)
    return np.mean(agreements) if agreements else 0.0


def print_results(results: List[dict], inter_agreements: dict):
    print("\n" + "=" * 80)
    print("ORACLE MODEL COMPARISON - TRAJECTORY SUITABILITY")
    print("=" * 80)

    # Split distribution
    print("\n1. SPLIT RATIO DISTRIBUTION")
    print("-" * 80)
    print(f"{'Model':<20} {'Very Rare':<12} {'Rare':<12} {'Skewed':<12} {'Balanced':<12}")
    print(f"{'(target for T4)':<20} {'(T2,T7)':<12} {'(T5)':<12} {'(T1,T6-T8)':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['model']:<20} {r['very_rare_pct']:>6.1f}%     "
              f"{r['rare_pct']:>6.1f}%     {r['skewed_pct']:>6.1f}%     "
              f"{r['balanced_pct']:>6.1f}%")

    # Useless questions
    print("\n2. USELESS QUESTIONS (all-YES or all-NO)")
    print("-" * 80)
    print(f"{'Model':<20} {'All YES':<10} {'All NO':<10} {'Useless %':<12} {'Avg YES':<12}")
    print("-" * 80)

    for r in results:
        status = "✓" if r['useless_pct'] < 5 else "⚠️" if r['useless_pct'] < 15 else "❌"
        print(f"{r['model']:<20} {r['all_yes']:<10} {r['all_no']:<10} "
              f"{r['useless_pct']:>6.1f}% {status}   {r['avg_yes_count']:>6.1f}/128")

    # Inter-model agreement
    print("\n3. INTER-MODEL AGREEMENT")
    print("-" * 80)
    for pair, agreement in inter_agreements.items():
        print(f"  {pair}: {100*agreement:.1f}%")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Score: balanced availability (40%) + low useless (40%) + moderate yes count (20%)
    for r in results:
        balanced_score = min(r['balanced_pct'] / 10, 1.0)  # 10% balanced = perfect
        useless_score = 1.0 - min(r['useless_pct'] / 20, 1.0)  # 0% useless = perfect
        # Prefer avg YES around 30-50 (informative questions)
        yes_score = 1.0 - abs(r['avg_yes_count'] - 40) / 88  # 40 is ideal

        r['score'] = 0.4 * balanced_score + 0.4 * useless_score + 0.2 * yes_score

    results.sort(key=lambda r: r['score'], reverse=True)

    for i, r in enumerate(results):
        marker = "🏆 BEST" if i == 0 else ""
        print(f"{r['model']:<20} Score: {r['score']:.3f}  {marker}")

    winner = results[0]
    print(f"\n→ Use {winner['model']}")
    print(f"  • {winner['balanced_pct']:.1f}% balanced splits")
    print(f"  • {winner['useless_pct']:.1f}% useless questions")
    print(f"  • {winner['avg_yes_count']:.1f} avg YES per question")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", "-n", type=int, default=30)
    parser.add_argument("--models", "-m", nargs="+", default=["gemini", "openai", "claude"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print("ORACLE MODEL COMPARISON")
    print("=" * 80)
    print(f"Testing {args.num_questions} questions across: {args.models}")

    items = load_items()
    questions = load_questions_sample(args.num_questions, args.seed)
    print(f"✓ Loaded {len(items)} items, {len(questions)} questions")

    model_funcs = {
        "gemini": ("Gemini Flash 2.0", query_gemini),
        "openai": ("GPT-4o-mini", query_openai),
        "claude": ("Claude Sonnet 4", query_claude),
    }

    all_bitmasks = {}
    results = []

    for model_key in args.models:
        if model_key not in model_funcs:
            continue

        name, query_func = model_funcs[model_key]
        print(f"\n{'='*80}\nQuerying {name}...\n{'='*80}")

        start = time.time()
        bitmasks = query_func(questions, items)
        elapsed = time.time() - start

        all_bitmasks[name] = bitmasks
        result = analyze_model(name, bitmasks, len(items))
        results.append(result)

        print(f"✓ Done in {elapsed:.1f}s")

    # Inter-model agreement
    inter_agreements = {}
    model_names = list(all_bitmasks.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            agreement = compute_inter_model_agreement(all_bitmasks[m1], all_bitmasks[m2])
            inter_agreements[f"{m1} vs {m2}"] = agreement

    print_results(results, inter_agreements)

    # Save
    output = {
        "results": results,
        "inter_model_agreement": inter_agreements,
    }
    output_file = OUTPUT_DIR / "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
