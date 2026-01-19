#!/usr/bin/env python3
"""
Oracle Consistency Test

Tests whether Gemini produces consistent and reasonable bitmasks
by comparing against the original CUQ (T5-XL) oracle.

This is a small-scale test before committing to full regeneration.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


@dataclass
class ConsistencyResult:
    """Result of comparing Gemini oracle to CUQ oracle."""
    question_id: str
    question_text: str
    cuq_yes_count: int
    gemini_yes_count: int
    agreement_count: int  # Items where both oracles agree
    jaccard_similarity: float  # Intersection / Union
    gemini_responses: list[bool]  # Raw responses for debugging


def load_items() -> list[str]:
    """Load the 128 items from items.txt."""
    items_path = DATA_DIR / "items.txt"
    with open(items_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_sample_questions(n: int = 10, seed: int = 42) -> list[dict]:
    """Load a random sample of questions from CUQ dataset."""
    import random
    random.seed(seed)

    questions_path = DATA_DIR / "questions.jsonl"
    all_questions = []

    with open(questions_path, "r") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    return random.sample(all_questions, min(n, len(all_questions)))


def int_to_bitmask(bitmask_int: int, num_bits: int = 128) -> list[bool]:
    """Convert integer bitmask to list of booleans."""
    binary_str = format(bitmask_int, f'0{num_bits}b')[::-1]
    return [c == '1' for c in binary_str]


def query_gemini_single(client: genai.Client, item: str, question: str) -> Optional[bool]:
    """
    Query Gemini for a single item-question pair.

    Returns True for YES, False for NO, None if unclear.
    """
    prompt = f"""Answer YES or NO only.

Question: {question}
Item: {item}

Does the item "{item}" answer YES to the question "{question}"?

Answer with just YES or NO:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0,  # Deterministic
                "max_output_tokens": 10,
            }
        )

        answer = response.text.strip().upper()

        if "YES" in answer:
            return True
        elif "NO" in answer:
            return False
        else:
            print(f"  Unclear response for {item}: {answer}")
            return None

    except Exception as e:
        print(f"  Error querying {item}: {e}")
        return None


def query_gemini_batch(client: genai.Client, items: list[str], question: str) -> list[Optional[bool]]:
    """
    Query Gemini for all items at once (more efficient).

    Returns list of booleans matching item order.
    """
    items_list = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    prompt = f"""For each item below, answer YES or NO to the question.
Reply with ONLY a numbered list of YES or NO answers, one per line.

Question: {question}

Items:
{items_list}

Answers (YES or NO for each item, numbered 1-{len(items)}):"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0,
                "max_output_tokens": 1000,
            }
        )

        # Parse response
        lines = response.text.strip().split("\n")
        results = []

        for i, item in enumerate(items):
            found = False
            for line in lines:
                # Try to find answer for this item number
                if line.strip().startswith(f"{i+1}.") or line.strip().startswith(f"{i+1}:"):
                    answer = line.upper()
                    if "YES" in answer:
                        results.append(True)
                        found = True
                        break
                    elif "NO" in answer:
                        results.append(False)
                        found = True
                        break

            if not found:
                # Fallback: look for item name in any line
                for line in lines:
                    if item.lower() in line.lower():
                        answer = line.upper()
                        if "YES" in answer:
                            results.append(True)
                            found = True
                            break
                        elif "NO" in answer:
                            results.append(False)
                            found = True
                            break

            if not found:
                results.append(None)

        return results

    except Exception as e:
        print(f"  Batch query error: {e}")
        return [None] * len(items)


def run_consistency_test(
    num_questions: int = 5,
    use_batch: bool = True,
    verbose: bool = True
) -> list[ConsistencyResult]:
    """
    Run the oracle consistency test.

    Args:
        num_questions: Number of questions to test
        use_batch: If True, query all items at once (faster but may be less accurate)
        verbose: Print progress

    Returns:
        List of ConsistencyResult objects
    """
    # Initialize Gemini client
    client = genai.Client()

    # Load data
    items = load_items()
    questions = load_sample_questions(num_questions)

    if verbose:
        print(f"Testing {num_questions} questions against {len(items)} items")
        print(f"Mode: {'batch' if use_batch else 'single'}")
        print()

    results = []

    for q_idx, question_data in enumerate(questions):
        question_id = str(question_data["question_id"])
        question_text = question_data["question"]
        cuq_bitmask = int_to_bitmask(question_data["bitmask"])

        if verbose:
            print(f"[{q_idx + 1}/{num_questions}] {question_text[:60]}...")

        # Query Gemini
        start = time.time()

        if use_batch:
            gemini_responses = query_gemini_batch(client, items, question_text)
        else:
            gemini_responses = []
            for item in items:
                response = query_gemini_single(client, item, question_text)
                gemini_responses.append(response)
                time.sleep(0.1)  # Rate limiting

        elapsed = time.time() - start

        # Calculate metrics
        cuq_yes = sum(cuq_bitmask)
        gemini_yes = sum(1 for r in gemini_responses if r is True)

        # Agreement and Jaccard
        agreement = 0
        intersection = 0
        union = 0

        for cuq_val, gem_val in zip(cuq_bitmask, gemini_responses):
            if gem_val is not None:
                if cuq_val == gem_val:
                    agreement += 1
                if cuq_val or gem_val:
                    union += 1
                if cuq_val and gem_val:
                    intersection += 1

        valid_responses = sum(1 for r in gemini_responses if r is not None)
        jaccard = intersection / union if union > 0 else 0

        result = ConsistencyResult(
            question_id=question_id,
            question_text=question_text,
            cuq_yes_count=cuq_yes,
            gemini_yes_count=gemini_yes,
            agreement_count=agreement,
            jaccard_similarity=jaccard,
            gemini_responses=[r if r is not None else False for r in gemini_responses]
        )
        results.append(result)

        if verbose:
            print(f"  CUQ: {cuq_yes}/128 YES, Gemini: {gemini_yes}/128 YES")
            print(f"  Agreement: {agreement}/{valid_responses} ({100*agreement/valid_responses:.1f}%)")
            print(f"  Jaccard: {jaccard:.3f}")
            print(f"  Time: {elapsed:.1f}s")
            print()

    return results


def analyze_results(results: list[ConsistencyResult]) -> dict:
    """Analyze consistency test results."""
    if not results:
        return {}

    agreements = [r.agreement_count / 128 for r in results]
    jaccards = [r.jaccard_similarity for r in results]

    analysis = {
        "num_questions_tested": len(results),
        "avg_agreement_rate": sum(agreements) / len(agreements),
        "min_agreement_rate": min(agreements),
        "max_agreement_rate": max(agreements),
        "avg_jaccard_similarity": sum(jaccards) / len(jaccards),
        "min_jaccard_similarity": min(jaccards),
        "max_jaccard_similarity": max(jaccards),
    }

    # Interpretation
    avg_agreement = analysis["avg_agreement_rate"]
    if avg_agreement >= 0.9:
        analysis["interpretation"] = "EXCELLENT - Gemini closely matches CUQ oracle"
        analysis["recommendation"] = "Safe to proceed with full regeneration"
    elif avg_agreement >= 0.8:
        analysis["interpretation"] = "GOOD - Gemini mostly agrees with CUQ oracle"
        analysis["recommendation"] = "Proceed with caution, may need validation"
    elif avg_agreement >= 0.7:
        analysis["interpretation"] = "MODERATE - Significant differences from CUQ"
        analysis["recommendation"] = "Consider testing more questions before proceeding"
    else:
        analysis["interpretation"] = "LOW - Gemini significantly differs from CUQ"
        analysis["recommendation"] = "May need different prompting or model"

    return analysis


def main():
    """Run the consistency test."""
    print("=" * 60)
    print("ORACLE CONSISTENCY TEST")
    print("=" * 60)
    print()

    # Run test with small sample
    results = run_consistency_test(
        num_questions=5,
        use_batch=True,
        verbose=True
    )

    # Analyze
    analysis = analyze_results(results)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Questions tested: {analysis['num_questions_tested']}")
    print(f"Average agreement: {100*analysis['avg_agreement_rate']:.1f}%")
    print(f"Agreement range: {100*analysis['min_agreement_rate']:.1f}% - {100*analysis['max_agreement_rate']:.1f}%")
    print(f"Average Jaccard: {analysis['avg_jaccard_similarity']:.3f}")
    print()
    print(f"Interpretation: {analysis['interpretation']}")
    print(f"Recommendation: {analysis['recommendation']}")

    # Save results
    output = {
        "results": [
            {
                "question_id": r.question_id,
                "question_text": r.question_text,
                "cuq_yes_count": r.cuq_yes_count,
                "gemini_yes_count": r.gemini_yes_count,
                "agreement_count": r.agreement_count,
                "jaccard_similarity": r.jaccard_similarity,
            }
            for r in results
        ],
        "analysis": analysis
    }

    output_path = OUTPUT_DIR / "oracle_consistency_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
