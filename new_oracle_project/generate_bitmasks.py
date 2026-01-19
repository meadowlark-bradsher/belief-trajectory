#!/usr/bin/env python3
"""
Generate bitmasks for questions using Gemini oracle.

This script queries Gemini for each question-item pair to build
bitmasks compatible with the trajectory generator.

Usage:
    # Test with 5 questions
    python generate_bitmasks.py --num-questions 5 --output test_bitmasks.jsonl

    # Generate for optimal sample (from coverage analysis)
    python generate_bitmasks.py --num-questions 1300 --output bitmasks.jsonl

    # Resume from checkpoint
    python generate_bitmasks.py --resume bitmasks.jsonl
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


@dataclass
class GeneratedQuestion:
    """A question with its Gemini-generated bitmask."""
    question_id: int
    question: str
    bitmask: int  # 128-bit integer (same format as CUQ)
    yes_count: int
    generation_time: float
    original_bitmask: Optional[int] = None  # CUQ bitmask for comparison


def load_items() -> list[str]:
    """Load the 128 items."""
    items_path = DATA_DIR / "items.txt"
    with open(items_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_questions(limit: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load questions from CUQ dataset."""
    questions_path = DATA_DIR / "questions.jsonl"
    all_questions = []

    with open(questions_path, "r") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    if limit and limit < len(all_questions):
        random.seed(seed)
        return random.sample(all_questions, limit)

    return all_questions


def load_checkpoint(output_path: Path) -> set[int]:
    """Load already-processed question IDs from output file."""
    processed = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed.add(data["question_id"])
    return processed


def bitmask_to_int(bitmask: list[bool]) -> int:
    """Convert boolean list to integer bitmask."""
    result = 0
    for i, val in enumerate(bitmask):
        if val:
            result |= (1 << i)
    return result


def query_gemini_batch(
    client: genai.Client,
    items: list[str],
    question: str,
    retries: int = 3
) -> list[bool]:
    """
    Query Gemini for all items at once.

    Returns list of booleans matching item order.
    """
    items_list = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    prompt = f"""For each item, answer YES if the item satisfies the question, NO otherwise.
Reply with ONLY numbered answers (1. YES or 1. NO), one per line.

Question: {question}

Items:
{items_list}

Answers:"""

    for attempt in range(retries):
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
            results = [False] * len(items)

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Try to parse "N. YES" or "N. NO" format
                for i in range(len(items)):
                    prefixes = [f"{i+1}.", f"{i+1}:", f"{i+1})"]
                    for prefix in prefixes:
                        if line.startswith(prefix):
                            if "YES" in line.upper():
                                results[i] = True
                            break

            return results

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Error after {retries} attempts: {e}")
                return [False] * len(items)

    return [False] * len(items)


def process_question(
    client: genai.Client,
    question_data: dict,
    items: list[str]
) -> GeneratedQuestion:
    """Process a single question and generate its bitmask."""
    start = time.time()

    responses = query_gemini_batch(client, items, question_data["question"])
    bitmask_int = bitmask_to_int(responses)
    yes_count = sum(responses)

    elapsed = time.time() - start

    return GeneratedQuestion(
        question_id=question_data["question_id"],
        question=question_data["question"],
        bitmask=bitmask_int,
        yes_count=yes_count,
        generation_time=elapsed,
        original_bitmask=question_data.get("bitmask")
    )


def run_generation(
    num_questions: int,
    output_path: Path,
    seed: int = 42,
    workers: int = 1,  # Keep at 1 for rate limiting
    verbose: bool = True
):
    """
    Generate bitmasks for questions.

    Args:
        num_questions: Number of questions to process
        output_path: Path to output JSONL file
        seed: Random seed for question sampling
        workers: Number of parallel workers (keep at 1 for rate limiting)
        verbose: Print progress
    """
    client = genai.Client()
    items = load_items()

    # Load checkpoint
    processed_ids = load_checkpoint(output_path)
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} questions already processed")

    # Load questions
    questions = load_questions(num_questions, seed)
    questions = [q for q in questions if q["question_id"] not in processed_ids]

    if not questions:
        print("All questions already processed!")
        return

    print(f"Processing {len(questions)} questions...")
    print(f"Output: {output_path}")
    print()

    # Process questions
    start_time = time.time()
    processed = 0
    errors = 0

    with open(output_path, "a") as f:
        for i, question_data in enumerate(questions):
            try:
                result = process_question(client, question_data, items)

                # Write to file
                f.write(json.dumps(asdict(result)) + "\n")
                f.flush()

                processed += 1

                if verbose and (processed % 10 == 0 or processed == len(questions)):
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (len(questions) - processed) / rate if rate > 0 else 0

                    print(f"  [{processed}/{len(questions)}] "
                          f"Rate: {rate:.1f} q/s, ETA: {eta/60:.1f} min")

                # Rate limiting - Gemini free tier is 60 RPM
                time.sleep(1.0)

            except Exception as e:
                errors += 1
                print(f"  Error processing question {question_data['question_id']}: {e}")

            except KeyboardInterrupt:
                print("\nInterrupted. Progress saved.")
                break

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Questions processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output: {output_path}")


def analyze_output(output_path: Path):
    """Analyze generated bitmasks and compare to original."""
    if not output_path.exists():
        print(f"Output file not found: {output_path}")
        return

    results = []
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        print("No results found")
        return

    print(f"\nAnalyzing {len(results)} generated bitmasks...")
    print()

    # Statistics
    yes_counts = [r["yes_count"] for r in results]
    gen_times = [r["generation_time"] for r in results]

    print(f"YES count distribution:")
    print(f"  Mean: {sum(yes_counts)/len(yes_counts):.1f}")
    print(f"  Min: {min(yes_counts)}")
    print(f"  Max: {max(yes_counts)}")
    print()
    print(f"Generation time:")
    print(f"  Mean: {sum(gen_times)/len(gen_times):.2f}s")
    print(f"  Total: {sum(gen_times)/60:.1f} min")

    # Compare to original if available
    comparisons = [r for r in results if r.get("original_bitmask")]
    if comparisons:
        print()
        print(f"Comparison to CUQ oracle ({len(comparisons)} questions):")

        agreements = []
        for r in comparisons:
            orig = r["original_bitmask"]
            new = r["bitmask"]

            # Count matching bits
            xor = orig ^ new
            diff_bits = bin(xor).count('1')
            agreement = (128 - diff_bits) / 128
            agreements.append(agreement)

        print(f"  Mean agreement: {100*sum(agreements)/len(agreements):.1f}%")
        print(f"  Min agreement: {100*min(agreements):.1f}%")
        print(f"  Max agreement: {100*max(agreements):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate bitmasks with Gemini oracle")

    parser.add_argument(
        "--num-questions", "-n",
        type=int,
        default=10,
        help="Number of questions to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="generated_bitmasks.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from existing output file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing output file instead of generating"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Determine output path
    if args.resume:
        output_path = Path(args.resume)
    else:
        output_path = OUTPUT_DIR / args.output

    if args.analyze:
        analyze_output(output_path)
    else:
        run_generation(
            num_questions=args.num_questions,
            output_path=output_path,
            seed=args.seed,
            verbose=not args.quiet
        )
        analyze_output(output_path)


if __name__ == "__main__":
    main()
