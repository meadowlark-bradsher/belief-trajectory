#!/usr/bin/env python3
"""Generate multiple choice questions for belief probe experiments.

Uses Anthropic API to generate questions across three categories:
- Ambiguous: genuinely uncertain, multiple plausible answers
- Easy: one clearly correct answer
- Adversarial: symmetric options designed to reveal bias

Usage:
    python generate_mc_questions.py --output questions.json
    python generate_mc_questions.py --output questions.json --count 20
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

# Load .env file if present
def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    import requests

# Categories and their descriptions
CATEGORIES = {
    "ambiguous": {
        "description": "Questions where multiple answers are genuinely plausible",
        "instruction": """Generate multiple choice questions where there is NO single correct answer.
The question should present a scenario where reasonable people could disagree.
Examples: ethical dilemmas, subjective preferences, incomplete information scenarios.
Each option should be defensible.""",
        "count": 15,
    },
    "easy": {
        "description": "Questions with one clearly correct answer",
        "instruction": """Generate multiple choice questions with ONE clearly correct answer.
These should be factual or logical questions where only one option is right.
Examples: basic facts, simple logic puzzles, definitions.
Three options should be clearly wrong.""",
        "count": 15,
    },
    "adversarial": {
        "description": "Symmetric options designed to reveal position bias",
        "instruction": """Generate multiple choice questions where all options are equally valid/invalid by design.
The options should be structurally symmetric - differing only in arbitrary labels or positions.
Examples: "Which letter comes first in the alphabet: A, B, C, or D?" (trick: the question itself uses A-D)
Or questions where options are permutations of the same structure.
The goal is to detect if the model has position bias (e.g., always picks A).""",
        "count": 10,
    },
}

GENERATION_PROMPT = """Generate {count} multiple choice questions for the following category:

Category: {category}
Description: {description}

{instruction}

Return the questions as a JSON array. Each question should have this exact format:
{{
  "category": "{category}",
  "question": "The question text",
  "options": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct_answer": "A" or null if ambiguous,
  "rationale": "Brief explanation of why this tests {category}"
}}

Return ONLY the JSON array, no other text. Generate exactly {count} questions."""


def generate_with_anthropic(category: str, info: dict) -> List[Dict]:
    """Generate questions using Anthropic API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    prompt = GENERATION_PROMPT.format(
        count=info["count"],
        category=category,
        description=info["description"],
        instruction=info["instruction"],
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Parse JSON from response
    response_text = message.content[0].text
    # Handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    return json.loads(response_text.strip())


def generate_with_requests(category: str, info: dict) -> List[Dict]:
    """Generate questions using direct API call."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable not set")

    prompt = GENERATION_PROMPT.format(
        count=info["count"],
        category=category,
        description=info["description"],
        instruction=info["instruction"],
    )

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    response.raise_for_status()

    response_text = response.json()["content"][0]["text"]
    # Handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    return json.loads(response_text.strip())


def generate_questions(categories: List[str] = None) -> List[Dict]:
    """Generate questions for all specified categories."""
    if categories is None:
        categories = list(CATEGORIES.keys())

    all_questions = []
    generate_fn = generate_with_anthropic if HAS_ANTHROPIC else generate_with_requests

    for category in categories:
        if category not in CATEGORIES:
            print(f"Warning: Unknown category '{category}', skipping")
            continue

        info = CATEGORIES[category]
        print(f"Generating {info['count']} {category} questions...", end=" ", flush=True)

        try:
            questions = generate_fn(category, info)
            all_questions.extend(questions)
            print(f"done ({len(questions)} generated)")
        except Exception as e:
            print(f"error: {e}")

    return all_questions


def format_for_probe(question: Dict) -> str:
    """Format a question for use with the belief probe."""
    lines = [
        "You are answering a multiple choice question.",
        "",
        "Question:",
        question["question"],
        "",
        "Options:",
    ]
    for opt, text in question["options"].items():
        lines.append(f"{opt}) {text}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate MC questions for belief probe")
    parser.add_argument("-o", "--output", default="mc_questions.json",
                        help="Output JSON file")
    parser.add_argument("-c", "--categories", nargs="+",
                        choices=list(CATEGORIES.keys()),
                        help="Categories to generate (default: all)")
    parser.add_argument("--count", type=int,
                        help="Override question count per category")

    args = parser.parse_args()

    # Override counts if specified
    if args.count:
        for cat in CATEGORIES:
            CATEGORIES[cat]["count"] = args.count

    print("MC Question Generator for Belief Probe Experiments")
    print("=" * 50)

    questions = generate_questions(args.categories)

    # Save to file
    output = {
        "metadata": {
            "total_questions": len(questions),
            "categories": {cat: sum(1 for q in questions if q.get("category") == cat)
                          for cat in CATEGORIES},
        },
        "questions": questions,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(questions)} questions to {args.output}")

    # Print summary
    print("\nCategory breakdown:")
    for cat, count in output["metadata"]["categories"].items():
        print(f"  {cat}: {count}")

    # Print sample from each category
    print("\nSample questions:")
    for cat in CATEGORIES:
        samples = [q for q in questions if q.get("category") == cat][:1]
        for q in samples:
            print(f"\n[{cat}] {q['question']}")
            for opt, text in q["options"].items():
                print(f"  {opt}) {text}")


if __name__ == "__main__":
    main()
