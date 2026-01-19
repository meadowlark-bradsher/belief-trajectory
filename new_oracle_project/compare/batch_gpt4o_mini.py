#!/usr/bin/env python3
"""
GPT-4o-mini Batch API for Question Bitmask Generation

Uses OpenAI's batch API to efficiently query 1000 questions across 128 items.
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

client = OpenAI()


def load_questions(limit: int = 1000) -> list:
    """Load random sample of questions from CUQ dataset"""
    questions_path = DATA_DIR / "questions.jsonl"

    all_questions = []
    with open(questions_path, "r") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    # Random sample
    if len(all_questions) > limit:
        questions = random.sample(all_questions, limit)
    else:
        questions = all_questions

    print(f"Loaded {len(questions)} questions")
    return questions


def load_items() -> list:
    """Load the 128 items"""
    items_path = DATA_DIR / "items.txt"
    with open(items_path, "r") as f:
        items = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(items)} items")
    return items


def create_batch_inputs(questions: list, items: list, max_requests: int = 50000) -> list:
    """Create JSONL files for batch API input, splitting if needed"""

    # Calculate how many questions per batch
    requests_per_question = len(items)  # 128
    questions_per_batch = max_requests // requests_per_question  # 390

    batch_files = []

    for batch_idx, start in enumerate(range(0, len(questions), questions_per_batch)):
        batch_questions = questions[start:start + questions_per_batch]
        batch_requests = []

        for q in batch_questions:
            qid = q["question_id"]
            qtext = q["question"]

            for idx, item in enumerate(items):
                custom_id = f"q{qid}_i{idx}"

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are playing 20 Questions. Answer YES or NO only."
                            },
                            {
                                "role": "user",
                                "content": f"The secret is: {item}\n\nQuestion: {qtext}\n\nAnswer YES or NO:"
                            }
                        ],
                        "max_tokens": 5,
                        "temperature": 0
                    }
                }
                batch_requests.append(request)

        # Write batch input file
        batch_file = SCRIPT_DIR / f"batch_input_{batch_idx}.jsonl"
        with open(batch_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")

        print(f"Batch {batch_idx}: {len(batch_requests)} requests ({len(batch_questions)} questions)")
        batch_files.append(batch_file)

    print(f"\nCreated {len(batch_files)} batch files")
    return batch_files


def submit_batches(batch_files: list, num_questions: int) -> list:
    """Upload files and submit batch jobs"""

    batch_ids = []

    for idx, batch_file in enumerate(batch_files):
        print(f"\nBatch {idx}:")
        print(f"  Uploading {batch_file.name}...")

        with open(batch_file, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"  File ID: {file_obj.id}")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"GPT-4o-mini bitmask batch {idx}"}
        )
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")

        batch_ids.append({
            "batch_id": batch.id,
            "file_id": file_obj.id,
            "batch_file": str(batch_file)
        })

    # Save all batch IDs
    status_file = SCRIPT_DIR / "batch_status.json"
    with open(status_file, "w") as f:
        json.dump({
            "batches": batch_ids,
            "num_questions": num_questions,
            "num_items": 128,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    return batch_ids


def check_batch_status() -> list:
    """Check status of all batch jobs"""

    status_file = SCRIPT_DIR / "batch_status.json"
    if not status_file.exists():
        print("No batch status found. Run with --submit first.")
        return None

    with open(status_file) as f:
        data = json.load(f)

    # Handle both old single-batch and new multi-batch format
    if "batches" in data:
        batch_ids = [b["batch_id"] for b in data["batches"]]
    else:
        batch_ids = [data["batch_id"]]

    results = []
    all_complete = True

    for idx, batch_id in enumerate(batch_ids):
        batch = client.batches.retrieve(batch_id)

        print(f"\nBatch {idx}: {batch.status}")
        print(f"  Total: {batch.request_counts.total}")
        print(f"  Completed: {batch.request_counts.completed}")
        print(f"  Failed: {batch.request_counts.failed}")

        if batch.status == "completed":
            print(f"  Output file: {batch.output_file_id}")
        elif batch.status != "completed":
            all_complete = False

        results.append({
            "batch_id": batch_id,
            "status": batch.status,
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
            "output_file_id": batch.output_file_id if batch.status == "completed" else None
        })

    if all_complete:
        print("\n✓ All batches complete! Run --download to get results.")

    return results


def download_results() -> Path:
    """Download all batch results and merge"""

    status_file = SCRIPT_DIR / "batch_status.json"
    with open(status_file) as f:
        data = json.load(f)

    # Handle both formats
    if "batches" in data:
        batch_ids = [b["batch_id"] for b in data["batches"]]
    else:
        batch_ids = [data["batch_id"]]

    output_file = SCRIPT_DIR / "batch_output.jsonl"

    with open(output_file, "w") as outf:
        for idx, batch_id in enumerate(batch_ids):
            batch = client.batches.retrieve(batch_id)

            if batch.status != "completed":
                print(f"Batch {idx} not complete yet. Status: {batch.status}")
                continue

            print(f"Downloading batch {idx} from {batch.output_file_id}...")
            content = client.files.content(batch.output_file_id)

            # Append to merged output
            outf.write(content.text)
            if not content.text.endswith("\n"):
                outf.write("\n")

    print(f"\nMerged results saved to {output_file}")
    return output_file


def process_results(questions: list, items: list) -> dict:
    """Process batch results into bitmasks"""

    output_file = SCRIPT_DIR / "batch_output.jsonl"

    # Parse results
    results = {}
    with open(output_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                custom_id = data["custom_id"]

                # Parse custom_id: q{qid}_i{idx}
                parts = custom_id.split("_")
                qid = int(parts[0][1:])
                idx = int(parts[1][1:])

                if qid not in results:
                    results[qid] = [None] * len(items)

                # Extract answer
                if data.get("response") and data["response"].get("body"):
                    content = data["response"]["body"]["choices"][0]["message"]["content"]
                    answer = "YES" in content.upper()
                    results[qid][idx] = answer
                elif data.get("error"):
                    print(f"  Error for {custom_id}: {data['error']}")
                    results[qid][idx] = False

    # Build question map for text lookup
    q_map = {q["question_id"]: q["question"] for q in questions}

    # Convert to bitmasks and save
    bitmasks_file = SCRIPT_DIR / "gpt4o_mini_bitmasks.jsonl"
    bitmasks = {}

    with open(bitmasks_file, "w") as f:
        for qid, answers in results.items():
            if None in answers:
                print(f"  Warning: Missing answers for question {qid}")
                answers = [a if a is not None else False for a in answers]

            bitmask = np.array(answers, dtype=bool)
            yes_count = np.sum(bitmask)

            record = {
                "question_id": qid,
                "question": q_map.get(qid, ""),
                "bitmask": answers,
                "yes_count": int(yes_count),
                "yes_rate": float(yes_count / len(items))
            }
            f.write(json.dumps(record) + "\n")
            bitmasks[str(qid)] = bitmask

    print(f"\nProcessed {len(bitmasks)} questions")
    print(f"Saved to {bitmasks_file}")

    # Quick stats
    yes_counts = [np.sum(b) for b in bitmasks.values()]
    print(f"\nYes count stats:")
    print(f"  Mean: {np.mean(yes_counts):.1f}")
    print(f"  Min: {min(yes_counts)}, Max: {max(yes_counts)}")
    print(f"  All-YES (128): {sum(1 for c in yes_counts if c == 128)}")
    print(f"  All-NO (0): {sum(1 for c in yes_counts if c == 0)}")

    return bitmasks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPT-4o-mini batch bitmask generation")
    parser.add_argument("--submit", action="store_true", help="Create and submit batch job")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--process", action="store_true", help="Process results into bitmasks")
    parser.add_argument("--questions", type=int, default=1000, help="Number of questions")
    args = parser.parse_args()

    print("=" * 70)
    print("GPT-4o-mini BATCH BITMASK GENERATOR")
    print("=" * 70)

    if args.submit:
        questions = load_questions(args.questions)
        items = load_items()

        # Save questions for later processing
        q_file = SCRIPT_DIR / "batch_questions.json"
        with open(q_file, "w") as f:
            json.dump(questions, f)

        batch_files = create_batch_inputs(questions, items)
        submit_batches(batch_files, len(questions))

        print("\nBatches submitted! Use --status to check progress.")
        print(f"Submitted {len(batch_files)} batches")

    elif args.status:
        check_batch_status()

    elif args.download:
        download_results()

    elif args.process:
        # Load saved questions
        q_file = SCRIPT_DIR / "batch_questions.json"
        with open(q_file) as f:
            questions = json.load(f)
        items = load_items()

        process_results(questions, items)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
