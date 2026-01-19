"""
Question Coverage Analysis for CUQ Dataset

Analyzes the 122k CUQ questions to determine how many need to be
regenerated with a new oracle for trajectory generation.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from question_coverage_analysis import QuestionCoverageAnalyzer, save_analysis


# Path to CUQ dataset (relative to this file or absolute)
CUQ_DATA_DIR = Path(__file__).parent.parent / "data"


def int_to_bitmask(bitmask_int: int, num_bits: int = 128) -> np.ndarray:
    """Convert integer bitmask to boolean numpy array (vectorized)."""
    # Convert to binary string, pad to num_bits, reverse (LSB first), convert to bool array
    binary_str = format(bitmask_int, f'0{num_bits}b')[::-1]
    return np.array([c == '1' for c in binary_str], dtype=bool)


def load_questions() -> List[Tuple[str, np.ndarray]]:
    """
    Load questions from CUQ dataset (questions.jsonl).

    Returns:
        List of (question_id, bitmask) tuples where:
        - question_id: str - unique identifier for the question
        - bitmask: np.ndarray - boolean array of shape (128,)
    """
    questions_path = CUQ_DATA_DIR / "questions.jsonl"

    if not questions_path.exists():
        raise FileNotFoundError(
            f"CUQ questions not found at {questions_path}\n"
            f"Please ensure data/questions.jsonl exists."
        )

    results = []
    with open(questions_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            question_id = str(data["question_id"])
            bitmask = int_to_bitmask(data["bitmask"])
            results.append((question_id, bitmask))

    return results


def main():
    """Run the complete coverage analysis"""
    
    print("Starting question coverage analysis...")
    print("Loading questions from your data source...")
    
    # ====================================================================
    # CONFIGURATION: Adjust these parameters for your use case
    # ====================================================================
    
    # How many trajectories do you need per type (T1-T8)?
    # Typical breakdown: 150 train + 25 val + 25 test = 200 total
    NUM_TRAJECTORIES_PER_TYPE = 200
    
    # Diversity factor: How much variety do you want?
    # 1.0x = Minimal (all trajectories nearly identical) - NOT RECOMMENDED for training
    # 3.0x = Moderate diversity (some question overlap okay) - MINIMUM for training
    # 5.0x = High diversity (preferred for robust training)
    # 10.0x = Maximum diversity (expensive, may be overkill)
    DIVERSITY_FACTOR = 3.0
    
    print(f"\nConfiguration:")
    print(f"  Trajectories per type: {NUM_TRAJECTORIES_PER_TYPE}")
    print(f"  Diversity factor: {DIVERSITY_FACTOR}x")
    
    # Initialize analyzer with training parameters
    analyzer = QuestionCoverageAnalyzer(
        num_trajectories_per_type=NUM_TRAJECTORIES_PER_TYPE,
        diversity_factor=DIVERSITY_FACTOR,
        include_training_requirements=True  # Set False for single trajectory analysis
    )
    
    # Run analysis
    analysis = analyzer.analyze_questions(load_questions)
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Run Monte Carlo simulations for different sample sizes
    print("\n" + "="*70)
    print("COVERAGE PROBABILITY ESTIMATION")
    print("="*70)
    print("\nRunning Monte Carlo simulations...")
    
    sample_sizes = [500, 1000, 2000, 5000]  # Reduced for faster runs
    
    for sample_size in sample_sizes:
        probs = analyzer.estimate_coverage_probability(
            analysis.split_ratio_histogram,
            sample_size,
            num_simulations=1000  # Reduced from 5000 for faster runs
        )
        min_prob = min(probs.values())
        avg_prob = np.mean(list(probs.values()))
        
        print(f"\n{sample_size:>5,} questions: "
              f"min_coverage={min_prob:.1%}, avg_coverage={avg_prob:.1%}")
    
    # Find optimal size
    print("\n" + "-"*70)
    print("Finding optimal sample size for 95% coverage confidence...")
    
    optimal_size, final_probs = analyzer.find_optimal_sample_size(
        analysis.split_ratio_histogram,
        target_probability=0.95
    )
    
    print(f"\n✓ Optimal sample size: {optimal_size:,} questions")
    print(f"  Coverage probabilities at this size:")
    for bucket, prob in sorted(final_probs.items()):
        status = "✓" if prob >= 0.95 else "⚠️"
        print(f"    {status} {bucket:<15}: {prob:.1%}")
    
    # ====================================================================
    # DIVERSITY FACTOR IMPACT ANALYSIS
    # ====================================================================
    
    print("\n" + "="*70)
    print("DIVERSITY FACTOR COMPARISON")
    print("="*70)
    print("\nHow different diversity factors affect requirements:\n")
    
    comparison_factors = [1.0, 2.0, 3.0, 5.0, 10.0]
    print(f"{'Factor':<8} {'Example: T4 very_rare':<25} {'Total Min Required':<20} {'Use Case'}")
    print("-"*70)
    
    use_cases = [
        "Single trajectory",
        "Minimal diversity (risky)",
        "Standard (MINIMUM)",
        "Robust (PREFERRED)",
        "Maximum diversity"
    ]
    
    for factor, use_case in zip(comparison_factors, use_cases):
        # T4 requires 7 very_rare questions base
        t4_requirement = int(7 * factor)
        
        # Calculate total minimum across all trajectory types
        temp_analyzer = QuestionCoverageAnalyzer(
            num_trajectories_per_type=NUM_TRAJECTORIES_PER_TYPE,
            diversity_factor=factor,
            include_training_requirements=True
        )
        
        total_min = 0
        for traj_reqs in temp_analyzer.trajectory_requirements.values():
            total_min = max(total_min, sum(traj_reqs.values()))
        
        print(f"{factor:<8.1f}x {t4_requirement} questions{'':<13} {total_min:<20,} {use_case}")
    
    print(f"\nYour setting: {DIVERSITY_FACTOR}x diversity")
    
    if DIVERSITY_FACTOR < 3.0:
        print("⚠️  WARNING: Diversity < 3.0x not recommended for training!")
        print("   Trajectories will be too similar → overfitting risk.")
    elif DIVERSITY_FACTOR >= 5.0:
        print("✓ Excellent diversity for robust training.")
    else:
        print("✓ Acceptable diversity. Consider 5.0x for even better results.")
    
    # Calculate cost estimates
    print("\n" + "="*70)
    print("COST ESTIMATES")
    print("="*70)
    
    num_items = 128
    tokens_per_call = 85  # 75 input + 10 output average
    
    # Format: (model_name, input_cost_per_1k, output_cost_per_1k, has_batch_api)
    models = [
        ("Gemini Flash 1.5", 0.000075, 0.0003, False),
        ("Gemini Flash 2.0", 0.0001, 0.0004, False),
        ("GPT-4o mini", 0.00015, 0.0006, True),
        ("GPT-4o mini (Batch)", 0.000075, 0.0003, True),  # 50% discount
        ("Sonnet 4.5", 0.003, 0.015, True),
        ("Sonnet 4.5 (Batch)", 0.0015, 0.0075, True),  # 50% discount
    ]
    
    print(f"\nFor {optimal_size:,} questions × {num_items} items = "
          f"{optimal_size * num_items:,} API calls:\n")
    
    print(f"{'Model':<25} {'Cost':<12} {'Notes'}")
    print("-"*70)
    
    for model_name, input_cost_1k, output_cost_1k, has_batch in models:
        input_tokens = optimal_size * num_items * 75
        output_tokens = optimal_size * num_items * 10
        
        cost = (input_tokens * input_cost_1k / 1000 + 
                output_tokens * output_cost_1k / 1000)
        
        notes = ""
        if "(Batch)" in model_name:
            notes = "50% off, async"
        elif has_batch:
            notes = "Batch available"
        
        print(f"  {model_name:<23} ${cost:>8,.2f}   {notes}")
    
    print("\n" + "-"*70)
    print("BATCH API RECOMMENDATION")
    print("-"*70)
    print("\nBatch API gives 50% discount but is asynchronous (hours delay).")
    print("For oracle regeneration, this is usually acceptable.\n")
    
    # Calculate what diversity factor you could afford with batch
    current_cost = optimal_size * num_items * 85 * 0.000075 / 1000  # Gemini Flash
    batch_cost = current_cost * 0.5
    
    print(f"Your current config ({DIVERSITY_FACTOR}x diversity):")
    print(f"  Sync cost (Gemini Flash): ${current_cost:.2f}")
    print(f"  Batch cost (GPT-4o mini): ${batch_cost:.2f}")
    
    # What diversity could you afford for same price?
    affordable_diversity = DIVERSITY_FACTOR * 2  # Since batch is 50% off
    
    print(f"\nFor the SAME ${current_cost:.2f} budget with Batch API:")
    print(f"  You could afford {affordable_diversity:.1f}x diversity!")
    print(f"  → {int(affordable_diversity / DIVERSITY_FACTOR)}x more question variety")
    print(f"  → Better generalization for same cost")
    
    if DIVERSITY_FACTOR < 5.0:
        print(f"\n💡 SUGGESTION: Consider using Batch API with {min(10.0, affordable_diversity):.1f}x diversity")
        print(f"   Same cost, {affordable_diversity / DIVERSITY_FACTOR:.1f}x better quality")
    
    print("\nBatch API Tradeoffs:")
    print("  ✓ 50% cheaper → 2x more diversity for same price")
    print("  ✓ Perfect for upfront oracle generation (not interactive)")
    print("  ✗ Async processing (hours delay, not real-time)")
    print("  ✗ Less control over rate limits")
    
    # Save results
    output_path = Path(__file__).parent / "coverage_analysis.json"
    save_analysis(analysis, str(output_path))
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
