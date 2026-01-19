"""
Test Example with Synthetic Data

This demonstrates the expected format and runs the analyzer on synthetic data.
Use this to verify the pipeline works before wiring up your real data.
"""

import numpy as np
from typing import List, Tuple
from question_coverage_analysis import QuestionCoverageAnalyzer, save_analysis


def generate_synthetic_questions(
    num_items: int = 128,
    num_questions: int = 2000
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate synthetic questions with realistic split ratio distribution.
    
    This simulates what a real question set might look like.
    """
    questions = []
    
    np.random.seed(42)
    
    for i in range(num_questions):
        question_id = f"synthetic_q_{i:05d}"
        
        # Generate split ratio with realistic distribution
        # NOTE: split_ratio = min(yes, no) / total is symmetric
        # So we only need to generate probabilities, not worry about sides
        split_type = np.random.choice(
            ['balanced', 'skewed', 'rare', 'very_rare'],
            p=[0.5, 0.3, 0.15, 0.05]  # Most balanced, few rare
        )
        
        if split_type == 'balanced':
            # Target split_ratio 0.35-0.50
            # This means min(yes, no) should be 35-50% of total
            # So yes_prob should be either 0.35-0.50 or 0.50-0.65
            target_split = np.random.uniform(0.35, 0.50)
            # Randomly choose which side is minority
            if np.random.rand() < 0.5:
                yes_prob = target_split
            else:
                yes_prob = 1.0 - target_split
                
        elif split_type == 'skewed':
            # Target split_ratio 0.15-0.35
            target_split = np.random.uniform(0.15, 0.35)
            if np.random.rand() < 0.5:
                yes_prob = target_split
            else:
                yes_prob = 1.0 - target_split
                
        elif split_type == 'rare':
            # Target split_ratio 0.05-0.15
            target_split = np.random.uniform(0.05, 0.15)
            if np.random.rand() < 0.5:
                yes_prob = target_split
            else:
                yes_prob = 1.0 - target_split
                
        else:  # very_rare
            # Target split_ratio 0.0-0.05
            target_split = np.random.uniform(0.0, 0.05)
            if np.random.rand() < 0.5:
                yes_prob = target_split
            else:
                yes_prob = 1.0 - target_split
        
        # Generate bitmask
        bitmask = np.random.rand(num_items) < yes_prob
        
        questions.append((question_id, bitmask))
    
    return questions


def test_with_synthetic_data():
    """Run analysis on synthetic data to verify everything works"""
    
    print("="*70)
    print("TESTING WITH SYNTHETIC DATA")
    print("="*70)
    print("\nGenerating 2000 synthetic questions for 128 items...")
    
    questions = generate_synthetic_questions(num_items=128, num_questions=2000)
    
    print(f"Generated {len(questions)} questions")
    print(f"First question: {questions[0][0]}")
    print(f"  Bitmask shape: {questions[0][1].shape}")
    print(f"  Bitmask dtype: {questions[0][1].dtype}")
    print(f"  Yes count: {questions[0][1].sum()}")
    print(f"  No count: {(~questions[0][1]).sum()}")
    
    # Run analysis
    analyzer = QuestionCoverageAnalyzer(
        num_trajectories_per_type=200,
        diversity_factor=3.0,
        include_training_requirements=True
    )
    analysis = analyzer.analyze_questions(lambda: questions)
    
    analyzer.print_analysis(analysis)
    
    # Also test single-trajectory mode for comparison
    print("\n" + "="*70)
    print("COMPARISON: SINGLE TRAJECTORY MODE")
    print("="*70)
    
    analyzer_single = QuestionCoverageAnalyzer(
        num_trajectories_per_type=1,
        diversity_factor=1.0,
        include_training_requirements=False
    )
    analysis_single = analyzer_single.analyze_questions(lambda: questions)
    analyzer_single.print_analysis(analysis_single)
    
    # Test probability estimation
    print("\n" + "="*70)
    print("TESTING COVERAGE PROBABILITY ESTIMATION")
    print("="*70)
    
    for sample_size in [500, 1000, 2000]:
        probs = analyzer.estimate_coverage_probability(
            analysis.split_ratio_histogram,
            sample_size,
            num_simulations=1000  # Fewer sims for test
        )
        min_prob = min(probs.values())
        print(f"\nWith {sample_size:,} questions: min_coverage={min_prob:.1%}")
    
    # Save test results
    save_analysis(analysis, "/home/claude/test_coverage_analysis.json")
    
    print("\n✓ Test complete! Now implement your real data loader in run_analysis.py")


if __name__ == "__main__":
    test_with_synthetic_data()
