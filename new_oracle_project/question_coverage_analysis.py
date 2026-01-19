"""
Question Coverage Analysis for Trajectory Generation

This module analyzes the coverage properties of a question set
to determine how many questions are needed for reliable trajectory generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from collections import defaultdict
import json


@dataclass
class QuestionStats:
    """Statistics for a single question"""
    question_id: str
    split_ratio: float  # min(yes, no) / total
    yes_count: int
    no_count: int
    total: int
    

@dataclass
class CoverageAnalysis:
    """Results of coverage analysis"""
    total_questions: int
    split_ratio_histogram: Dict[str, int]  # bucket -> count
    bucket_definitions: Dict[str, Tuple[float, float]]  # bucket -> (min, max)
    trajectory_requirements: Dict[str, Dict]  # trajectory type -> requirements
    recommended_sample_size: int
    redundancy_factor: float
    coverage_gaps: List[str]  # buckets with insufficient questions


class QuestionCoverageAnalyzer:
    """Analyzes question set coverage for trajectory generation requirements"""
    
    def __init__(
        self,
        num_trajectories_per_type: int = 200,
        diversity_factor: float = 3.0,
        include_training_requirements: bool = True
    ):
        """
        Initialize analyzer with training parameters.
        
        Args:
            num_trajectories_per_type: Number of trajectories needed per type (train+val+test)
            diversity_factor: Multiplier for question diversity (3.0-5.0 recommended)
            include_training_requirements: If True, scale requirements for training sets
        """
        self.num_trajectories_per_type = num_trajectories_per_type
        self.diversity_factor = diversity_factor
        self.include_training_requirements = include_training_requirements
        
        # Define split ratio buckets aligned with trajectory needs
        # NOTE: split_ratio = min(yes, no) / total is symmetric
        # So we only need buckets from 0 to 0.5
        self.buckets = {
            'very_rare': (0.0, 0.05),      # T4 redundant, T2/T7 rare branches
            'rare': (0.05, 0.15),          # T2/T7 rare branches, transitions
            'skewed': (0.15, 0.35),        # T5 multi-modal, skewed splits
            'balanced': (0.35, 0.50),      # T1 baseline, general use
        }
        
        # Base trajectory type requirements (for single trajectory generation)
        self.base_trajectory_requirements = {
            'T1': {'balanced': 15},  # Smooth halving needs many balanced splits
            'T2': {'rare': 2, 'very_rare': 1, 'balanced': 10},  # Early collapse
            'T3': {'very_rare': 5, 'balanced': 8},  # Plateau then resolution
            'T4': {'very_rare': 7},  # Redundant loop - many low-IG
            'T5': {'skewed': 6, 'balanced': 5},  # Multi-modal
            'T6': {'balanced': 10},  # Prediction mismatch
            'T7': {'balanced': 8, 'rare': 2},  # Late shock
            'T8': {'balanced': 10},  # Wrong verification
        }
        
        # Calculate training requirements with diversity factor
        if include_training_requirements:
            self.trajectory_requirements = self._scale_requirements_for_training()
        else:
            self.trajectory_requirements = self.base_trajectory_requirements
    
    def _scale_requirements_for_training(self) -> Dict[str, Dict[str, int]]:
        """
        Scale base requirements for training dataset generation.
        
        For training, we don't need num_trajectories × base_requirement questions
        because trajectories can share questions. But we need enough diversity
        to avoid overfitting.
        
        Formula: required = base_requirement × diversity_factor
        
        This gives us enough questions to generate diverse trajectory subsets.
        """
        scaled = {}
        for traj_type, requirements in self.base_trajectory_requirements.items():
            scaled[traj_type] = {
                bucket: int(count * self.diversity_factor)
                for bucket, count in requirements.items()
            }
        return scaled
        
    def compute_split_ratio(self, yes_count: int, total: int) -> float:
        """Compute split ratio: min(yes, no) / total"""
        no_count = total - yes_count
        return min(yes_count, no_count) / total
    
    def classify_split(self, split_ratio: float) -> str:
        """Classify split ratio into bucket"""
        for bucket_name, (min_val, max_val) in self.buckets.items():
            if min_val <= split_ratio < max_val:
                return bucket_name
        # Split ratio is 0.5 exactly
        return 'balanced'
    
    def analyze_questions(
        self, 
        load_questions_fn: Callable[[], List[Tuple[str, np.ndarray]]]
    ) -> CoverageAnalysis:
        """
        Analyze coverage of question set.
        
        Args:
            load_questions_fn: Function that returns list of (question_id, bitmask)
                where bitmask is boolean numpy array of shape (num_items,)
        
        Returns:
            CoverageAnalysis with statistics and recommendations
        """
        print("Loading questions...")
        questions_data = load_questions_fn()
        total_questions = len(questions_data)
        
        print(f"Analyzing {total_questions} questions...")
        
        # Compute statistics for each question
        question_stats = []
        bucket_counts = defaultdict(int)
        
        for question_id, bitmask in questions_data:
            yes_count = int(bitmask.sum())
            total = len(bitmask)
            split_ratio = self.compute_split_ratio(yes_count, total)
            bucket = self.classify_split(split_ratio)
            
            question_stats.append(QuestionStats(
                question_id=question_id,
                split_ratio=split_ratio,
                yes_count=yes_count,
                no_count=total - yes_count,
                total=total
            ))
            bucket_counts[bucket] += 1
        
        # Calculate minimum requirements across all trajectory types
        min_required_per_bucket = defaultdict(int)
        for traj_type, requirements in self.trajectory_requirements.items():
            for bucket, count in requirements.items():
                min_required_per_bucket[bucket] = max(
                    min_required_per_bucket[bucket], 
                    count
                )
        
        # Identify coverage gaps
        coverage_gaps = []
        for bucket, min_required in min_required_per_bucket.items():
            if bucket_counts[bucket] < min_required:
                coverage_gaps.append(
                    f"{bucket}: have {bucket_counts[bucket]}, need {min_required}"
                )
        
        # Calculate redundancy factor
        total_min_required = sum(min_required_per_bucket.values())
        redundancy_factor = total_questions / total_min_required if total_min_required > 0 else 0
        
        # Recommend sample size with safety margin
        # Use 3x redundancy as baseline, adjusted for empirical distribution
        safety_margin = 3.0
        recommended_size = int(total_min_required * safety_margin)
        
        # Adjust if distribution is highly skewed
        smallest_bucket_count = min(bucket_counts.values()) if bucket_counts else 0
        if smallest_bucket_count < 10:
            # Need more questions if rare buckets are very sparse
            safety_margin = 5.0
            recommended_size = int(total_min_required * safety_margin)
        
        return CoverageAnalysis(
            total_questions=total_questions,
            split_ratio_histogram=dict(bucket_counts),
            bucket_definitions=self.buckets,
            trajectory_requirements=self.trajectory_requirements,
            recommended_sample_size=recommended_size,
            redundancy_factor=redundancy_factor,
            coverage_gaps=coverage_gaps
        )
    
    def print_analysis(self, analysis: CoverageAnalysis):
        """Pretty print coverage analysis results"""
        print("\n" + "="*70)
        print("QUESTION COVERAGE ANALYSIS")
        print("="*70)
        
        print(f"\nTotal questions analyzed: {analysis.total_questions:,}")
        
        if self.include_training_requirements:
            print(f"\nTRAINING PARAMETERS:")
            print(f"  Trajectories per type: {self.num_trajectories_per_type}")
            print(f"  Diversity factor: {self.diversity_factor}x")
            print(f"  (Requirements scaled for diverse training set generation)")
        else:
            print(f"\nSINGLE TRAJECTORY MODE (not scaled for training)")
        
        print(f"\nCurrent redundancy factor: {analysis.redundancy_factor:.2f}x")
        print(f"Recommended sample size: {analysis.recommended_sample_size:,}")
        
        print("\n" + "-"*70)
        print("SPLIT RATIO DISTRIBUTION")
        print("-"*70)
        print(f"{'Bucket':<15} {'Range':<20} {'Count':<10} {'% of Total':<12}")
        print("-"*70)
        
        for bucket, count in sorted(
            analysis.split_ratio_histogram.items(),
            key=lambda x: analysis.bucket_definitions[x[0]][0]
        ):
            min_val, max_val = analysis.bucket_definitions[bucket]
            pct = 100 * count / analysis.total_questions
            print(f"{bucket:<15} [{min_val:.2f}, {max_val:.2f})  {count:<10,} {pct:>6.2f}%")
        
        print("\n" + "-"*70)
        print("TRAJECTORY REQUIREMENTS")
        print("-"*70)
        
        if self.include_training_requirements:
            print(f"{'Trajectory':<12} {'Base (1x)':<35} {'Scaled ({self.diversity_factor}x)':<30}")
            print("-"*70)
            
            for traj_type in sorted(analysis.trajectory_requirements.keys()):
                scaled_reqs = analysis.trajectory_requirements[traj_type]
                base_reqs = self.base_trajectory_requirements[traj_type]
                
                base_str = ", ".join(f"{b}≥{c}" for b, c in base_reqs.items())
                scaled_str = ", ".join(f"{b}≥{c}" for b, c in scaled_reqs.items())
                
                print(f"{traj_type:<12} {base_str:<35} {scaled_str:<30}")
        else:
            print(f"{'Trajectory':<12} {'Requirements':<50}")
            print("-"*70)
            for traj_type, reqs in sorted(analysis.trajectory_requirements.items()):
                req_str = ", ".join(f"{bucket}≥{count}" for bucket, count in reqs.items())
                print(f"{traj_type:<12} {req_str}")
        
        if analysis.coverage_gaps:
            print("\n" + "-"*70)
            print("⚠️  COVERAGE GAPS DETECTED")
            print("-"*70)
            for gap in analysis.coverage_gaps:
                print(f"  • {gap}")
        else:
            print("\n✓ All trajectory requirements satisfied")
        
        print("\n" + "="*70)
    
    def estimate_coverage_probability(
        self,
        current_histogram: Dict[str, int],
        sample_size: int,
        num_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Estimate probability of meeting requirements with sample_size questions.
        
        Uses Monte Carlo simulation assuming current distribution holds.
        
        Returns:
            Dictionary of bucket -> probability of meeting minimum requirement
        """
        total = sum(current_histogram.values())
        bucket_probs = {k: v/total for k, v in current_histogram.items()}
        
        # Get minimum requirements
        min_required = defaultdict(int)
        for reqs in self.trajectory_requirements.values():
            for bucket, count in reqs.items():
                min_required[bucket] = max(min_required[bucket], count)
        
        # Monte Carlo simulation
        success_counts = defaultdict(int)
        
        for _ in range(num_simulations):
            # Sample sample_size questions according to current distribution
            sampled_counts = defaultdict(int)
            for _ in range(sample_size):
                # Sample bucket according to probability
                bucket = np.random.choice(
                    list(bucket_probs.keys()),
                    p=list(bucket_probs.values())
                )
                sampled_counts[bucket] += 1
            
            # Check if all requirements met
            for bucket, required in min_required.items():
                if sampled_counts[bucket] >= required:
                    success_counts[bucket] += 1
        
        # Calculate probabilities
        probabilities = {
            bucket: success_counts[bucket] / num_simulations
            for bucket in min_required.keys()
        }
        
        return probabilities
    
    def find_optimal_sample_size(
        self,
        current_histogram: Dict[str, int],
        target_probability: float = 0.95,
        max_size: int = 10000
    ) -> Tuple[int, Dict[str, float]]:
        """
        Find minimum sample size to achieve target_probability coverage.
        
        Returns:
            (optimal_size, probabilities_at_optimal_size)
        """
        # Binary search for optimal size
        low, high = 100, max_size
        
        while low < high:
            mid = (low + high) // 2
            probs = self.estimate_coverage_probability(
                current_histogram, mid, num_simulations=1000
            )
            
            min_prob = min(probs.values())
            
            if min_prob >= target_probability:
                high = mid
            else:
                low = mid + 1
        
        # Get final probabilities at optimal size
        final_probs = self.estimate_coverage_probability(
            current_histogram, low, num_simulations=5000
        )
        
        return low, final_probs


def save_analysis(analysis: CoverageAnalysis, filepath: str):
    """Save analysis results to JSON"""
    data = {
        'total_questions': analysis.total_questions,
        'split_ratio_histogram': analysis.split_ratio_histogram,
        'bucket_definitions': {
            k: {'min': v[0], 'max': v[1]} 
            for k, v in analysis.bucket_definitions.items()
        },
        'trajectory_requirements': analysis.trajectory_requirements,
        'recommended_sample_size': analysis.recommended_sample_size,
        'redundancy_factor': analysis.redundancy_factor,
        'coverage_gaps': analysis.coverage_gaps
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nAnalysis saved to {filepath}")


# Example usage template (user needs to implement load_questions_fn)
if __name__ == "__main__":
    
    def load_questions_example():
        """
        USER TODO: Implement this function to load your questions.
        
        Should return: List[Tuple[str, np.ndarray]]
        where each tuple is (question_id, bitmask)
        and bitmask is boolean array of shape (128,)
        """
        raise NotImplementedError(
            "You need to implement load_questions_fn to load your question data"
        )
    
    # Run analysis
    analyzer = QuestionCoverageAnalyzer()
    analysis = analyzer.analyze_questions(load_questions_example)
    analyzer.print_analysis(analysis)
    
    # Estimate probability of coverage with different sample sizes
    print("\n" + "="*70)
    print("COVERAGE PROBABILITY ESTIMATION")
    print("="*70)
    
    for sample_size in [500, 1000, 2000, 5000]:
        probs = analyzer.estimate_coverage_probability(
            analysis.split_ratio_histogram,
            sample_size,
            num_simulations=5000
        )
        min_prob = min(probs.values())
        print(f"\nWith {sample_size:,} questions:")
        print(f"  Minimum coverage probability: {min_prob:.1%}")
        print(f"  Per-bucket probabilities:")
        for bucket, prob in sorted(probs.items()):
            print(f"    {bucket:<15}: {prob:.1%}")
    
    # Find optimal size for 95% confidence
    print("\n" + "-"*70)
    optimal_size, final_probs = analyzer.find_optimal_sample_size(
        analysis.split_ratio_histogram,
        target_probability=0.95
    )
    print(f"\nOptimal sample size for 95% coverage confidence: {optimal_size:,}")
    
    # Save results
    save_analysis(analysis, "/home/claude/coverage_analysis.json")
