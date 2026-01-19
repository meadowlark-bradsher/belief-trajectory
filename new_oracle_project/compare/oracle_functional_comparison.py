"""
Oracle Functional Comparison for Trajectory Generation

Tests which LLM oracle provides the best FUNCTIONAL COVERAGE for trajectory generation,
not which one agrees most with CUQ.

Key Question: "Can I generate all 8 trajectory types with this oracle?"

Priority Metrics:
1. Useless question rate (<5% critical)
2. Coverage for all trajectory types (T1-T8)
3. Self-consistency (deterministic answers)
4. Cost per useful question
5. Distribution shape (has some of each bucket)

CUQ agreement is NOT a primary concern - we're building a new oracle world.
"""

import numpy as np
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import sys
sys.path.append('/home/claude')

# Import our coverage analysis toolkit
from question_coverage_analysis import QuestionCoverageAnalyzer


@dataclass
class FunctionalCoverageResult:
    """Results focused on functional trajectory generation capability"""
    model_name: str
    
    # CRITICAL METRICS
    useless_rate: float  # % questions with no information (all YES/NO)
    coverage_feasible: bool  # Can generate all trajectory types?
    optimal_sample_size: int  # Questions needed for 95% coverage
    self_consistency_score: float  # % same answers when queried twice
    
    # Distribution metrics
    very_rare_pct: float
    rare_pct: float
    skewed_pct: float
    balanced_pct: float
    
    # Problem detection
    all_yes_count: int
    all_no_count: int
    pathological_questions: List[str]  # Questions with extreme behavior
    
    # Cost metrics
    cost_per_1k_questions: float
    cost_per_1k_useful: float  # Accounting for waste
    cost_for_optimal_coverage: float
    
    # Coverage gaps
    coverage_gaps: List[str]  # Missing trajectory requirements
    
    # Quality score (functional, not CUQ-relative)
    functional_quality: float  # 0-1 scale


class FunctionalOracleComparator:
    """Compare oracles based on trajectory generation capability"""
    
    def __init__(self, num_items: int = 128, diversity_factor: float = 50.0):
        self.num_items = num_items
        self.diversity_factor = diversity_factor
        
        # Model costs (per 1M tokens): (input, output, has_batch, batch_discount)
        self.model_configs = {
            'gemini_flash_1.5': (75, 300, False, 1.0),
            'gemini_flash_2.0': (100, 400, False, 1.0),
            'gpt4o_mini': (150, 600, True, 0.5),
            'claude_sonnet_4': (3000, 15000, True, 0.5),
        }
        
        # Initialize coverage analyzer
        self.coverage_analyzer = QuestionCoverageAnalyzer(
            num_trajectories_per_type=200,
            diversity_factor=diversity_factor,
            include_training_requirements=True
        )
    
    def compute_split_ratio(self, bitmask: np.ndarray) -> float:
        """Compute split ratio: min(yes, no) / total"""
        yes_count = np.sum(bitmask)
        no_count = len(bitmask) - yes_count
        return min(yes_count, no_count) / len(bitmask)
    
    def is_useless(self, bitmask: np.ndarray) -> bool:
        """Check if question provides no information"""
        yes_count = np.sum(bitmask)
        return yes_count == 0 or yes_count == len(bitmask)
    
    def classify_split(self, split_ratio: float) -> str:
        """Classify split into bucket"""
        if split_ratio < 0.05:
            return 'very_rare'
        elif split_ratio < 0.15:
            return 'rare'
        elif split_ratio < 0.35:
            return 'skewed'
        else:
            return 'balanced'
    
    def estimate_cost(self, model_name: str, num_questions: int, use_batch: bool = True) -> float:
        """Estimate cost for generating bitmasks"""
        if model_name not in self.model_configs:
            return 0.0
        
        input_cost_per_1m, output_cost_per_1m, has_batch, batch_discount = self.model_configs[model_name]
        
        # Tokens per call
        input_tokens = 75
        output_tokens = 10
        
        total_calls = num_questions * self.num_items
        
        input_cost = (total_calls * input_tokens * input_cost_per_1m) / 1_000_000
        output_cost = (total_calls * output_tokens * output_cost_per_1m) / 1_000_000
        
        total_cost = input_cost + output_cost
        
        if use_batch and has_batch:
            total_cost *= batch_discount
        
        return total_cost
    
    def test_self_consistency(
        self,
        query_function: callable,
        sample_questions: List[Dict],
        items: List[str],
        num_repeats: int = 2
    ) -> float:
        """
        Test if oracle gives consistent answers when queried multiple times.
        
        Returns:
            Consistency score (0-1): % of question-item pairs with consistent answers
        """
        print(f"    Testing self-consistency ({num_repeats} repeats on {len(sample_questions)} questions)...")
        
        # Query each question multiple times
        all_results = []
        
        for repeat in range(num_repeats):
            print(f"      Repeat {repeat + 1}/{num_repeats}...")
            results = query_function(sample_questions, items)
            all_results.append(results)
        
        # Compare results
        agreements = []
        
        for qid in all_results[0].keys():
            # Get bitmasks from all repeats
            bitmasks = [results[qid] for results in all_results]
            
            # Check if all repeats agree
            for i in range(len(bitmasks)):
                for j in range(i + 1, len(bitmasks)):
                    agreement = np.mean(bitmasks[i] == bitmasks[j])
                    agreements.append(agreement)
        
        return np.mean(agreements)
    
    def analyze_functional_coverage(
        self,
        model_name: str,
        model_bitmasks: Dict[str, np.ndarray],
        test_questions: List[Dict],
        self_consistency_score: float
    ) -> FunctionalCoverageResult:
        """
        Analyze oracle's functional capability for trajectory generation.
        """
        print(f"    Analyzing functional coverage...")
        
        # 1. Detect useless questions
        useless_count = 0
        all_yes_count = 0
        all_no_count = 0
        pathological = []
        
        for qid, bitmask in model_bitmasks.items():
            if self.is_useless(bitmask):
                useless_count += 1
                yes_count = np.sum(bitmask)
                
                if yes_count == len(bitmask):
                    all_yes_count += 1
                else:
                    all_no_count += 1
                
                # Find the question text
                q_text = next((q['text'] for q in test_questions if q['id'] == qid), 'unknown')
                pathological.append(f"{qid}: '{q_text}' ({yes_count}/{len(bitmask)} YES)")
        
        useless_rate = (useless_count / len(model_bitmasks)) * 100
        
        # 2. Compute split distribution
        split_ratios = [self.compute_split_ratio(bitmask) for bitmask in model_bitmasks.values()]
        split_buckets = [self.classify_split(sr) for sr in split_ratios]
        bucket_counts = Counter(split_buckets)
        total = len(split_ratios)
        
        very_rare_pct = (bucket_counts.get('very_rare', 0) / total) * 100
        rare_pct = (bucket_counts.get('rare', 0) / total) * 100
        skewed_pct = (bucket_counts.get('skewed', 0) / total) * 100
        balanced_pct = (bucket_counts.get('balanced', 0) / total) * 100
        
        # 3. Run coverage analysis
        print(f"    Running coverage analysis...")
        
        # Convert to format expected by coverage analyzer
        question_data = [
            (qid, bitmask) 
            for qid, bitmask in model_bitmasks.items()
        ]
        
        coverage_analysis = self.coverage_analyzer.analyze_questions(lambda: question_data)
        
        # 4. Check if coverage is feasible
        coverage_feasible = len(coverage_analysis.coverage_gaps) == 0
        optimal_sample_size = coverage_analysis.recommended_sample_size
        coverage_gaps = coverage_analysis.coverage_gaps
        
        # 5. Cost analysis
        cost_per_1k = self.estimate_cost(model_name, 1000, use_batch=True)
        
        # Cost per useful question (accounting for waste)
        useful_fraction = 1.0 - (useless_rate / 100.0)
        cost_per_1k_useful = cost_per_1k / useful_fraction if useful_fraction > 0 else float('inf')
        
        # Cost for optimal coverage
        cost_for_optimal = self.estimate_cost(model_name, optimal_sample_size, use_batch=True)
        
        # 6. Functional quality score (0-1 scale)
        # Heavily weight useless rate and coverage feasibility
        quality_components = {
            'low_useless': max(0, 1.0 - useless_rate / 20.0),  # 30% weight
            'coverage_ok': 1.0 if coverage_feasible else 0.0,  # 30% weight
            'self_consistent': self_consistency_score,          # 20% weight
            'has_balanced': min(balanced_pct / 3.0, 1.0),      # 10% weight
            'has_very_rare': min(very_rare_pct / 3.0, 1.0),    # 10% weight
        }
        
        functional_quality = (
            0.30 * quality_components['low_useless'] +
            0.30 * quality_components['coverage_ok'] +
            0.20 * quality_components['self_consistent'] +
            0.10 * quality_components['has_balanced'] +
            0.10 * quality_components['has_very_rare']
        )
        
        return FunctionalCoverageResult(
            model_name=model_name,
            useless_rate=useless_rate,
            coverage_feasible=coverage_feasible,
            optimal_sample_size=optimal_sample_size,
            self_consistency_score=self_consistency_score,
            very_rare_pct=very_rare_pct,
            rare_pct=rare_pct,
            skewed_pct=skewed_pct,
            balanced_pct=balanced_pct,
            all_yes_count=all_yes_count,
            all_no_count=all_no_count,
            pathological_questions=pathological[:10],  # Top 10 worst
            cost_per_1k_questions=cost_per_1k,
            cost_per_1k_useful=cost_per_1k_useful,
            cost_for_optimal_coverage=cost_for_optimal,
            coverage_gaps=coverage_gaps,
            functional_quality=functional_quality
        )
    
    def print_comparison(self, results: List[FunctionalCoverageResult]):
        """Print functional comparison results"""
        
        print("\n" + "="*100)
        print("FUNCTIONAL ORACLE COMPARISON")
        print("="*100)
        print("\nFocus: Can this oracle generate all 8 trajectory types?")
        print("NOT comparing to CUQ - we're building a new oracle world!\n")
        
        # Sort by functional quality
        results = sorted(results, key=lambda r: r.functional_quality, reverse=True)
        
        # 1. CRITICAL: Useless Rate
        print("-"*100)
        print("1. USELESS QUESTION RATE (CRITICAL)")
        print("-"*100)
        print(f"{'Model':<20} {'Useless %':<12} {'All YES':<12} {'All NO':<12} {'Status'}")
        print("-"*100)
        
        for r in results:
            if r.useless_rate < 5:
                status = "✓✓✓ EXCELLENT"
            elif r.useless_rate < 10:
                status = "✓ ACCEPTABLE"
            elif r.useless_rate < 15:
                status = "⚠️ CONCERNING"
            else:
                status = "❌ DISQUALIFYING"
            
            print(f"{r.model_name:<20} {r.useless_rate:>6.1f}%{' '*5} "
                  f"{r.all_yes_count:<12} {r.all_no_count:<12} {status}")
        
        # 2. Coverage Feasibility
        print("\n" + "-"*100)
        print("2. TRAJECTORY GENERATION COVERAGE")
        print("-"*100)
        print(f"{'Model':<20} {'Can Generate T1-T8?':<25} {'Optimal Size':<15} {'Coverage Gaps'}")
        print("-"*100)
        
        for r in results:
            feasible = "✓ YES" if r.coverage_feasible else "❌ NO"
            gaps = len(r.coverage_gaps)
            gap_str = f"{gaps} gaps" if gaps > 0 else "None"
            
            print(f"{r.model_name:<20} {feasible:<25} {r.optimal_sample_size:<15,} {gap_str}")
            
            if r.coverage_gaps:
                for gap in r.coverage_gaps[:3]:  # Show first 3
                    print(f"{'':20}   • {gap}")
        
        # 3. Self-Consistency
        print("\n" + "-"*100)
        print("3. SELF-CONSISTENCY (Deterministic Answers)")
        print("-"*100)
        print(f"{'Model':<20} {'Consistency':<15} {'Status'}")
        print("-"*100)
        
        for r in results:
            if r.self_consistency_score > 0.95:
                status = "✓✓✓ EXCELLENT - Highly deterministic"
            elif r.self_consistency_score > 0.90:
                status = "✓✓ GOOD - Mostly consistent"
            elif r.self_consistency_score > 0.85:
                status = "✓ ACCEPTABLE - Some variance"
            else:
                status = "⚠️ PROBLEMATIC - High variance"
            
            print(f"{r.model_name:<20} {r.self_consistency_score:>6.1%}{' '*8} {status}")
        
        # 4. Distribution Shape
        print("\n" + "-"*100)
        print("4. SPLIT DISTRIBUTION (Not comparing to CUQ - just checking shape)")
        print("-"*100)
        print(f"{'Model':<20} {'Very Rare':<12} {'Rare':<12} {'Skewed':<12} {'Balanced':<12}")
        print("-"*100)
        
        for r in results:
            print(f"{r.model_name:<20} {r.very_rare_pct:>6.1f}%{' '*5} "
                  f"{r.rare_pct:>6.1f}%{' '*5} {r.skewed_pct:>6.1f}%{' '*5} "
                  f"{r.balanced_pct:>6.1f}%")
        
        # Check for pathological distributions
        for r in results:
            warnings = []
            if r.very_rare_pct < 1:
                warnings.append("⚠️ Very few very_rare questions")
            if r.balanced_pct < 2:
                warnings.append("⚠️ Very few balanced questions")
            if r.very_rare_pct > 50:
                warnings.append("⚠️ Too many very_rare questions")
            
            if warnings:
                print(f"{r.model_name}:")
                for w in warnings:
                    print(f"  {w}")
        
        # 5. Cost Effectiveness
        print("\n" + "-"*100)
        print("5. COST ANALYSIS (Accounting for waste)")
        print("-"*100)
        print(f"{'Model':<20} {'$/1k Questions':<18} {'$/1k USEFUL':<18} {'Optimal Coverage Cost'}")
        print("-"*100)
        
        for r in results:
            waste_indicator = ""
            if r.cost_per_1k_useful > r.cost_per_1k_questions * 1.2:
                waste_indicator = "⚠️"
            
            print(f"{r.model_name:<20} ${r.cost_per_1k_questions:>7.2f}{' '*10} "
                  f"${r.cost_per_1k_useful:>7.2f} {waste_indicator}{' '*8} "
                  f"${r.cost_for_optimal_coverage:>7.2f}")
        
        # 6. Overall Ranking
        print("\n" + "-"*100)
        print("6. FUNCTIONAL QUALITY RANKING")
        print("-"*100)
        print(f"{'Rank':<6} {'Model':<20} {'Score':<10} {'Assessment'}")
        print("-"*100)
        
        for i, r in enumerate(results, 1):
            if i == 1:
                assessment = "✓✓✓ BEST FOR TRAJECTORY GENERATION"
            elif r.functional_quality > 0.80:
                assessment = "✓✓ EXCELLENT - Highly functional"
            elif r.functional_quality > 0.70:
                assessment = "✓ GOOD - Will work"
            elif r.functional_quality > 0.60:
                assessment = "⚠️ MARGINAL - Consider alternatives"
            else:
                assessment = "❌ POOR - Not recommended"
            
            print(f"{i:<6} {r.model_name:<20} {r.functional_quality:>6.3f}    {assessment}")
        
        # RECOMMENDATION
        print("\n" + "="*100)
        print("🏆 RECOMMENDATION")
        print("="*100)
        
        winner = results[0]
        
        print(f"\nRecommended Model: {winner.model_name}")
        print(f"Functional Quality: {winner.functional_quality:.3f}")
        
        print(f"\n✓ Key Strengths:")
        
        if winner.useless_rate < 5:
            print(f"  • Low waste: Only {winner.useless_rate:.1f}% useless questions")
        
        if winner.coverage_feasible:
            print(f"  • Complete coverage: Can generate all trajectory types")
            print(f"  • Needs {winner.optimal_sample_size:,} questions for 95% confidence")
        
        if winner.self_consistency_score > 0.95:
            print(f"  • Highly consistent: {winner.self_consistency_score:.1%} deterministic")
        
        print(f"  • Cost: ${winner.cost_for_optimal_coverage:.2f} for optimal coverage")
        
        # Warnings
        print(f"\n⚠️ Watch Out For:")
        
        if winner.useless_rate > 5:
            waste_cost = winner.cost_for_optimal_coverage * (winner.useless_rate / 100)
            print(f"  • {winner.useless_rate:.1f}% useless questions (~${waste_cost:.2f} wasted)")
            print(f"    → Consider prompt refinement or post-generation filtering")
        
        if not winner.coverage_feasible:
            print(f"  • Coverage gaps detected:")
            for gap in winner.coverage_gaps:
                print(f"      {gap}")
            print(f"    → May need to generate more questions or bias toward missing buckets")
        
        if winner.self_consistency_score < 0.90:
            print(f"  • Moderate consistency: {winner.self_consistency_score:.1%}")
            print(f"    → May need to set temperature=0 or use majority voting")
        
        # Pathological questions
        if winner.pathological_questions:
            print(f"\n📋 Examples of Problematic Questions:")
            for pq in winner.pathological_questions[:5]:
                print(f"  • {pq}")
            print(f"  → These will be filtered out during generation")
        
        # Budget guidance
        print(f"\n💰 Budget Guidance:")
        print(f"  For {winner.diversity_factor}x diversity (~{winner.optimal_sample_size:,} questions):")
        print(f"    • Generation cost: ${winner.cost_for_optimal_coverage:.2f}")
        
        if winner.useless_rate > 0:
            useful_count = int(winner.optimal_sample_size * (1 - winner.useless_rate / 100))
            print(f"    • Useful questions: ~{useful_count:,} ({100 - winner.useless_rate:.1f}%)")
            print(f"    • Effective cost: ${winner.cost_per_1k_useful:.2f} per 1k useful questions")
        
        print(f"\n📊 Next Steps:")
        print(f"  1. Generate {winner.optimal_sample_size:,} questions with {winner.model_name}")
        print(f"  2. Filter out useless questions (expect ~{int(winner.optimal_sample_size * winner.useless_rate / 100):,} filtered)")
        print(f"  3. Run consistency check on generated bitmasks")
        print(f"  4. Verify coverage for all trajectory types")
        print(f"  5. Generate training trajectories")
        
        print("\n" + "="*100)
    
    def save_results(self, results: List[FunctionalCoverageResult], output_file: str):
        """Save results to JSON"""
        output_data = {
            'recommendation': results[0].model_name if results else None,
            'focus': 'functional_coverage_not_cuq_agreement',
            'models': []
        }
        
        for r in results:
            model_data = asdict(r)
            # Convert lists that might be too long
            if len(model_data['pathological_questions']) > 10:
                model_data['pathological_questions'] = model_data['pathological_questions'][:10]
            output_data['models'].append(model_data)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


# =============================================================================
# USER IMPLEMENTATION REQUIRED
# =============================================================================

def load_test_questions(questions_file: str, num_questions: int = 1000) -> Tuple[List[Dict], List[str]]:
    """
    Load test questions and items.
    
    USER TODO: Implement loading your questions.
    
    Returns:
        questions: List of dicts with 'id', 'text' fields
        items: List of 128 item names
    """
    raise NotImplementedError("Implement your question loading logic")


def query_oracle_with_consistency_test(
    model_name: str,
    questions: List[Dict],
    items: List[str]
) -> Dict[str, np.ndarray]:
    """
    Query oracle model to generate bitmasks.
    
    USER TODO: Implement API calling for your chosen models.
    
    This will be called multiple times for consistency testing.
    """
    raise NotImplementedError(f"Implement API calling for {model_name}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Run functional oracle comparison"""
    
    print("="*100)
    print("FUNCTIONAL ORACLE COMPARISON")
    print("="*100)
    print("\nGoal: Find which oracle can generate all 8 trajectory types")
    print("NOT finding which oracle matches CUQ the best\n")
    
    # Configuration
    TEST_SIZE = 1000  # Larger test for coverage analysis
    DIVERSITY_FACTOR = 50.0  # Target diversity
    CONSISTENCY_REPEATS = 2  # How many times to test same questions
    
    print(f"Configuration:")
    print(f"  Test size: {TEST_SIZE} questions")
    print(f"  Diversity factor: {DIVERSITY_FACTOR}x")
    print(f"  Consistency repeats: {CONSISTENCY_REPEATS}")
    
    # Load test questions
    print(f"\nLoading test data...")
    questions, items = load_test_questions("test_questions.json", TEST_SIZE)
    
    print(f"✓ Loaded {len(questions)} questions and {len(items)} items")
    
    # Initialize comparator
    comparator = FunctionalOracleComparator(
        num_items=len(items),
        diversity_factor=DIVERSITY_FACTOR
    )
    
    # Test models
    models_to_test = ['gemini_flash_2.0', 'gpt4o_mini', 'claude_sonnet_4']
    results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*100}")
        print(f"Testing {model_name}")
        print(f"{'='*100}")
        
        # Estimate cost
        est_cost = comparator.estimate_cost(model_name, len(questions), use_batch=False)
        print(f"  Estimated test cost: ${est_cost:.2f}")
        
        try:
            # 1. Test self-consistency
            print(f"\n  Step 1: Testing self-consistency...")
            
            # Sample 50 questions for consistency test (cheaper)
            import random
            consistency_sample = random.sample(questions, min(50, len(questions)))
            
            def query_fn(qs, itms):
                return query_oracle_with_consistency_test(model_name, qs, itms)
            
            consistency_score = comparator.test_self_consistency(
                query_fn,
                consistency_sample,
                items,
                CONSISTENCY_REPEATS
            )
            
            print(f"    ✓ Consistency: {consistency_score:.1%}")
            
            # 2. Generate full test set
            print(f"\n  Step 2: Generating bitmasks for {len(questions)} questions...")
            model_bitmasks = query_oracle_with_consistency_test(model_name, questions, items)
            
            print(f"    ✓ Generated {len(model_bitmasks)} bitmasks")
            
            # 3. Analyze functional coverage
            print(f"\n  Step 3: Analyzing functional coverage...")
            result = comparator.analyze_functional_coverage(
                model_name,
                model_bitmasks,
                questions,
                consistency_score
            )
            
            results.append(result)
            
            print(f"\n  ✓ Results:")
            print(f"    Useless rate: {result.useless_rate:.1f}%")
            print(f"    Coverage feasible: {'YES' if result.coverage_feasible else 'NO'}")
            print(f"    Optimal size: {result.optimal_sample_size:,}")
            print(f"    Functional quality: {result.functional_quality:.3f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\n✗ No models completed testing")
        return
    
    # Print comparison
    comparator.print_comparison(results)
    
    # Save results
    comparator.save_results(results, '/mnt/user-data/outputs/oracle_functional_comparison.json')
    
    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
