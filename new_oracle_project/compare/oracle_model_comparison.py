"""
Oracle Model Comparison Script

Tests multiple LLM models as oracle replacements for CUQ (T5-XL) oracle.
Compares agreement rates, split distributions, and identifies problematic patterns.

Example Usage:
    # Test 200 questions across 3 models
    python oracle_model_comparison.py
    
    # Results show:
    # - Which model has best CUQ agreement
    # - Split ratio distribution comparison
    # - Problematic pattern detection (128/128 YES, etc.)
    # - Cost analysis
    # - Recommendation
"""

import numpy as np
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path


@dataclass
class OracleTestResult:
    """Results from testing one oracle model"""
    model_name: str
    
    # Agreement metrics
    agreement_with_cuq: float
    agreement_std: float
    min_agreement: float
    max_agreement: float
    avg_jaccard: float
    
    # Split ratio distribution (matches coverage analysis buckets)
    very_rare_pct: float  # [0, 0.05)
    rare_pct: float       # [0.05, 0.15)
    skewed_pct: float     # [0.15, 0.35)
    balanced_pct: float   # [0.35, 0.50]
    
    # Problematic patterns
    all_yes_count: int    # Questions with 128/128 YES
    all_no_count: int     # Questions with 0/128 YES
    useless_pct: float    # % questions with no information
    
    # Cost
    cost_per_1k_questions: float
    total_test_cost: float
    
    # Quality score (composite metric: 0-1 scale)
    quality_score: float


class OracleModelComparator:
    """Compare different LLM models as oracle replacements"""
    
    def __init__(self, num_items: int = 128):
        self.num_items = num_items
        
        # Model pricing (per 1M tokens)
        # Format: (input_cost, output_cost, has_batch, batch_discount)
        self.model_configs = {
            'gemini_flash_1.5': (75, 300, False, 1.0),
            'gemini_flash_2.0': (100, 400, False, 1.0),
            'gpt4o_mini': (150, 600, True, 0.5),
            'claude_sonnet_4': (3000, 15000, True, 0.5),
        }
    
    def compute_split_ratio(self, bitmask: np.ndarray) -> float:
        """Compute split ratio: min(yes, no) / total"""
        yes_count = np.sum(bitmask)
        no_count = len(bitmask) - yes_count
        return min(yes_count, no_count) / len(bitmask)
    
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
    
    def compute_jaccard(self, bitmask1: np.ndarray, bitmask2: np.ndarray) -> float:
        """Compute Jaccard similarity"""
        intersection = np.sum(bitmask1 & bitmask2)
        union = np.sum(bitmask1 | bitmask2)
        return intersection / union if union > 0 else 0.0
    
    def estimate_cost(self, model_name: str, num_questions: int, use_batch: bool = True) -> float:
        """Estimate cost for generating bitmasks"""
        if model_name not in self.model_configs:
            return 0.0
        
        input_cost_per_1m, output_cost_per_1m, has_batch, batch_discount = self.model_configs[model_name]
        
        # Tokens per call
        input_tokens = 75  # prompt + question + item
        output_tokens = 10  # "yes" or "no"
        
        total_calls = num_questions * self.num_items
        
        input_cost = (total_calls * input_tokens * input_cost_per_1m) / 1_000_000
        output_cost = (total_calls * output_tokens * output_cost_per_1m) / 1_000_000
        
        total_cost = input_cost + output_cost
        
        # Apply batch discount if available and requested
        if use_batch and has_batch:
            total_cost *= batch_discount
        
        return total_cost
    
    def analyze_model_results(
        self,
        model_name: str,
        model_bitmasks: Dict[str, np.ndarray],
        cuq_bitmasks: Dict[str, np.ndarray],
        num_test_questions: int
    ) -> OracleTestResult:
        """Analyze results from one model"""
        
        # Compute per-question metrics
        agreements = []
        jaccards = []
        split_ratios = []
        
        all_yes_count = 0
        all_no_count = 0
        
        for qid in model_bitmasks.keys():
            if qid not in cuq_bitmasks:
                continue
            
            model_mask = model_bitmasks[qid]
            cuq_mask = cuq_bitmasks[qid]
            
            # Agreement rate
            agreement = np.mean(model_mask == cuq_mask)
            agreements.append(agreement)
            
            # Jaccard similarity
            jaccard = self.compute_jaccard(model_mask, cuq_mask)
            jaccards.append(jaccard)
            
            # Split ratio
            split_ratio = self.compute_split_ratio(model_mask)
            split_ratios.append(split_ratio)
            
            # Check for useless questions
            yes_count = np.sum(model_mask)
            if yes_count == self.num_items:
                all_yes_count += 1
            elif yes_count == 0:
                all_no_count += 1
        
        # Aggregate metrics
        avg_agreement = np.mean(agreements)
        agreement_std = np.std(agreements)
        min_agreement = np.min(agreements)
        max_agreement = np.max(agreements)
        avg_jaccard = np.mean(jaccards)
        
        # Split distribution
        split_buckets = [self.classify_split(sr) for sr in split_ratios]
        bucket_counts = Counter(split_buckets)
        total = len(split_ratios)
        
        very_rare_pct = (bucket_counts.get('very_rare', 0) / total) * 100
        rare_pct = (bucket_counts.get('rare', 0) / total) * 100
        skewed_pct = (bucket_counts.get('skewed', 0) / total) * 100
        balanced_pct = (bucket_counts.get('balanced', 0) / total) * 100
        
        # Useless questions
        useless_count = all_yes_count + all_no_count
        useless_pct = (useless_count / total) * 100
        
        # Cost estimates
        cost_per_1k = self.estimate_cost(model_name, 1000, use_batch=True)
        test_cost = self.estimate_cost(model_name, num_test_questions, use_batch=False)
        
        # Quality score (0-1 scale)
        # Weights: agreement (40%), balanced bucket availability (20%), 
        #          low useless rate (20%), jaccard (20%)
        quality_score = (
            0.40 * avg_agreement +
            0.20 * min(balanced_pct / 4.4, 1.0) +  # Normalize to CUQ baseline (4.4%)
            0.20 * (1.0 - min(useless_pct / 100.0, 1.0)) +
            0.20 * avg_jaccard
        )
        
        return OracleTestResult(
            model_name=model_name,
            agreement_with_cuq=avg_agreement,
            agreement_std=agreement_std,
            min_agreement=min_agreement,
            max_agreement=max_agreement,
            avg_jaccard=avg_jaccard,
            very_rare_pct=very_rare_pct,
            rare_pct=rare_pct,
            skewed_pct=skewed_pct,
            balanced_pct=balanced_pct,
            all_yes_count=all_yes_count,
            all_no_count=all_no_count,
            useless_pct=useless_pct,
            cost_per_1k_questions=cost_per_1k,
            total_test_cost=test_cost,
            quality_score=quality_score
        )
    
    def print_comparison(self, results: List[OracleTestResult], cuq_baseline: Dict = None):
        """Print formatted comparison table"""
        
        print("\n" + "="*100)
        print("ORACLE MODEL COMPARISON RESULTS")
        print("="*100)
        
        # Sort by quality score
        results = sorted(results, key=lambda r: r.quality_score, reverse=True)
        
        # Agreement metrics
        print("\n" + "-"*100)
        print("1. AGREEMENT WITH CUQ ORACLE")
        print("-"*100)
        print(f"{'Model':<20} {'Avg':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Jaccard':<10}")
        print("-"*100)
        
        for r in results:
            print(f"{r.model_name:<20} {r.agreement_with_cuq:>7.1%}   "
                  f"{r.agreement_std:>7.3f}   {r.min_agreement:>7.1%}   "
                  f"{r.max_agreement:>7.1%}   {r.avg_jaccard:>7.3f}")
        
        # Split distribution
        print("\n" + "-"*100)
        print("2. SPLIT RATIO DISTRIBUTION")
        print("-"*100)
        
        if cuq_baseline:
            print(f"{'Model':<20} {'Very Rare':<15} {'Rare':<15} {'Skewed':<15} {'Balanced':<15}")
            print(f"{'CUQ Baseline':<20} {cuq_baseline.get('very_rare_pct', 0):>6.1f}%{' '*8} "
                  f"{cuq_baseline.get('rare_pct', 0):>6.1f}%{' '*8} "
                  f"{cuq_baseline.get('skewed_pct', 0):>6.1f}%{' '*8} "
                  f"{cuq_baseline.get('balanced_pct', 4.4):>6.1f}%")
            print("-"*100)
        
        for r in results:
            print(f"{r.model_name:<20} {r.very_rare_pct:>6.1f}%{' '*8} "
                  f"{r.rare_pct:>6.1f}%{' '*8} {r.skewed_pct:>6.1f}%{' '*8} "
                  f"{r.balanced_pct:>6.1f}%")
        
        # Problematic patterns
        print("\n" + "-"*100)
        print("3. PROBLEMATIC PATTERNS")
        print("-"*100)
        print(f"{'Model':<20} {'All YES':<15} {'All NO':<15} {'Useless %':<15} {'Status'}")
        print("-"*100)
        
        for r in results:
            status = "✓ Good" if r.useless_pct < 5 else "⚠️ High" if r.useless_pct < 15 else "❌ Critical"
            print(f"{r.model_name:<20} {r.all_yes_count:<15} {r.all_no_count:<15} "
                  f"{r.useless_pct:>6.1f}%{' '*8} {status}")
        
        # Cost
        print("\n" + "-"*100)
        print("4. COST ANALYSIS (Batch API pricing)")
        print("-"*100)
        print(f"{'Model':<20} {'Test Cost':<15} {'Per 1k Qs':<15} {'Per 50k Qs':<15}")
        print("-"*100)
        
        for r in results:
            print(f"{r.model_name:<20} ${r.total_test_cost:>7.2f}{' '*7} "
                  f"${r.cost_per_1k_questions:>7.2f}{' '*7} "
                  f"${r.cost_per_1k_questions * 50:>7.2f}")
        
        # Overall ranking
        print("\n" + "-"*100)
        print("5. OVERALL QUALITY RANKING")
        print("-"*100)
        print(f"{'Rank':<6} {'Model':<20} {'Score':<10} {'Assessment':<50}")
        print("-"*100)
        
        best_score = results[0].quality_score
        
        for i, r in enumerate(results, 1):
            score_pct = (r.quality_score / best_score) * 100 if best_score > 0 else 0
            
            if i == 1:
                assessment = "✓✓✓ BEST CHOICE"
            elif score_pct >= 95:
                assessment = "✓✓ EXCELLENT - Very close to best"
            elif score_pct >= 85:
                assessment = "✓ GOOD - Acceptable alternative"
            else:
                assessment = "⚠️ SUBOPTIMAL - Consider carefully"
            
            print(f"{i:<6} {r.model_name:<20} {r.quality_score:>6.3f}    {assessment}")
        
        # Recommendation
        print("\n" + "="*100)
        print("🏆 RECOMMENDATION")
        print("="*100)
        
        winner = results[0]
        
        print(f"\nRecommended Model: {winner.model_name}")
        print(f"\nKey Metrics:")
        print(f"  • Quality Score: {winner.quality_score:.3f}")
        print(f"  • CUQ Agreement: {winner.agreement_with_cuq:.1%} (avg), "
              f"{winner.min_agreement:.1%}-{winner.max_agreement:.1%} range")
        print(f"  • Jaccard Similarity: {winner.avg_jaccard:.3f}")
        print(f"  • Balanced Splits: {winner.balanced_pct:.1f}% "
              f"(CUQ baseline: {cuq_baseline.get('balanced_pct', 4.4):.1f}%)")
        print(f"  • Useless Questions: {winner.useless_pct:.1f}%")
        print(f"  • Cost: ${winner.cost_per_1k_questions:.2f} per 1k questions")
        
        print(f"\nReasoning:")
        
        if winner.agreement_with_cuq >= 0.85:
            print(f"  ✓ High agreement with CUQ ({winner.agreement_with_cuq:.1%})")
        else:
            print(f"  ⚠️ Moderate agreement with CUQ ({winner.agreement_with_cuq:.1%})")
        
        if winner.balanced_pct >= 4.0:
            print(f"  ✓ Good balanced split coverage ({winner.balanced_pct:.1f}%)")
        else:
            print(f"  ⚠️ Lower balanced splits than ideal ({winner.balanced_pct:.1f}%)")
        
        if winner.useless_pct < 5:
            print(f"  ✓ Low useless question rate ({winner.useless_pct:.1f}%)")
        else:
            print(f"  ⚠️ Notable useless questions ({winner.useless_pct:.1f}%)")
        
        print(f"  ✓ Cost-effective at ${winner.cost_per_1k_questions:.2f} per 1k questions")
        
        # Warnings
        if winner.useless_pct > 10:
            print(f"\n⚠️ ACTION REQUIRED:")
            print(f"   {winner.useless_pct:.1f}% of questions return all-YES or all-NO")
            print(f"   Recommendations:")
            print(f"   1. Refine prompt: Add 'Answer NO unless clearly YES' instruction")
            print(f"   2. Filter post-generation: Remove questions with extreme splits")
            print(f"   3. Consider alternative model with lower useless rate")
        
        if winner.balanced_pct < 3.0:
            print(f"\n⚠️ WARNING:")
            print(f"   Only {winner.balanced_pct:.1f}% balanced splits (CUQ: 4.4%)")
            print(f"   This may cause issues generating T1/T6/T7/T8 trajectories")
            print(f"   Consider generating extra questions or biasing toward balanced splits")
        
        # Alternative recommendations
        if len(results) > 1:
            runner_up = results[1]
            score_diff = (winner.quality_score - runner_up.quality_score) * 100
            
            if score_diff < 5:
                print(f"\nAlternative: {runner_up.model_name}")
                print(f"  • Only {score_diff:.1f} points behind in quality score")
                
                cost_diff = winner.cost_per_1k_questions - runner_up.cost_per_1k_questions
                if abs(cost_diff) > 1:
                    if cost_diff > 0:
                        print(f"  • ${abs(cost_diff):.2f} cheaper per 1k questions")
                    else:
                        print(f"  • ${abs(cost_diff):.2f} more expensive per 1k questions")
        
        print("\n" + "="*100)
    
    def save_results(self, results: List[OracleTestResult], output_file: str):
        """Save results to JSON"""
        output_data = {
            'recommendation': results[0].model_name if results else None,
            'models': [asdict(r) for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


# =============================================================================
# USER IMPLEMENTATION REQUIRED
# =============================================================================

def load_test_data(items_file: str, questions_file: str, cuq_bitmasks_file: str):
    """
    Load test data for oracle comparison.
    
    USER TODO: Implement loading your actual data format.
    
    Returns:
        items: List of item names ['Airplane', 'Banana', ...]
        questions: List of dicts with 'id', 'text' fields
        cuq_bitmasks: Dict mapping question_id -> np.ndarray(128, dtype=bool)
    """
    raise NotImplementedError(
        "Implement your data loading logic here.\n"
        "Load items, questions, and CUQ bitmasks from your files."
    )


def query_oracle(model_name: str, questions: List[Dict], items: List[str]) -> Dict[str, np.ndarray]:
    """
    Query an oracle model to generate bitmasks.
    
    USER TODO: Implement your API calling logic for each model.
    
    Args:
        model_name: One of ['gemini_flash_1.5', 'gpt4o_mini', 'claude_sonnet_4']
        questions: List of question dicts
        items: List of item names
    
    Returns:
        Dict mapping question_id -> bitmask (np.ndarray, shape=(128,), dtype=bool)
    """
    raise NotImplementedError(
        f"Implement API calling for {model_name}.\n"
        "For each (question, item) pair, call model API and get yes/no answer.\n"
        "Return dict of bitmasks."
    )


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Run oracle model comparison"""
    
    print("="*100)
    print("ORACLE MODEL COMPARISON SCRIPT")
    print("="*100)
    print("\nThis script tests different LLM models as oracle replacements for CUQ.")
    print("It compares agreement rates, split distributions, and costs.")
    
    # Configuration
    ITEMS_FILE = "/mnt/user-data/uploads/items.txt"
    QUESTIONS_FILE = "test_questions.json"  # Your test questions
    CUQ_BITMASKS_FILE = "cuq_bitmasks.npy"  # Your CUQ ground truth
    TEST_SIZE = 200  # Number of questions to test
    
    print(f"\nConfiguration:")
    print(f"  Test size: {TEST_SIZE} questions")
    print(f"  Items: {ITEMS_FILE}")
    print(f"  Questions: {QUESTIONS_FILE}")
    print(f"  CUQ bitmasks: {CUQ_BITMASKS_FILE}")
    
    # Initialize
    comparator = OracleModelComparator(num_items=128)
    
    # Load data
    print(f"\nLoading test data...")
    items, questions, cuq_bitmasks = load_test_data(ITEMS_FILE, QUESTIONS_FILE, CUQ_BITMASKS_FILE)
    
    # Sample if needed
    if len(questions) > TEST_SIZE:
        import random
        random.seed(42)
        questions = random.sample(questions, TEST_SIZE)
    
    print(f"✓ Loaded {len(items)} items and {len(questions)} questions")
    
    # Test models
    models_to_test = ['gemini_flash_2.0', 'gpt4o_mini', 'claude_sonnet_4']
    results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*100}")
        print(f"Testing {model_name}...")
        print(f"{'='*100}")
        
        # Estimate cost
        est_cost = comparator.estimate_cost(model_name, len(questions), use_batch=False)
        print(f"Estimated cost: ${est_cost:.2f} ({len(questions)} questions × 128 items)")
        
        # Query oracle
        print(f"Querying {model_name} oracle...")
        model_bitmasks = query_oracle(model_name, questions, items)
        
        # Analyze
        result = comparator.analyze_model_results(
            model_name,
            model_bitmasks,
            cuq_bitmasks,
            len(questions)
        )
        
        results.append(result)
        
        print(f"\n✓ Complete:")
        print(f"  Agreement: {result.agreement_with_cuq:.1%}")
        print(f"  Balanced: {result.balanced_pct:.1f}%")
        print(f"  Useless: {result.useless_pct:.1f}%")
        print(f"  Quality score: {result.quality_score:.3f}")
    
    # Print comparison
    cuq_baseline = {
        'very_rare_pct': 0,  # Fill from your CUQ analysis
        'rare_pct': 0,
        'skewed_pct': 0,
        'balanced_pct': 4.4,
    }
    
    comparator.print_comparison(results, cuq_baseline)
    
    # Save results
    comparator.save_results(results, '/mnt/user-data/outputs/oracle_comparison_results.json')
    
    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
