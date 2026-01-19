"""
Question Pool Capability Analyzer

Tests whether a question pool has sufficient CONDITIONAL coverage to support
production-scale trajectory generation (2000+ per type).

Key insight: p(q|S) = |S ∧ m_q| / |S| matters more than global p_q

Usage:
    python question_capability_analyzer.py --bitmasks questions.npy
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
import json


@dataclass
class ConditionalCoverage:
    """Coverage at a specific |S| size"""
    size: int
    balanced: int
    plateau_high: int
    plateau_low: int
    near_plateau_high: int
    near_plateau_low: int
    shock_high: int
    shock_low: int
    noop: int


class CapabilityAnalyzer:
    """Analyzes question pool for production trajectory generation"""
    
    def __init__(self, num_items: int = 128, samples_per_size: int = 100):
        self.num_items = num_items
        self.samples_per_size = samples_per_size
        self.test_sizes = [128, 64, 32, 16, 8, 4]
        
        # Trajectory requirements
        # Format: {trajectory: {stage: {range: min_count}}}
        self.requirements = {
            'T1': {
                'all': {'balanced': 15}
            },
            'T2': {
                'early': {'shock_low': 2},
                'mid': {'balanced': 5}
            },
            'T3': {
                'early': {'near_plateau_high': 3},
                'late': {'balanced': 5}
            },
            'T4': {
                'all': {'near_plateau_high': 5, 'noop': 1}
            },
            'T5': {
                'all': {'shock_low': 3, 'shock_high': 3, 'balanced': 5}
            },
            'T6': {
                'all': {'balanced': 10}
            },
            'T7': {
                'early': {'balanced': 8},
                'late': {'shock_low': 2}
            },
            'T8': {
                'early': {'balanced': 8},
                'late': {'balanced': 2, 'noop': 1}
            },
        }
        
        # Expected base pass rates (before coverage penalties)
        self.base_pass_rates = {
            'T1': 0.7, 'T2': 0.3, 'T3': 0.4, 'T4': 0.5,
            'T5': 0.6, 'T6': 0.7, 'T7': 0.5, 'T8': 0.6
        }
    
    def sample_feasible_sets(self, size: int, num_samples: int) -> List[np.ndarray]:
        """Sample random feasible sets of given size"""
        sets = []
        for _ in range(num_samples):
            indices = np.random.choice(self.num_items, size, replace=False)
            mask = np.zeros(self.num_items, dtype=bool)
            mask[indices] = True
            sets.append(mask)
        return sets
    
    def conditional_yes_rate(self, bitmask: np.ndarray, feasible_set: np.ndarray) -> float:
        """Compute p(q|S) = |S ∧ m_q| / |S|"""
        intersection = np.sum(bitmask & feasible_set)
        set_size = np.sum(feasible_set)
        return intersection / set_size if set_size > 0 else 0.0
    
    def classify_rate(self, p: float) -> List[str]:
        """Classify p(q|S) into bins"""
        bins = []
        
        if p == 0.0 or p == 1.0:
            bins.append('noop')
        if 0.98 <= p <= 1.0:
            bins.append('plateau_high')
        if 0.0 <= p <= 0.02:
            bins.append('plateau_low')
        if 0.9 <= p < 0.98:
            bins.append('near_plateau_high')
        if 0.02 < p <= 0.1:
            bins.append('near_plateau_low')
        if 0.8 <= p < 0.95:
            bins.append('shock_high')
        if 0.05 < p <= 0.2:
            bins.append('shock_low')
        if 0.45 <= p <= 0.55:
            bins.append('balanced')
        
        return bins
    
    def analyze_conditional_coverage(
        self,
        bitmasks: Dict[str, np.ndarray],
        size: int
    ) -> ConditionalCoverage:
        """Analyze coverage at given |S| size"""
        
        print(f"  Analyzing |S|={size}...")
        
        # Sample random feasible sets
        feasible_sets = self.sample_feasible_sets(size, self.samples_per_size)
        
        # Track which questions fall into each bin
        question_bins = defaultdict(set)
        
        for S in feasible_sets:
            for q_id, bitmask in bitmasks.items():
                p = self.conditional_yes_rate(bitmask, S)
                bins = self.classify_rate(p)
                
                for bin_name in bins:
                    question_bins[bin_name].add(q_id)
        
        return ConditionalCoverage(
            size=size,
            balanced=len(question_bins.get('balanced', set())),
            plateau_high=len(question_bins.get('plateau_high', set())),
            plateau_low=len(question_bins.get('plateau_low', set())),
            near_plateau_high=len(question_bins.get('near_plateau_high', set())),
            near_plateau_low=len(question_bins.get('near_plateau_low', set())),
            shock_high=len(question_bins.get('shock_high', set())),
            shock_low=len(question_bins.get('shock_low', set())),
            noop=len(question_bins.get('noop', set())),
        )
    
    def analyze(self, bitmasks: Dict[str, np.ndarray]) -> Dict:
        """Generate complete capability report"""
        
        print(f"\nAnalyzing {len(bitmasks)} questions...")
        print(f"Testing at |S| sizes: {self.test_sizes}")
        print(f"Sampling {self.samples_per_size} random sets per size\n")
        
        # 1. Conditional coverage
        print("1. Computing conditional coverage...")
        conditional = {}
        for size in self.test_sizes:
            conditional[size] = self.analyze_conditional_coverage(bitmasks, size)
        
        # 2. Trajectory feasibility
        print("\n2. Analyzing trajectory feasibility...")
        trajectory_feas = {}
        gaps = []
        
        for t_name, stage_reqs in self.requirements.items():
            bottlenecks = []
            min_coverage = 1.0
            
            # Check each stage's requirements
            for stage, range_reqs in stage_reqs.items():
                for range_name, min_needed in range_reqs.items():
                    # Check at middle size (simplified - should check all relevant sizes)
                    size = 32  # Middle size for simplicity
                    cov = conditional.get(size)
                    
                    if cov:
                        have = getattr(cov, range_name, 0)
                        coverage_ratio = have / min_needed if min_needed > 0 else 1.0
                        min_coverage = min(min_coverage, coverage_ratio)
                        
                        if have < min_needed:
                            bottlenecks.append(f"{range_name}@|S|={size}: have {have}, need {min_needed}")
                            gaps.append({
                                'trajectory': t_name,
                                'stage': stage,
                                'range': range_name,
                                'size': size,
                                'have': have,
                                'need': min_needed
                            })
            
            pass_rate = self.base_pass_rates[t_name] * min_coverage
            attempts = int(2000 / pass_rate) if pass_rate > 0 else 999999
            
            trajectory_feas[t_name] = {
                'pass_rate': pass_rate,
                'attempts_for_2k': attempts,
                'feasible': pass_rate >= 0.1,
                'bottlenecks': bottlenecks
            }
        
        return {
            'conditional_coverage': {
                size: {
                    'balanced': cov.balanced,
                    'plateau_high': cov.plateau_high,
                    'near_plateau_high': cov.near_plateau_high,
                    'shock_high': cov.shock_high,
                    'shock_low': cov.shock_low,
                    'noop': cov.noop,
                }
                for size, cov in conditional.items()
            },
            'trajectory_feasibility': trajectory_feas,
            'gaps': gaps
        }
    
    def print_report(self, report: Dict):
        """Print formatted report"""
        
        print("\n" + "="*100)
        print("CAPABILITY REPORT")
        print("="*100)
        
        # Conditional coverage
        print("\nCONDITIONAL COVERAGE AT DIFFERENT |S|")
        print("-"*100)
        print(f"{'|S|':<6} {'Balanced':<12} {'Plateau':<12} {'Shock':<12} {'No-op':<12}")
        print("-"*100)
        
        for size in sorted(report['conditional_coverage'].keys(), reverse=True):
            cov = report['conditional_coverage'][size]
            plateau = cov['plateau_high']
            shock = cov['shock_high'] + cov['shock_low']
            
            print(f"{size:<6} {cov['balanced']:<12,} {plateau:<12,} {shock:<12,} {cov['noop']:<12,}")
        
        # Trajectory feasibility
        print("\nTRAJECTORY FEASIBILITY (2000 per type)")
        print("-"*100)
        print(f"{'Type':<6} {'Pass Rate':<12} {'Attempts':<15} {'Feasible':<10} {'Bottlenecks'}")
        print("-"*100)
        
        for t_name in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            feas = report['trajectory_feasibility'][t_name]
            status = "✓ YES" if feas['feasible'] else "❌ NO"
            bottleneck = feas['bottlenecks'][0] if feas['bottlenecks'] else "None"
            
            print(f"{t_name:<6} {feas['pass_rate']:>7.1%}{' '*4} {feas['attempts_for_2k']:>10,}{' '*4} "
                  f"{status:<10} {bottleneck}")
        
        # Gaps
        if report['gaps']:
            print("\nCOVERAGE GAPS")
            print("-"*100)
            for gap in report['gaps'][:10]:
                print(f"  {gap['trajectory']} {gap['stage']}: needs {gap['need']} {gap['range']}@|S|={gap['size']}, "
                      f"have {gap['have']}")
        
        # Summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        feasible = sum(1 for f in report['trajectory_feasibility'].values() if f['feasible'])
        
        if feasible == 8:
            print("\n✓✓✓ All 8 trajectory types are production-ready")
        elif feasible >= 6:
            print(f"\n✓✓ {feasible}/8 trajectory types are production-ready")
        else:
            print(f"\n⚠️  Only {feasible}/8 trajectory types are production-ready")
        
        print("="*100)


def load_bitmasks(filepath: str) -> Dict[str, np.ndarray]:
    """
    USER TODO: Load your question bitmasks
    
    Returns:
        Dict mapping question_id -> boolean array (128 items)
    """
    raise NotImplementedError("Implement bitmask loading")


def main():
    print("="*100)
    print("QUESTION POOL CAPABILITY ANALYZER")
    print("="*100)
    
    # Load bitmasks
    bitmasks = load_bitmasks("questions.npy")
    
    # Run analysis
    analyzer = CapabilityAnalyzer(num_items=128, samples_per_size=100)
    report = analyzer.analyze(bitmasks)
    
    # Print
    analyzer.print_report(report)
    
    # Save
    with open('/mnt/user-data/outputs/capability_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved")


if __name__ == "__main__":
    main()
