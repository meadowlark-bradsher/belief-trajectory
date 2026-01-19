#!/usr/bin/env python3
"""
Question Pool Capability Analyzer

Tests conditional coverage at different |S| sizes to determine if a question pool
can support production-scale trajectory generation (2000+ per type).

Key insight: p(q|S) = |S ∧ m_q| / |S| matters more than global distribution.
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"


@dataclass
class ConditionalCoverage:
    """Coverage at a specific |S| size"""
    size: int
    balanced: int      # 0.40-0.60
    plateau_high: int  # 0.98-1.00
    plateau_low: int   # 0.00-0.02
    near_plateau_high: int  # 0.90-0.98
    near_plateau_low: int   # 0.02-0.10
    shock_high: int    # 0.80-0.90
    shock_low: int     # 0.10-0.20
    noop: int          # exactly 0.0 or 1.0


class CapabilityAnalyzer:
    """Analyzes question pool for production trajectory generation"""

    def __init__(self, num_items: int = 128, samples_per_size: int = 100):
        self.num_items = num_items
        self.samples_per_size = samples_per_size
        self.test_sizes = [128, 64, 32, 16, 8, 4]

        # Trajectory requirements per stage
        self.requirements = {
            'T1': {'all': {'balanced': 15}},
            'T2': {'early': {'shock_low': 2}, 'mid': {'balanced': 5}},
            'T3': {'early': {'near_plateau_high': 3}, 'late': {'balanced': 5}},
            'T4': {'all': {'near_plateau_high': 5, 'noop': 1}},
            'T5': {'all': {'shock_low': 3, 'shock_high': 3, 'balanced': 5}},
            'T6': {'all': {'balanced': 10}},
            'T7': {'early': {'balanced': 8}, 'late': {'shock_low': 2}},
            'T8': {'early': {'balanced': 8}, 'late': {'balanced': 2, 'noop': 1}},
        }

        # Base pass rates (before coverage penalties)
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
        if 0.90 <= p < 0.98:
            bins.append('near_plateau_high')
        if 0.02 < p <= 0.10:
            bins.append('near_plateau_low')
        if 0.80 <= p < 0.90:
            bins.append('shock_high')
        if 0.10 < p <= 0.20:
            bins.append('shock_low')
        if 0.40 <= p <= 0.60:
            bins.append('balanced')

        return bins

    def analyze_conditional_coverage(
        self,
        bitmasks: Dict[str, np.ndarray],
        size: int
    ) -> ConditionalCoverage:
        """Analyze coverage at given |S| size"""

        feasible_sets = self.sample_feasible_sets(size, self.samples_per_size)
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

        print(f"Analyzing {len(bitmasks)} questions...")
        print(f"Testing at |S| sizes: {self.test_sizes}")
        print(f"Sampling {self.samples_per_size} random sets per size\n")

        # 1. Conditional coverage at each size
        print("Computing conditional coverage...")
        conditional = {}
        for size in self.test_sizes:
            print(f"  |S|={size}...", end=" ", flush=True)
            conditional[size] = self.analyze_conditional_coverage(bitmasks, size)
            print(f"balanced={conditional[size].balanced}, noop={conditional[size].noop}")

        # 2. Trajectory feasibility
        print("\nAnalyzing trajectory feasibility...")
        trajectory_feas = {}
        gaps = []

        for t_name, stage_reqs in self.requirements.items():
            bottlenecks = []
            min_coverage = 1.0

            for stage, range_reqs in stage_reqs.items():
                for range_name, min_needed in range_reqs.items():
                    # Check at representative size
                    size = 32 if stage == 'all' else (64 if stage == 'early' else 16)
                    cov = conditional.get(size)

                    if cov:
                        have = getattr(cov, range_name, 0)
                        coverage_ratio = have / min_needed if min_needed > 0 else 1.0
                        min_coverage = min(min_coverage, coverage_ratio)

                        if have < min_needed:
                            bottlenecks.append(f"{range_name}@|S|={size}: have {have}, need {min_needed}")
                            gaps.append({
                                'trajectory': t_name, 'stage': stage,
                                'range': range_name, 'size': size,
                                'have': have, 'need': min_needed
                            })

            pass_rate = self.base_pass_rates[t_name] * min(min_coverage, 1.0)
            attempts = int(2000 / pass_rate) if pass_rate > 0 else 999999

            trajectory_feas[t_name] = {
                'pass_rate': pass_rate,
                'attempts_for_2k': attempts,
                'feasible': pass_rate >= 0.10,
                'bottlenecks': bottlenecks
            }

        # 3. Pool quality score
        pass_rates = [f['pass_rate'] for f in trajectory_feas.values()]
        avg_pass = np.mean(pass_rates)
        min_pass = min(pass_rates)
        coverage_complete = 1.0 - len(gaps) / 20  # Normalize by max expected gaps

        quality_score = 0.40 * avg_pass + 0.40 * min_pass + 0.20 * max(coverage_complete, 0)

        return {
            'total_questions': len(bitmasks),
            'conditional_coverage': {
                size: asdict(cov) for size, cov in conditional.items()
            },
            'trajectory_feasibility': trajectory_feas,
            'gaps': gaps,
            'quality_score': quality_score,
            'summary': {
                'avg_pass_rate': avg_pass,
                'min_pass_rate': min_pass,
                'feasible_types': sum(1 for f in trajectory_feas.values() if f['feasible']),
                'critical_gaps': len([g for g in gaps if g['have'] < g['need'] * 0.5])
            }
        }

    def print_report(self, report: Dict):
        """Print formatted report"""

        print("\n" + "=" * 90)
        print("CAPABILITY REPORT")
        print("=" * 90)

        print(f"\nTotal questions: {report['total_questions']:,}")
        print(f"Quality score: {report['quality_score']:.3f}")

        # Conditional coverage
        print("\nCONDITIONAL COVERAGE AT DIFFERENT |S|")
        print("-" * 90)
        print(f"{'|S|':<6} {'Balanced':<10} {'Plateau':<10} {'Near-Plat':<12} {'Shock':<10} {'No-op':<10}")
        print("-" * 90)

        for size in sorted(report['conditional_coverage'].keys(), reverse=True):
            cov = report['conditional_coverage'][size]
            plateau = cov['plateau_high'] + cov['plateau_low']
            near_plat = cov['near_plateau_high'] + cov['near_plateau_low']
            shock = cov['shock_high'] + cov['shock_low']

            print(f"{size:<6} {cov['balanced']:<10} {plateau:<10} {near_plat:<12} {shock:<10} {cov['noop']:<10}")

        # Trajectory feasibility
        print("\nTRAJECTORY FEASIBILITY (target: 2000 per type)")
        print("-" * 90)
        print(f"{'Type':<6} {'Pass Rate':<12} {'Attempts':<12} {'Status':<12} {'Bottleneck'}")
        print("-" * 90)

        for t_name in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            feas = report['trajectory_feasibility'][t_name]

            if feas['pass_rate'] >= 0.50:
                status = "✓✓✓ Fast"
            elif feas['pass_rate'] >= 0.20:
                status = "✓✓ Good"
            elif feas['pass_rate'] >= 0.10:
                status = "✓ Slow"
            else:
                status = "❌ Problem"

            bottleneck = feas['bottlenecks'][0][:35] if feas['bottlenecks'] else "None"

            print(f"{t_name:<6} {feas['pass_rate']:>7.1%}{' '*4} {feas['attempts_for_2k']:>8,}{' '*3} "
                  f"{status:<12} {bottleneck}")

        # Gaps
        if report['gaps']:
            print("\nCOVERAGE GAPS")
            print("-" * 90)
            for gap in report['gaps'][:8]:
                deficit = gap['need'] - gap['have']
                print(f"  {gap['trajectory']} {gap['stage']}: {gap['range']}@|S|={gap['size']} "
                      f"— have {gap['have']}, need {gap['need']} (deficit: {deficit})")

        # Summary
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)

        s = report['summary']
        print(f"\nFeasible trajectory types: {s['feasible_types']}/8")
        print(f"Average pass rate: {s['avg_pass_rate']:.1%}")
        print(f"Minimum pass rate: {s['min_pass_rate']:.1%}")
        print(f"Critical gaps: {s['critical_gaps']}")

        if s['feasible_types'] == 8 and s['min_pass_rate'] >= 0.20:
            print("\n✓✓✓ Pool is PRODUCTION-READY")
        elif s['feasible_types'] >= 6:
            print("\n✓✓ Pool is USABLE with targeted generation")
        else:
            print("\n⚠️ Pool needs significant augmentation")

        print("=" * 90)


def load_bitmasks_from_jsonl() -> Dict[str, np.ndarray]:
    """Load bitmasks from CUQ questions.jsonl"""
    questions_path = DATA_DIR / "questions.jsonl"

    bitmasks = {}
    with open(questions_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                qid = str(data["question_id"])
                # Convert int bitmask to numpy array
                bitmask_int = data["bitmask"]
                binary_str = format(bitmask_int, '0128b')[::-1]
                bitmask = np.array([c == '1' for c in binary_str], dtype=bool)
                bitmasks[qid] = bitmask

    return bitmasks


def load_bitmasks_from_generated(filepath: str) -> Dict[str, np.ndarray]:
    """Load bitmasks from generated JSONL file"""
    bitmasks = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                qid = str(data.get("question_id", data.get("id")))

                if "bitmask" in data:
                    if isinstance(data["bitmask"], int):
                        binary_str = format(data["bitmask"], '0128b')[::-1]
                        bitmask = np.array([c == '1' for c in binary_str], dtype=bool)
                    else:
                        bitmask = np.array(data["bitmask"], dtype=bool)
                elif "gemini_responses" in data:
                    bitmask = np.array(data["gemini_responses"], dtype=bool)
                else:
                    continue

                bitmasks[qid] = bitmask

    return bitmasks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze question pool capability")
    parser.add_argument("--source", choices=["cuq", "generated"], default="cuq",
                       help="Source of bitmasks")
    parser.add_argument("--file", type=str, help="Path to generated bitmasks file")
    parser.add_argument("--samples", type=int, default=100,
                       help="Samples per |S| size")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of questions (for testing)")
    args = parser.parse_args()

    print("=" * 90)
    print("QUESTION POOL CAPABILITY ANALYZER")
    print("=" * 90)

    # Load bitmasks
    if args.source == "cuq":
        print("\nLoading CUQ bitmasks...")
        bitmasks = load_bitmasks_from_jsonl()
    else:
        print(f"\nLoading generated bitmasks from {args.file}...")
        bitmasks = load_bitmasks_from_generated(args.file)

    if args.limit:
        import random
        keys = random.sample(list(bitmasks.keys()), min(args.limit, len(bitmasks)))
        bitmasks = {k: bitmasks[k] for k in keys}

    print(f"Loaded {len(bitmasks):,} questions")

    # Run analysis
    analyzer = CapabilityAnalyzer(num_items=128, samples_per_size=args.samples)
    report = analyzer.analyze(bitmasks)

    # Print
    analyzer.print_report(report)

    # Save
    output_file = SCRIPT_DIR / "capability_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {output_file}")


if __name__ == "__main__":
    main()
