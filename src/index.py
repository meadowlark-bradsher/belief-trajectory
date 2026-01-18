"""Question indexing for efficient split-based lookup.

This module provides efficient ways to find questions based on their
split characteristics on a given feasible set state.
"""

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .bitmask import (
    popcount,
    split_ratio,
    information_gain,
    bitmask_overlap,
    get_yes_set,
    get_no_set,
)
from .loader import CUQDataset, Question


@dataclass
class QuestionCandidate:
    """A question with its computed characteristics for a given state."""
    question: Question
    split_ratio: float  # Proportion answering YES
    information_gain: float
    yes_count: int
    no_count: int


# Popcount bucket boundaries for fast lookup
POPCOUNT_BUCKETS = {
    'plateau_low': (0, 6),      # 0-5% YES → almost always low-IG
    'near_plateau_low': (7, 16),  # 5-12% YES → usually low-IG
    'low': (17, 32),            # 13-25% YES
    'balanced': (33, 95),       # 26-74% YES → good for balanced splits
    'high': (96, 111),          # 75-87% YES
    'near_plateau_high': (112, 121),  # 88-95% YES → usually low-IG
    'plateau_high': (122, 128), # 95-100% YES → almost always low-IG
}


class QuestionIndex:
    """Index for efficient question lookup by split characteristics.

    This class provides methods to find questions matching various
    split criteria on a given feasible set state.

    Optimizations:
    - Precomputed popcount buckets for O(bucket_size) instead of O(n) searches
    - LRU cache for compute_candidates to avoid redundant work on similar states
    """

    def __init__(
        self,
        dataset: CUQDataset,
        seed: Optional[int] = None
    ):
        """Initialize the question index.

        Args:
            dataset: The CUQ dataset with questions
            seed: Optional random seed for reproducibility
        """
        self.dataset = dataset
        self.questions = dataset.questions
        self.rng = random.Random(seed)

        # Build popcount buckets for fast lookup (Fix B)
        self._build_popcount_buckets()

        # Cache for compute_candidates (Fix C)
        self._candidate_cache: dict[int, list[QuestionCandidate]] = {}
        self._cache_max_size = 10

    def _build_popcount_buckets(self):
        """Precompute global popcount for each question and bucket them."""
        self.question_popcount: dict[int, int] = {}
        self.popcount_buckets: dict[str, list[Question]] = {
            name: [] for name in POPCOUNT_BUCKETS
        }

        for q in self.questions:
            pc = popcount(q.bitmask)
            self.question_popcount[q.question_id] = pc

            # Add to appropriate bucket(s)
            for bucket_name, (lo, hi) in POPCOUNT_BUCKETS.items():
                if lo <= pc <= hi:
                    self.popcount_buckets[bucket_name].append(q)

        # Also create combined buckets for common use cases
        self.low_popcount_questions = (
            self.popcount_buckets['plateau_low'] +
            self.popcount_buckets['near_plateau_low']
        )
        self.high_popcount_questions = (
            self.popcount_buckets['plateau_high'] +
            self.popcount_buckets['near_plateau_high']
        )
        self.balanced_questions = self.popcount_buckets['balanced']

    def _get_cached_candidates(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None
    ) -> Optional[list[QuestionCandidate]]:
        """Check cache for precomputed candidates."""
        if state in self._candidate_cache:
            cached = self._candidate_cache[state]
            if exclude_ids:
                return [c for c in cached if c.question.question_id not in exclude_ids]
            return cached
        return None

    def _cache_candidates(self, state: int, candidates: list[QuestionCandidate]):
        """Store candidates in cache with LRU eviction."""
        if len(self._candidate_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._candidate_cache))
            del self._candidate_cache[oldest_key]
        self._candidate_cache[state] = candidates

    def clear_cache(self):
        """Clear the candidate cache (call between trajectories)."""
        self._candidate_cache.clear()

    def compute_candidates(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        use_cache: bool = True
    ) -> list[QuestionCandidate]:
        """Compute characteristics for all questions on a state.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude (already asked)
            use_cache: Whether to use/update the cache (Fix C)

        Returns:
            List of QuestionCandidate objects
        """
        # Check cache first (Fix C)
        if use_cache:
            cached = self._get_cached_candidates(state, exclude_ids)
            if cached is not None:
                return cached

        exclude_ids = exclude_ids or set()
        total = popcount(state)
        candidates = []

        for q in self.questions:
            if q.question_id in exclude_ids:
                continue

            yes_count = popcount(get_yes_set(state, q.bitmask))
            no_count = total - yes_count

            # Skip questions that don't split the state
            if yes_count == 0 or no_count == 0:
                continue

            ratio = yes_count / total
            ig = information_gain(state, q.bitmask)

            candidates.append(QuestionCandidate(
                question=q,
                split_ratio=ratio,
                information_gain=ig,
                yes_count=yes_count,
                no_count=no_count,
            ))

        # Cache the results (Fix C)
        if use_cache:
            self._cache_candidates(state, candidates)

        return candidates

    def _compute_candidates_from_bucket(
        self,
        state: int,
        bucket: list,
        exclude_ids: Optional[set[int]] = None
    ) -> list[QuestionCandidate]:
        """Compute candidates from a specific bucket only (Fix B).

        This is much faster than compute_candidates when you only need
        questions from a specific popcount range.

        Args:
            state: Current feasible set
            bucket: List of questions to consider (from popcount buckets)
            exclude_ids: Question IDs to exclude

        Returns:
            List of QuestionCandidate objects from the bucket
        """
        exclude_ids = exclude_ids or set()
        total = popcount(state)
        candidates = []

        for q in bucket:
            if q.question_id in exclude_ids:
                continue

            yes_count = popcount(get_yes_set(state, q.bitmask))
            no_count = total - yes_count

            # Skip questions that don't split the state
            if yes_count == 0 or no_count == 0:
                continue

            ratio = yes_count / total
            ig = information_gain(state, q.bitmask)

            candidates.append(QuestionCandidate(
                question=q,
                split_ratio=ratio,
                information_gain=ig,
                yes_count=yes_count,
                no_count=no_count,
            ))

        return candidates

    def find_by_split_range(
        self,
        state: int,
        ratio_min: float,
        ratio_max: float,
        exclude_ids: Optional[set[int]] = None,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with split ratio in a range.

        Args:
            state: Current feasible set
            ratio_min: Minimum split ratio (YES proportion)
            ratio_max: Maximum split ratio
            exclude_ids: Question IDs to exclude
            max_results: Maximum number of results

        Returns:
            List of matching QuestionCandidate objects
        """
        candidates = self.compute_candidates(state, exclude_ids)
        matches = [
            c for c in candidates
            if ratio_min <= c.split_ratio <= ratio_max
        ]

        # Sort by how close to the middle of the range
        target_ratio = (ratio_min + ratio_max) / 2
        matches.sort(key=lambda c: abs(c.split_ratio - target_ratio))

        return matches[:max_results]

    def find_near_balanced(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with near-balanced splits (0.4-0.6).

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            max_results: Maximum number of results

        Returns:
            Questions with ~50% split
        """
        return self.find_by_split_range(
            state, 0.4, 0.6, exclude_ids, max_results
        )

    def find_very_balanced(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with very balanced splits (0.45-0.55).

        For T5 (multi-modal ambiguity) which needs very even splits.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            max_results: Maximum number of results

        Returns:
            Questions with ~50% split (tight range)
        """
        return self.find_by_split_range(
            state, 0.45, 0.55, exclude_ids, max_results
        )

    def find_extreme_split(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        threshold: float = 0.1,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with extreme splits (<threshold or >1-threshold).

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            threshold: Extreme threshold (default 0.1 = <10% or >90%)
            max_results: Maximum number of results

        Returns:
            Questions with extreme YES/NO imbalance
        """
        candidates = self.compute_candidates(state, exclude_ids)
        matches = [
            c for c in candidates
            if c.split_ratio < threshold or c.split_ratio > (1 - threshold)
        ]

        # Sort by extremity (furthest from 0.5)
        matches.sort(key=lambda c: -abs(c.split_ratio - 0.5))

        return matches[:max_results]

    def find_no_op(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        ig_threshold: float = 0.1,
        max_results: int = 100,
        use_bucket: bool = True
    ) -> list[QuestionCandidate]:
        """Find questions that provide minimal information gain.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            ig_threshold: Maximum IG to consider "no-op"
            max_results: Maximum number of results
            use_bucket: If True, search low_popcount bucket first (Fix B)

        Returns:
            Questions that barely reduce entropy
        """
        # Fix B: Search low_popcount bucket first (plateau_low + near_plateau_low)
        # These questions are most likely to have low IG on any state
        if use_bucket:
            candidates = self._compute_candidates_from_bucket(
                state, self.low_popcount_questions, exclude_ids
            )
            matches = [c for c in candidates if c.information_gain <= ig_threshold]

            # If we found enough, return them
            if len(matches) >= max_results:
                matches.sort(key=lambda c: c.information_gain)
                return matches[:max_results]

            # Also try high_popcount bucket (plateau_high + near_plateau_high)
            candidates_high = self._compute_candidates_from_bucket(
                state, self.high_popcount_questions, exclude_ids
            )
            matches_high = [c for c in candidates_high if c.information_gain <= ig_threshold]
            matches.extend(matches_high)

            if matches:
                matches.sort(key=lambda c: c.information_gain)
                return matches[:max_results]

        # Fallback: full scan if buckets didn't yield results
        candidates = self.compute_candidates(state, exclude_ids)
        matches = [c for c in candidates if c.information_gain <= ig_threshold]

        # Sort by lowest IG first
        matches.sort(key=lambda c: c.information_gain)

        return matches[:max_results]

    def find_redundant_with(
        self,
        state: int,
        reference_mask: int,
        exclude_ids: Optional[set[int]] = None,
        overlap_threshold: float = 0.8,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with high bitmask overlap to a reference.

        For T4 (redundant loop) - questions that are similar to previously asked.

        Args:
            state: Current feasible set
            reference_mask: Bitmask to compare against (previous question)
            exclude_ids: Question IDs to exclude
            overlap_threshold: Minimum Jaccard similarity
            max_results: Maximum number of results

        Returns:
            Questions that overlap significantly with reference
        """
        candidates = self.compute_candidates(state, exclude_ids)
        matches = []

        for c in candidates:
            overlap = bitmask_overlap(c.question.bitmask, reference_mask)
            if overlap >= overlap_threshold:
                matches.append((c, overlap))

        # Sort by highest overlap first
        matches.sort(key=lambda x: -x[1])

        return [m[0] for m in matches[:max_results]]

    def find_high_ig(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with highest information gain.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            max_results: Maximum number of results

        Returns:
            Questions sorted by descending IG
        """
        candidates = self.compute_candidates(state, exclude_ids)
        candidates.sort(key=lambda c: -c.information_gain)
        return candidates[:max_results]

    def find_skewed_only(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None,
        low_range: tuple[float, float] = (0.1, 0.3),
        high_range: tuple[float, float] = (0.7, 0.9),
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions with skewed splits only (no near-balanced options).

        For T5 (menu-induced ambiguity) - only extreme splits available.
        Returns questions with split_ratio in [low_range] ∪ [high_range],
        explicitly excluding balanced splits.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            low_range: Low split range (default 0.1-0.3)
            high_range: High split range (default 0.7-0.9)
            max_results: Maximum number of results

        Returns:
            Questions with skewed splits only
        """
        candidates = self.compute_candidates(state, exclude_ids)
        matches = [
            c for c in candidates
            if (low_range[0] <= c.split_ratio <= low_range[1]) or
               (high_range[0] <= c.split_ratio <= high_range[1])
        ]

        # Sort by how far from balanced (more extreme = better for ambiguity)
        matches.sort(key=lambda c: -abs(c.split_ratio - 0.5))

        return matches[:max_results]

    def find_for_branch(
        self,
        state: int,
        target_branch: str,  # "yes" or "no"
        ratio_min: float,
        ratio_max: float,
        exclude_ids: Optional[set[int]] = None,
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions where the target branch falls within a probability range.

        Used for path-first generation to control branch probability.

        Args:
            state: Current feasible set
            target_branch: "yes" or "no" - which branch we want to take
            ratio_min: Minimum probability for the target branch
            ratio_max: Maximum probability for the target branch
            exclude_ids: Question IDs to exclude
            max_results: Maximum number of results

        Returns:
            Questions where P(target_branch) is in [ratio_min, ratio_max]
        """
        if target_branch == "yes":
            return self.find_by_split_range(
                state, ratio_min, ratio_max, exclude_ids, max_results
            )
        else:
            # For NO branch, invert the ratio range
            return self.find_by_split_range(
                state, 1 - ratio_max, 1 - ratio_min, exclude_ids, max_results
            )

    def sample(
        self,
        candidates: list[QuestionCandidate],
        n: int = 1,
        weighted: bool = False
    ) -> list[QuestionCandidate]:
        """Sample from candidate questions.

        Args:
            candidates: List of candidates to sample from
            n: Number to sample
            weighted: If True, weight by IG; if False, uniform

        Returns:
            Sampled candidates
        """
        if not candidates:
            return []

        n = min(n, len(candidates))

        if weighted:
            # Weight by IG (add small epsilon to avoid zero weights)
            weights = [c.information_gain + 0.01 for c in candidates]
            return self.rng.choices(candidates, weights=weights, k=n)
        else:
            return self.rng.sample(candidates, n)

    def sample_one(
        self,
        candidates: list[QuestionCandidate],
        weighted: bool = False
    ) -> Optional[QuestionCandidate]:
        """Sample a single candidate.

        Args:
            candidates: List of candidates
            weighted: Weight by IG

        Returns:
            Single sampled candidate, or None if empty
        """
        samples = self.sample(candidates, 1, weighted)
        return samples[0] if samples else None
