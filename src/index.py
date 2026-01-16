"""Question indexing for efficient split-based lookup.

This module provides efficient ways to find questions based on their
split characteristics on a given feasible set state.
"""

import random
from dataclasses import dataclass
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


class QuestionIndex:
    """Index for efficient question lookup by split characteristics.

    This class provides methods to find questions matching various
    split criteria on a given feasible set state.
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

    def compute_candidates(
        self,
        state: int,
        exclude_ids: Optional[set[int]] = None
    ) -> list[QuestionCandidate]:
        """Compute characteristics for all questions on a state.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude (already asked)

        Returns:
            List of QuestionCandidate objects
        """
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
        max_results: int = 100
    ) -> list[QuestionCandidate]:
        """Find questions that provide minimal information gain.

        Args:
            state: Current feasible set
            exclude_ids: Question IDs to exclude
            ig_threshold: Maximum IG to consider "no-op"
            max_results: Maximum number of results

        Returns:
            Questions that barely reduce entropy
        """
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
