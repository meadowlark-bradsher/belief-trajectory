"""Path-first trajectory generator.

In path-first mode:
1. Build question-answer sequence following constraints
2. Sample secret from final feasible set
3. Answers are verified consistent with chosen secret

Best for: T2 (early collapse → reversal), T3 (plateau → resolution),
          T5 (multi-modal ambiguity), T8 (wrong-way update)
"""

import random
from typing import Optional

from ..models import Trajectory, TrajectoryTurn
from ..config import GeneratorConfig
from ..loader import CUQDataset
from ..bitmask import (
    full_state,
    popcount,
    entropy,
    split_ratio,
    update_state,
    is_singleton,
    items_in_state,
    to_hex,
)
from ..index import QuestionIndex, QuestionCandidate
from .base import TrajectoryGenerator
from .archetypes import (
    get_archetype_constraints,
    SplitConstraint,
    BranchPolicy,
)


class PathFirstGenerator(TrajectoryGenerator):
    """Generator that builds path first, then samples secret."""

    def __init__(
        self,
        dataset: CUQDataset,
        config: Optional[GeneratorConfig] = None,
        seed: Optional[int] = None
    ):
        super().__init__(dataset, config, seed)
        self.rng = random.Random(seed)

    @property
    def generation_mode(self) -> str:
        return "path_first"

    def generate(
        self,
        trajectory_type: str,
        secret: Optional[str] = None,
        secret_index: Optional[int] = None,
        max_turns: Optional[int] = None
    ) -> Trajectory:
        """Generate a trajectory with path-first mode.

        In path-first mode, secret/secret_index are ignored (unless provided
        as a constraint). The secret is sampled from the final feasible set.

        Args:
            trajectory_type: "T1" through "T8"
            secret: Ignored in path-first mode (sampled at end)
            secret_index: Ignored in path-first mode
            max_turns: Override default max turns

        Returns:
            Generated Trajectory
        """
        # Get archetype constraints
        constraints = get_archetype_constraints(trajectory_type)
        max_turns = max_turns or constraints.recommended_turns

        # Build the path (questions and answers determined by policy)
        path_result = self._build_path(
            trajectory_type=trajectory_type,
            max_turns=max_turns,
            constraints=constraints
        )

        if path_result is None:
            raise RuntimeError(
                f"Failed to generate path for {trajectory_type}. "
                "Final feasible set was empty."
            )

        turns, final_state, chosen_secret_index = path_result
        chosen_secret = self.dataset.get_item(chosen_secret_index)

        return self._create_trajectory(
            trajectory_type=trajectory_type,
            secret=chosen_secret,
            secret_index=chosen_secret_index,
            turns=turns
        )

    def _build_path(
        self,
        trajectory_type: str,
        max_turns: int,
        constraints
    ) -> Optional[tuple[list[TrajectoryTurn], int, int]]:
        """Build the question-answer path.

        Args:
            trajectory_type: The trajectory type
            max_turns: Maximum number of turns
            constraints: TurnConstraints for the archetype

        Returns:
            Tuple of (turns, final_state, secret_index) or None if failed
        """
        turns = []
        state = full_state(self.dataset.num_items)
        asked_ids: set[int] = set()
        prev_bitmask: Optional[int] = None

        for turn_num in range(1, max_turns + 1):
            # Check if already solved (need at least 1 item)
            if popcount(state) <= 1:
                break

            # Get constraint for this turn
            split_constraint = constraints.constraint_func(turn_num, max_turns)

            # Find candidate questions matching the split constraint
            candidate = self._find_candidate(
                state=state,
                split_constraint=split_constraint,
                asked_ids=asked_ids,
                prev_bitmask=prev_bitmask
            )

            if candidate is None:
                # Fallback: any question that splits the state
                candidates = self.index.find_by_split_range(
                    state, 0.01, 0.99, asked_ids, max_results=20
                )
                if not candidates:
                    break
                candidate = self.rng.choice(candidates)

            question = candidate.question

            # Determine answer based on branch policy
            answer = self._determine_answer(candidate, split_constraint)

            # Compute turn stats
            entropy_before = entropy(state)
            new_state = update_state(state, question.bitmask, answer)
            min_feasible = split_constraint.min_feasible_after

            # Check floor constraint (T2: don't collapse too far during early phase)
            if popcount(new_state) < min_feasible:
                # Try the other answer
                alt_answer = not answer
                alt_state = update_state(state, question.bitmask, alt_answer)
                if popcount(alt_state) >= min_feasible:
                    answer = alt_answer
                    new_state = alt_state
                else:
                    # Neither answer satisfies floor, skip this question
                    continue

            # Verify new state is non-empty
            if popcount(new_state) == 0:
                # This path leads to empty set, try the other answer
                answer = not answer
                new_state = update_state(state, question.bitmask, answer)
                if popcount(new_state) == 0:
                    # Both answers lead to empty, skip this question
                    continue

            entropy_after = entropy(new_state)
            ratio = candidate.split_ratio
            branch_prob = ratio if answer else (1 - ratio)

            turn = TrajectoryTurn(
                turn=turn_num,
                question_id=question.question_id,
                question=question.question,
                answer=answer,
                feasible_set_size_before=popcount(state),
                feasible_set_size_after=popcount(new_state),
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                split_ratio=ratio,
                branch_taken="yes" if answer else "no",
                branch_probability=branch_prob,
                state_before_hex=to_hex(state),
                question_bitmask_hex=to_hex(question.bitmask),
            )

            turns.append(turn)
            asked_ids.add(question.question_id)
            prev_bitmask = question.bitmask
            state = new_state

        # Sample secret from final feasible set
        if popcount(state) == 0:
            return None

        possible_secrets = items_in_state(state)
        chosen_secret_index = self.rng.choice(possible_secrets)

        return turns, state, chosen_secret_index

    def _find_candidate(
        self,
        state: int,
        split_constraint: SplitConstraint,
        asked_ids: set[int],
        prev_bitmask: Optional[int]
    ) -> Optional[QuestionCandidate]:
        """Find a question matching the split constraint.

        Args:
            state: Current feasible set
            split_constraint: Constraint for this turn
            asked_ids: Already asked question IDs
            prev_bitmask: Previous question's bitmask (for redundancy)

        Returns:
            QuestionCandidate or None
        """
        # Handle redundancy requirement (T4 - legacy, now uses require_no_op)
        if split_constraint.require_redundant and prev_bitmask is not None:
            candidates = self.index.find_redundant_with(
                state=state,
                reference_mask=prev_bitmask,
                exclude_ids=asked_ids,
                overlap_threshold=split_constraint.redundancy_threshold,
                max_results=50
            )
            # Filter to split range
            candidates = [
                c for c in candidates
                if split_constraint.ratio_min <= c.split_ratio <= split_constraint.ratio_max
            ]
            if candidates:
                return self.rng.choice(candidates)

        # Handle skewed split requirement (T5 menu-induced ambiguity)
        if split_constraint.require_skewed_split:
            candidates = self.index.find_skewed_only(
                state=state,
                exclude_ids=asked_ids,
                max_results=100
            )
            if candidates:
                return self.rng.choice(candidates)
            # Fallback: find most skewed available
            candidates = self.index.find_by_split_range(
                state, 0.01, 0.99, asked_ids, max_results=100
            )
            if candidates:
                # Sort by distance from 0.5 (most skewed first)
                candidates.sort(key=lambda c: -abs(c.split_ratio - 0.5))
                return candidates[0]

        # Handle no-op requirement (T3/T4 plateau phase)
        if split_constraint.require_no_op:
            candidates = self.index.find_no_op(
                state=state,
                exclude_ids=asked_ids,
                ig_threshold=split_constraint.no_op_max_ig,
                max_results=100
            )
            if candidates:
                return self.rng.choice(candidates)
            # Fallback: find lowest IG questions available
            candidates = self.index.find_by_split_range(
                state, 0.01, 0.99, asked_ids, max_results=100
            )
            if candidates:
                # Sort by IG ascending and take from the low end
                candidates.sort(key=lambda c: c.information_gain)
                return candidates[0]

        # Handle high-IG requirement (T3 resolution phase)
        if split_constraint.require_high_ig:
            candidates = self.index.find_high_ig(
                state=state,
                exclude_ids=asked_ids,
                max_results=50
            )
            # Also filter by split range if specified
            if split_constraint.ratio_min > 0 or split_constraint.ratio_max < 1:
                candidates = [
                    c for c in candidates
                    if split_constraint.ratio_min <= c.split_ratio <= split_constraint.ratio_max
                ]
            if candidates:
                return self.rng.choice(candidates[:10])  # Sample from top 10

        # Standard split-based search
        candidates = self.index.find_by_split_range(
            state=state,
            ratio_min=split_constraint.ratio_min,
            ratio_max=split_constraint.ratio_max,
            exclude_ids=asked_ids,
            max_results=100
        )

        if candidates:
            return self.rng.choice(candidates)

        return None

    def _determine_answer(
        self,
        candidate: QuestionCandidate,
        constraint: SplitConstraint
    ) -> bool:
        """Determine the answer based on branch policy.

        Args:
            candidate: The selected question candidate
            constraint: Split constraint with branch policy

        Returns:
            True for YES, False for NO
        """
        ratio = candidate.split_ratio
        policy = constraint.branch_policy

        if policy == BranchPolicy.LIKELY:
            # Take the more probable branch
            return ratio >= 0.5
        elif policy == BranchPolicy.UNLIKELY:
            # Take the less probable branch
            return ratio < 0.5
        elif policy == BranchPolicy.RANDOM:
            return self.rng.random() < 0.5
        else:
            # SECRET policy doesn't apply in path-first, use LIKELY
            return ratio >= 0.5
