"""Secret-first trajectory generator.

In secret-first mode:
1. Choose a secret
2. Select questions matching target entropy curve
3. Answers are determined by the secret

Best for: T1 (smooth halving), T4 (redundant loop), T6 (prediction mismatch), T7 (late shock)
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
    information_gain,
    update_state,
    get_answer_for_secret,
    is_singleton,
    to_hex,
)
from ..index import QuestionIndex, QuestionCandidate
from .base import TrajectoryGenerator
from .archetypes import (
    get_archetype_constraints,
    SplitConstraint,
    BranchPolicy,
)


class SecretFirstGenerator(TrajectoryGenerator):
    """Generator that selects secret first, then builds trajectory."""

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
        return "secret_first"

    def generate(
        self,
        trajectory_type: str,
        secret: Optional[str] = None,
        secret_index: Optional[int] = None,
        max_turns: Optional[int] = None
    ) -> Trajectory:
        """Generate a trajectory with secret-first mode.

        Args:
            trajectory_type: "T1" through "T8"
            secret: Secret item name (if None, randomly chosen)
            secret_index: Secret index (alternative to secret name)
            max_turns: Override default max turns

        Returns:
            Generated Trajectory
        """
        # Resolve secret
        if secret_index is not None:
            secret = self.dataset.get_item(secret_index)
        elif secret is not None:
            secret_index = self.dataset.get_item_index(secret)
            if secret_index is None:
                raise ValueError(f"Unknown secret: {secret}")
        else:
            secret_index = self.rng.randint(0, self.dataset.num_items - 1)
            secret = self.dataset.get_item(secret_index)

        # Get archetype constraints
        constraints = get_archetype_constraints(trajectory_type)
        max_turns = max_turns or constraints.recommended_turns

        # Build trajectory
        turns = self._build_trajectory(
            trajectory_type=trajectory_type,
            secret_index=secret_index,
            max_turns=max_turns,
            constraints=constraints
        )

        return self._create_trajectory(
            trajectory_type=trajectory_type,
            secret=secret,
            secret_index=secret_index,
            turns=turns
        )

    def _build_trajectory(
        self,
        trajectory_type: str,
        secret_index: int,
        max_turns: int,
        constraints
    ) -> list[TrajectoryTurn]:
        """Build the turn sequence.

        Args:
            trajectory_type: The trajectory type
            secret_index: Index of the secret
            max_turns: Maximum number of turns
            constraints: TurnConstraints for the archetype

        Returns:
            List of TrajectoryTurn objects
        """
        turns = []
        state = full_state(self.dataset.num_items)
        asked_ids: set[int] = set()
        prev_bitmask: Optional[int] = None

        for turn_num in range(1, max_turns + 1):
            # Check if already solved
            if is_singleton(state):
                break

            # Get constraint for this turn
            split_constraint = constraints.constraint_func(turn_num, max_turns)

            # Find candidate questions
            candidate = self._find_candidate(
                state=state,
                split_constraint=split_constraint,
                asked_ids=asked_ids,
                prev_bitmask=prev_bitmask,
                secret_index=secret_index
            )

            if candidate is None:
                # Fallback: any high-IG question
                candidates = self.index.find_high_ig(state, asked_ids, max_results=10)
                if not candidates:
                    break
                candidate = self.rng.choice(candidates)

            question = candidate.question

            # Get answer from secret
            answer = get_answer_for_secret(question.bitmask, secret_index)

            # Compute turn stats
            entropy_before = entropy(state)
            new_state = update_state(state, question.bitmask, answer)
            entropy_after = entropy(new_state)

            ratio = split_ratio(state, question.bitmask)
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

        return turns

    def _find_candidate(
        self,
        state: int,
        split_constraint: SplitConstraint,
        asked_ids: set[int],
        prev_bitmask: Optional[int],
        secret_index: int
    ) -> Optional[QuestionCandidate]:
        """Find a question matching the split constraint.

        Args:
            state: Current feasible set
            split_constraint: Constraint for this turn
            asked_ids: Already asked question IDs
            prev_bitmask: Previous question's bitmask (for redundancy)
            secret_index: The secret (for filtering)

        Returns:
            QuestionCandidate or None
        """
        # Handle redundancy requirement (T4)
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

        # Standard split-based search
        candidates = self.index.find_by_split_range(
            state=state,
            ratio_min=split_constraint.ratio_min,
            ratio_max=split_constraint.ratio_max,
            exclude_ids=asked_ids,
            max_results=100
        )

        if not candidates:
            return None

        # For secret-first mode with UNLIKELY policy, we need to filter
        # for questions where the secret's answer is the unlikely branch
        if split_constraint.branch_policy == BranchPolicy.UNLIKELY:
            unlikely_candidates = []
            for c in candidates:
                answer = get_answer_for_secret(c.question.bitmask, secret_index)
                # If answer=YES and ratio < 0.5, or answer=NO and ratio > 0.5
                # then we're taking the unlikely branch
                is_unlikely = (answer and c.split_ratio < 0.5) or \
                              (not answer and c.split_ratio > 0.5)
                if is_unlikely:
                    unlikely_candidates.append(c)

            if unlikely_candidates:
                return self.rng.choice(unlikely_candidates)

        # Default: sample from matching candidates
        return self.index.sample_one(candidates, weighted=False)
