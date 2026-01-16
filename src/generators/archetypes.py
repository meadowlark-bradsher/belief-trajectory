"""Archetype constraint definitions for T1-T8 trajectory types.

Each archetype defines constraints per turn that control:
- Target split ratio ranges
- Whether to take the likely or unlikely branch
- Special conditions (redundancy, plateaus, etc.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class BranchPolicy(Enum):
    """Policy for which branch to take when answer is undetermined."""
    LIKELY = "likely"  # Take the more probable branch
    UNLIKELY = "unlikely"  # Take the less probable branch
    RANDOM = "random"  # Random choice
    SECRET = "secret"  # Determined by secret (secret-first mode)


@dataclass
class SplitConstraint:
    """Constraint on question split ratio for a turn."""
    ratio_min: float = 0.0
    ratio_max: float = 1.0
    branch_policy: BranchPolicy = BranchPolicy.SECRET

    # T4: redundant loop
    require_redundant: bool = False
    redundancy_threshold: float = 0.8  # Jaccard overlap required

    # T3: IG-based selection
    require_no_op: bool = False  # Use find_no_op instead of split range
    no_op_max_ig: float = 0.1  # Max IG for "no-op" questions
    require_high_ig: bool = False  # Use find_high_ig for resolution

    # T2: collapse floor + reversal
    min_feasible_after: int = 1  # Don't let feasible set drop below this
    is_reversal_turn: bool = False  # Force contradiction on this turn

    # T5: menu-induced ambiguity (Gate B)
    require_skewed_split: bool = False  # Only allow skewed splits [0.1-0.3] ∪ [0.7-0.9]


@dataclass
class TurnConstraints:
    """Constraints for an entire trajectory by turn number.

    The constraint_func takes (turn_number, total_turns) and returns
    the SplitConstraint for that turn.
    """
    constraint_func: Callable[[int, int], SplitConstraint]
    min_turns: int = 5
    max_turns: int = 20
    recommended_turns: int = 10
    description: str = ""


def _t1_constraint(turn: int, total: int) -> SplitConstraint:
    """T1: Smooth halving - always near-balanced splits."""
    return SplitConstraint(
        ratio_min=0.4,
        ratio_max=0.6,
        branch_policy=BranchPolicy.SECRET
    )


def _t2_constraint(turn: int, total: int) -> SplitConstraint:
    """T2: Early collapse → reversal → resolution.

    Phase 1 (turns 1 to total//3): Extreme splits, take unlikely branch
        - Floor at 4 items to preserve room for reversal
    Phase 2 (reversal turn): Extreme split, take unlikely branch (contradiction)
        - Tests belief revision after confident early commitment
    Phase 3 (remaining): Balanced splits to resolve
    """
    collapse_end = total // 3
    reversal_turn = collapse_end + 1

    if turn <= collapse_end:
        # Early collapse: rare branch, but don't collapse below 4 items
        return SplitConstraint(
            ratio_min=0.05,
            ratio_max=0.2,
            branch_policy=BranchPolicy.UNLIKELY,
            min_feasible_after=4  # Floor to preserve reversal room
        )
    elif turn == reversal_turn:
        # Reversal: extreme split, take the UNLIKELY branch
        # This contradicts the "confident" state from early collapse
        return SplitConstraint(
            ratio_min=0.7,
            ratio_max=0.95,
            branch_policy=BranchPolicy.UNLIKELY,
            is_reversal_turn=True
        )
    else:
        # Resolution: balanced splits
        return SplitConstraint(
            ratio_min=0.35,
            ratio_max=0.65,
            branch_policy=BranchPolicy.LIKELY
        )


def _t3_constraint(turn: int, total: int) -> SplitConstraint:
    """T3: Plateau → forced resolution.

    Most turns: low IG questions (≤0.1 bits), take likely branch (stall)
    Final turns: high IG questions (near-balanced) to collapse the set
    """
    resolution_phase = turn >= total - 2

    if resolution_phase:
        # Force resolution: high IG (near-balanced split)
        return SplitConstraint(
            ratio_min=0.4,
            ratio_max=0.6,
            branch_policy=BranchPolicy.LIKELY,
            require_high_ig=True
        )
    else:
        # Plateau: minimal information gain, take likely branch
        return SplitConstraint(
            ratio_min=0.0,  # Ratio doesn't matter when using no-op
            ratio_max=1.0,
            branch_policy=BranchPolicy.LIKELY,
            require_no_op=True,
            no_op_max_ig=0.1  # ≤0.1 bits per turn
        )


def _t4_constraint(turn: int, total: int) -> SplitConstraint:
    """T4: Redundant loop - low information gain questions (step repetition).

    Phase 1 (turns 1 to total-3): Low IG questions (≤0.1 bits), take likely branch
        - Creates the "spinning wheels" pattern where feasible set barely changes
        - Tests agent's ability to recognize unproductive question sequences
    Phase 2 (final 3 turns): High IG questions to resolve
        - Forces resolution after the redundant loop
    """
    resolution_phase = turn > total - 3

    if resolution_phase:
        # Force resolution: high IG (near-balanced split)
        return SplitConstraint(
            ratio_min=0.4,
            ratio_max=0.6,
            branch_policy=BranchPolicy.LIKELY,
            require_high_ig=True
        )
    else:
        # Redundant loop: minimal information gain
        return SplitConstraint(
            ratio_min=0.0,  # Ratio doesn't matter when using no-op
            ratio_max=1.0,
            branch_policy=BranchPolicy.LIKELY,
            require_no_op=True,
            no_op_max_ig=0.1  # ≤0.1 bits per turn
        )


def _t5_constraint(turn: int, total: int) -> SplitConstraint:
    """T5: Multi-modal ambiguity - menu-induced (Gate B).

    Phase 1 (turns 1-3): Balanced splits to narrow to mid-game
        - Get from 128 items to ~16 items
    Phase 2 (turns 4 to total-3): Skewed splits only
        - Only questions with splits in [0.1-0.3] ∪ [0.7-0.9] available
        - Creates ambiguity where no good disambiguating question exists
    Phase 3 (final 3 turns): Balanced splits to resolve
    """
    early_phase_end = 3
    resolution_phase_start = total - 2

    if turn <= early_phase_end:
        # Early: balanced splits to reach mid-game
        return SplitConstraint(
            ratio_min=0.4,
            ratio_max=0.6,
            branch_policy=BranchPolicy.LIKELY
        )
    elif turn >= resolution_phase_start:
        # Resolution: balanced splits to finish
        return SplitConstraint(
            ratio_min=0.4,
            ratio_max=0.6,
            branch_policy=BranchPolicy.LIKELY,
            require_high_ig=True
        )
    else:
        # Mid-game: skewed splits only (menu-induced ambiguity)
        return SplitConstraint(
            ratio_min=0.0,
            ratio_max=1.0,
            branch_policy=BranchPolicy.LIKELY,
            require_skewed_split=True
        )


def _t6_constraint(turn: int, total: int) -> SplitConstraint:
    """T6: Prediction-belief mismatch - alternate likely/unlikely branches.

    Even turns: likely branch
    Odd turns: unlikely branch
    """
    if turn % 2 == 0:
        return SplitConstraint(
            ratio_min=0.6,
            ratio_max=0.8,
            branch_policy=BranchPolicy.LIKELY
        )
    else:
        return SplitConstraint(
            ratio_min=0.6,
            ratio_max=0.8,
            branch_policy=BranchPolicy.UNLIKELY
        )


def _t7_constraint(turn: int, total: int) -> SplitConstraint:
    """T7: Late shock after confidence.

    Phase 1 (turns 1 to total-3): Build confidence with moderate splits
        - Floor at 8 items to ensure shock happens before |S|=1
        - Taking likely branches builds "everything is fine" belief
    Phase 2 (shock turn = total-2): Extreme split, UNLIKELY branch
        - The "shock" that should trigger verification/revision
    Phase 3 (final turn): Resolution after shock
    """
    shock_turn = total - 2

    if turn < shock_turn:
        # Build confidence: moderate splits (slower than balanced)
        # Take likely branch consistently to build false confidence
        return SplitConstraint(
            ratio_min=0.55,
            ratio_max=0.75,
            branch_policy=BranchPolicy.LIKELY,
            min_feasible_after=4  # Don't collapse below 4 before shock
        )
    elif turn == shock_turn:
        # SHOCK: extreme split, take the unlikely branch
        # This contradicts the "everything is fine" belief
        return SplitConstraint(
            ratio_min=0.85,
            ratio_max=0.98,
            branch_policy=BranchPolicy.UNLIKELY,
            is_reversal_turn=True  # Mark as the shock point
        )
    else:
        # Resolution: balanced splits after shock
        return SplitConstraint(
            ratio_min=0.4,
            ratio_max=0.6,
            branch_policy=BranchPolicy.LIKELY
        )


def _t8_constraint(turn: int, total: int) -> SplitConstraint:
    """T8: Wrong-way update - contradiction points at specific turns.

    Periodically forces unlikely branches to test belief revision.
    """
    # Contradiction every 3rd turn
    is_contradiction = turn % 3 == 0

    if is_contradiction:
        return SplitConstraint(
            ratio_min=0.75,
            ratio_max=0.95,
            branch_policy=BranchPolicy.UNLIKELY
        )
    else:
        return SplitConstraint(
            ratio_min=0.4,
            ratio_max=0.6,
            branch_policy=BranchPolicy.LIKELY
        )


ARCHETYPE_CONSTRAINTS: dict[str, TurnConstraints] = {
    "T1": TurnConstraints(
        constraint_func=_t1_constraint,
        min_turns=7,
        max_turns=12,
        recommended_turns=8,
        description="Smooth halving (control) - balanced splits throughout"
    ),
    "T2": TurnConstraints(
        constraint_func=_t2_constraint,
        min_turns=8,
        max_turns=15,
        recommended_turns=10,
        description="Early collapse → rare-branch reversal"
    ),
    "T3": TurnConstraints(
        constraint_func=_t3_constraint,
        min_turns=10,
        max_turns=20,
        recommended_turns=12,
        description="Plateau → forced resolution"
    ),
    "T4": TurnConstraints(
        constraint_func=_t4_constraint,
        min_turns=8,
        max_turns=15,
        recommended_turns=10,
        description="Redundant loop - low IG questions (step repetition)"
    ),
    "T5": TurnConstraints(
        constraint_func=_t5_constraint,
        min_turns=8,
        max_turns=12,
        recommended_turns=10,
        description="Multi-modal ambiguity - menu-induced (skewed splits only)"
    ),
    "T6": TurnConstraints(
        constraint_func=_t6_constraint,
        min_turns=8,
        max_turns=14,
        recommended_turns=10,
        description="Prediction-belief mismatch - alternating branches"
    ),
    "T7": TurnConstraints(
        constraint_func=_t7_constraint,
        min_turns=8,
        max_turns=14,
        recommended_turns=10,
        description="Late shock after confidence"
    ),
    "T8": TurnConstraints(
        constraint_func=_t8_constraint,
        min_turns=9,
        max_turns=15,
        recommended_turns=12,
        description="Wrong-way update - periodic contradictions"
    ),
}


def get_archetype_constraints(trajectory_type: str) -> TurnConstraints:
    """Get the constraints for a trajectory type.

    Args:
        trajectory_type: "T1" through "T8"

    Returns:
        TurnConstraints for the archetype

    Raises:
        ValueError: If trajectory_type is not valid
    """
    if trajectory_type not in ARCHETYPE_CONSTRAINTS:
        valid = ", ".join(ARCHETYPE_CONSTRAINTS.keys())
        raise ValueError(
            f"Invalid trajectory type '{trajectory_type}'. Valid types: {valid}"
        )
    return ARCHETYPE_CONSTRAINTS[trajectory_type]
