"""Tests for gate validators."""

import pytest
from src.models import Trajectory, TrajectoryTurn, Guess
from src.validators import (
    ValidationResult,
    validate_t1_gate,
    validate_t2_gate,
    validate_t3_gate,
    validate_t4_gate,
    validate_t5_gate,
    validate_t6_gate,
    validate_t7_gate,
    validate_t8_gate,
    validate_trajectory,
    BALANCED_EPSILON,
    RARE_BRANCH_THRESHOLD,
    LOW_IG_THRESHOLD,
    PLATEAU_MIN_TURNS,
)


def make_turn(
    turn_num: int,
    split_ratio: float = 0.5,
    entropy_before: float = 6.0,
    entropy_after: float = 5.0,
    feasible_before: int = 64,
    feasible_after: int = 32,
    branch_prob: float = None,
    guess: Guess = None,
    guess_correct: bool = None,
) -> TrajectoryTurn:
    """Helper to create test turns."""
    branch_taken = "yes" if split_ratio >= 0.5 else "no"
    if branch_prob is None:
        branch_prob = split_ratio if branch_taken == "yes" else (1 - split_ratio)

    return TrajectoryTurn(
        turn=turn_num,
        question_id=turn_num * 100,
        question=f"Question {turn_num}?",
        answer=(branch_taken == "yes"),
        feasible_set_size_before=feasible_before,
        feasible_set_size_after=feasible_after,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        split_ratio=split_ratio,
        branch_taken=branch_taken,
        branch_probability=branch_prob,
        guess=guess,
        guess_correct=guess_correct,
        model_action="guess" if guess else "continue",
    )


def make_trajectory(
    ttype: str,
    turns: list[TrajectoryTurn],
    target_mast_modes: list[str] = None,
) -> Trajectory:
    """Helper to create test trajectories."""
    return Trajectory(
        trajectory_id="test-123",
        trajectory_type=ttype,
        target_mast_modes=target_mast_modes or [],
        generation_mode="path_first",
        secret="TestSecret",
        secret_index=0,
        turns=turns,
    )


class TestValidationResult:
    def test_passed(self):
        result = ValidationResult(passed=True)
        assert result.passed == True
        assert result.reason is None

    def test_failed_with_reason(self):
        result = ValidationResult(passed=False, reason="Test failure")
        assert result.passed == False
        assert result.reason == "Test failure"

    def test_with_metrics(self):
        result = ValidationResult(passed=True, metrics={"count": 5})
        assert result.metrics["count"] == 5


class TestT1Gate:
    """T1: All balanced splits."""

    def test_all_balanced_passes(self):
        turns = [
            make_turn(1, split_ratio=0.5),
            make_turn(2, split_ratio=0.48),
            make_turn(3, split_ratio=0.52),
        ]
        traj = make_trajectory("T1", turns)
        result = validate_t1_gate(traj)
        assert result.passed == True

    def test_unbalanced_fails(self):
        turns = [
            make_turn(1, split_ratio=0.5),
            make_turn(2, split_ratio=0.1),  # Too skewed
            make_turn(3, split_ratio=0.5),
        ]
        traj = make_trajectory("T1", turns)
        result = validate_t1_gate(traj)
        assert result.passed == False
        assert "Unbalanced" in result.reason

    def test_skewed_at_singleton_ok(self):
        # Skewed split is OK if feasible_set_size_after == 1 (final turn)
        turns = [
            make_turn(1, split_ratio=0.5, feasible_after=2),
            make_turn(2, split_ratio=0.1, feasible_after=1),  # Skewed but final
        ]
        traj = make_trajectory("T1", turns)
        result = validate_t1_gate(traj)
        assert result.passed == True


class TestT2Gate:
    """T2: Early rare branch AND later rare branch."""

    def test_early_and_late_rare_passes(self):
        turns = [
            make_turn(1, split_ratio=0.1, branch_prob=0.1),  # Early rare
            make_turn(2, split_ratio=0.5, branch_prob=0.5),
            make_turn(3, split_ratio=0.5, branch_prob=0.5),
            make_turn(4, split_ratio=0.9, branch_prob=0.1),  # Late rare
        ]
        traj = make_trajectory("T2", turns)
        result = validate_t2_gate(traj)
        assert result.passed == True

    def test_no_early_rare_fails(self):
        turns = [
            make_turn(1, split_ratio=0.5, branch_prob=0.5),  # Not rare
            make_turn(2, split_ratio=0.5, branch_prob=0.5),
            make_turn(3, split_ratio=0.5, branch_prob=0.5),
            make_turn(4, split_ratio=0.1, branch_prob=0.1),  # Late rare but no early
        ]
        traj = make_trajectory("T2", turns)
        result = validate_t2_gate(traj)
        assert result.passed == False
        assert "early" in result.reason.lower()

    def test_no_late_rare_fails(self):
        turns = [
            make_turn(1, split_ratio=0.1, branch_prob=0.1),  # Early rare
            make_turn(2, split_ratio=0.5, branch_prob=0.5),
            make_turn(3, split_ratio=0.5, branch_prob=0.5),
        ]
        traj = make_trajectory("T2", turns)
        result = validate_t2_gate(traj)
        assert result.passed == False
        assert "later" in result.reason.lower()


class TestT3Gate:
    """T3: Plateau followed by balanced resolution."""

    def test_plateau_with_resolution_passes(self):
        # Create plateau (low IG) followed by resolution (balanced, high IG)
        turns = [
            make_turn(1, entropy_before=6.0, entropy_after=5.95),  # Low IG
            make_turn(2, entropy_before=5.95, entropy_after=5.90),  # Low IG
            make_turn(3, entropy_before=5.90, entropy_after=5.85),  # Low IG
            make_turn(4, split_ratio=0.5, entropy_before=5.85, entropy_after=4.85),  # Resolution
        ]
        traj = make_trajectory("T3", turns)
        result = validate_t3_gate(traj)
        assert result.passed == True

    def test_no_plateau_fails(self):
        turns = [
            make_turn(1, entropy_before=6.0, entropy_after=5.0),  # High IG
            make_turn(2, entropy_before=5.0, entropy_after=4.0),  # High IG
        ]
        traj = make_trajectory("T3", turns)
        result = validate_t3_gate(traj)
        assert result.passed == False
        assert "plateau" in result.reason.lower()


class TestT4Gate:
    """T4: Consecutive low-IG turns (redundant loop)."""

    def test_low_ig_streak_passes(self):
        turns = [
            make_turn(1, entropy_before=6.0, entropy_after=5.95),  # Low IG
            make_turn(2, entropy_before=5.95, entropy_after=5.90),  # Low IG
            make_turn(3, entropy_before=5.90, entropy_after=5.85),  # Low IG
        ]
        traj = make_trajectory("T4", turns)
        result = validate_t4_gate(traj)
        assert result.passed == True
        assert result.metrics["max_consecutive_low_ig"] >= PLATEAU_MIN_TURNS

    def test_no_streak_fails(self):
        turns = [
            make_turn(1, entropy_before=6.0, entropy_after=5.0),  # High IG
            make_turn(2, entropy_before=5.0, entropy_after=4.9),  # Low IG
            make_turn(3, entropy_before=4.9, entropy_after=3.9),  # High IG
        ]
        traj = make_trajectory("T4", turns)
        result = validate_t4_gate(traj)
        assert result.passed == False


class TestT5Gate:
    """T5: Skewed mid-game, not all balanced."""

    def test_skewed_midgame_passes(self):
        turns = [
            make_turn(1, split_ratio=0.5, feasible_before=128),  # Early, balanced
            make_turn(2, split_ratio=0.2, feasible_before=16),   # Mid, skewed
            make_turn(3, split_ratio=0.5, feasible_before=4),    # Late, balanced
        ]
        traj = make_trajectory("T5", turns)
        result = validate_t5_gate(traj)
        assert result.passed == True

    def test_all_balanced_fails(self):
        # All balanced is T1, not T5
        turns = [
            make_turn(1, split_ratio=0.5, feasible_before=128),
            make_turn(2, split_ratio=0.5, feasible_before=16),
            make_turn(3, split_ratio=0.5, feasible_before=4),
        ]
        traj = make_trajectory("T5", turns)
        result = validate_t5_gate(traj)
        assert result.passed == False
        assert "T1" in result.reason

    def test_no_midgame_skew_fails(self):
        # Skewed but not in mid-game range
        turns = [
            make_turn(1, split_ratio=0.1, feasible_before=128),  # Early, skewed (too big)
            make_turn(2, split_ratio=0.5, feasible_before=64),   # Balanced
            make_turn(3, split_ratio=0.1, feasible_before=4),    # Late, skewed (too small)
        ]
        traj = make_trajectory("T5", turns)
        result = validate_t5_gate(traj)
        assert result.passed == False
        assert "mid-game" in result.reason.lower()


class TestT6Gate:
    """T6: No world constraints (always passes)."""

    def test_always_passes(self):
        turns = [make_turn(1), make_turn(2)]
        traj = make_trajectory("T6", turns)
        result = validate_t6_gate(traj)
        assert result.passed == True


class TestT7Gate:
    """T7: Late shock (rare branch in last few turns)."""

    def test_late_shock_passes(self):
        turns = [
            make_turn(1, branch_prob=0.5),
            make_turn(2, branch_prob=0.5),
            make_turn(3, branch_prob=0.5),
            make_turn(4, branch_prob=0.1),  # Late shock
        ]
        traj = make_trajectory("T7", turns)
        result = validate_t7_gate(traj)
        assert result.passed == True

    def test_no_late_shock_fails(self):
        turns = [
            make_turn(1, branch_prob=0.1),  # Shock too early (not in last 4)
            make_turn(2, branch_prob=0.5),
            make_turn(3, branch_prob=0.5),
            make_turn(4, branch_prob=0.5),
            make_turn(5, branch_prob=0.5),
            make_turn(6, branch_prob=0.5),  # Last 4 turns are 3,4,5,6 - all balanced
        ]
        traj = make_trajectory("T7", turns)
        result = validate_t7_gate(traj)
        assert result.passed == False

    def test_too_few_turns_fails(self):
        turns = [make_turn(1, branch_prob=0.5)]
        traj = make_trajectory("T7", turns)
        result = validate_t7_gate(traj)
        assert result.passed == False


class TestT8Gate:
    """T8: Wrong guess with verification claim."""

    def test_wrong_guess_with_claim_passes(self):
        guess = Guess(
            secret_index=10,
            secret="WrongAnswer",
            confidence=0.9,
            verification_claim="I'm confident because X, Y, Z"
        )
        turns = [
            make_turn(1),
            make_turn(2, guess=guess, guess_correct=False),
        ]
        traj = make_trajectory("T8", turns, target_mast_modes=["FM-3.3"])
        result = validate_t8_gate(traj)
        assert result.passed == True

    def test_correct_guess_fails(self):
        guess = Guess(
            secret_index=10,
            secret="CorrectAnswer",
            confidence=0.9,
            verification_claim="Reasoning..."
        )
        turns = [
            make_turn(1),
            make_turn(2, guess=guess, guess_correct=True),  # Correct, not wrong
        ]
        traj = make_trajectory("T8", turns, target_mast_modes=["FM-3.3"])
        result = validate_t8_gate(traj)
        assert result.passed == False

    def test_no_claim_fails(self):
        guess = Guess(
            secret_index=10,
            secret="WrongAnswer",
            confidence=0.9,
            # No verification_claim
        )
        turns = [
            make_turn(1),
            make_turn(2, guess=guess, guess_correct=False),
        ]
        traj = make_trajectory("T8", turns, target_mast_modes=["FM-3.3"])
        result = validate_t8_gate(traj)
        assert result.passed == False

    def test_missing_mast_mode_fails(self):
        guess = Guess(
            secret_index=10,
            secret="WrongAnswer",
            confidence=0.9,
            verification_claim="Reasoning..."
        )
        turns = [
            make_turn(1),
            make_turn(2, guess=guess, guess_correct=False),
        ]
        traj = make_trajectory("T8", turns, target_mast_modes=[])  # Missing FM-3.3
        result = validate_t8_gate(traj)
        assert result.passed == False


class TestValidateTrajectory:
    """Test the main validate_trajectory dispatcher."""

    def test_dispatches_to_correct_validator(self):
        # T1 with all balanced should pass
        turns = [make_turn(1, split_ratio=0.5), make_turn(2, split_ratio=0.5)]
        traj = make_trajectory("T1", turns)
        result = validate_trajectory(traj)
        assert result.passed == True

    def test_unknown_type_fails(self):
        turns = [make_turn(1)]
        # Bypass Trajectory validation to test validator
        traj = Trajectory.__new__(Trajectory)
        traj.trajectory_id = "test"
        traj.trajectory_type = "T99"  # Invalid
        traj.target_mast_modes = []
        traj.generation_mode = "path_first"
        traj.secret = "Test"
        traj.secret_index = 0
        traj.turns = turns
        traj.overlay_tags = []
        traj.metadata = {}

        result = validate_trajectory(traj)
        assert result.passed == False
        assert "Unknown" in result.reason
