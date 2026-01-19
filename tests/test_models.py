"""Tests for data models."""

import pytest
from src.models import (
    Prediction,
    Guess,
    TrajectoryTurn,
    Trajectory,
    get_mast_modes,
    derive_mast_modes,
    TRAJECTORY_MAST_MAPPING,
)


class TestPrediction:
    def test_creation(self):
        pred = Prediction(predicted_answer=True, confidence=0.8)
        assert pred.predicted_answer == True
        assert pred.confidence == 0.8

    def test_no_answer(self):
        pred = Prediction(predicted_answer=False, confidence=0.3)
        assert pred.predicted_answer == False


class TestGuess:
    def test_basic_guess(self):
        guess = Guess(secret_index=42, secret="Apple", confidence=0.95)
        assert guess.secret_index == 42
        assert guess.secret == "Apple"
        assert guess.confidence == 0.95
        assert guess.verification_claim is None

    def test_with_verification_claim(self):
        guess = Guess(
            secret_index=42,
            secret="Apple",
            confidence=0.95,
            verification_claim="It must be Apple because it's red and edible"
        )
        assert guess.verification_claim is not None


class TestTrajectoryTurn:
    def test_basic_turn(self):
        turn = TrajectoryTurn(
            turn=1,
            question_id=123,
            question="Is it alive?",
            answer=True,
            feasible_set_size_before=128,
            feasible_set_size_after=64,
            entropy_before=7.0,
            entropy_after=6.0,
            split_ratio=0.5,
            branch_taken="yes",
            branch_probability=0.5,
        )
        assert turn.turn == 1
        assert turn.model_action == "continue"
        assert turn.guess is None
        assert turn.prediction is None

    def test_turn_with_guess(self):
        guess = Guess(secret_index=10, secret="Banana", confidence=0.9)
        turn = TrajectoryTurn(
            turn=5,
            question_id=456,
            question="Is it yellow?",
            answer=True,
            feasible_set_size_before=4,
            feasible_set_size_after=1,
            entropy_before=2.0,
            entropy_after=0.0,
            split_ratio=0.25,
            branch_taken="yes",
            branch_probability=0.25,
            model_action="guess",
            guess=guess,
            guess_correct=True,
        )
        assert turn.model_action == "guess"
        assert turn.guess is not None
        assert turn.guess_correct == True

    def test_turn_with_prediction(self):
        pred = Prediction(predicted_answer=True, confidence=0.7)
        turn = TrajectoryTurn(
            turn=2,
            question_id=789,
            question="Can you eat it?",
            answer=True,
            feasible_set_size_before=64,
            feasible_set_size_after=32,
            entropy_before=6.0,
            entropy_after=5.0,
            split_ratio=0.5,
            branch_taken="yes",
            branch_probability=0.5,
            prediction=pred,
        )
        assert turn.prediction is not None
        assert turn.prediction.confidence == 0.7


class TestTrajectory:
    def test_valid_trajectory_types(self):
        for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
            traj = Trajectory(
                trajectory_id="test-123",
                trajectory_type=t,
                target_mast_modes=[],
                generation_mode="path_first",
                secret="Apple",
                secret_index=42,
            )
            assert traj.trajectory_type == t

    def test_invalid_trajectory_type(self):
        with pytest.raises(ValueError, match="Invalid trajectory type"):
            Trajectory(
                trajectory_id="test-123",
                trajectory_type="T9",  # Invalid
                target_mast_modes=[],
                generation_mode="path_first",
                secret="Apple",
                secret_index=42,
            )

    def test_num_turns_empty(self):
        traj = Trajectory(
            trajectory_id="test-123",
            trajectory_type="T1",
            target_mast_modes=[],
            generation_mode="path_first",
            secret="Apple",
            secret_index=42,
        )
        assert traj.num_turns == 0

    def test_num_turns_with_turns(self):
        turn = TrajectoryTurn(
            turn=1,
            question_id=123,
            question="Test?",
            answer=True,
            feasible_set_size_before=128,
            feasible_set_size_after=64,
            entropy_before=7.0,
            entropy_after=6.0,
            split_ratio=0.5,
            branch_taken="yes",
            branch_probability=0.5,
        )
        traj = Trajectory(
            trajectory_id="test-123",
            trajectory_type="T1",
            target_mast_modes=[],
            generation_mode="path_first",
            secret="Apple",
            secret_index=42,
            turns=[turn, turn, turn],
        )
        assert traj.num_turns == 3

    def test_entropy_properties(self):
        turns = [
            TrajectoryTurn(
                turn=1,
                question_id=1,
                question="Q1",
                answer=True,
                feasible_set_size_before=128,
                feasible_set_size_after=64,
                entropy_before=7.0,
                entropy_after=6.0,
                split_ratio=0.5,
                branch_taken="yes",
                branch_probability=0.5,
            ),
            TrajectoryTurn(
                turn=2,
                question_id=2,
                question="Q2",
                answer=False,
                feasible_set_size_before=64,
                feasible_set_size_after=32,
                entropy_before=6.0,
                entropy_after=5.0,
                split_ratio=0.5,
                branch_taken="no",
                branch_probability=0.5,
            ),
        ]
        traj = Trajectory(
            trajectory_id="test-123",
            trajectory_type="T1",
            target_mast_modes=[],
            generation_mode="path_first",
            secret="Apple",
            secret_index=42,
            turns=turns,
        )
        assert traj.initial_entropy == 7.0
        assert traj.final_entropy == 5.0


class TestGetMastModes:
    def test_t1_baseline(self):
        assert get_mast_modes("T1") == []

    def test_t2_early_collapse(self):
        modes = get_mast_modes("T2")
        assert "FM-1.1" in modes
        assert "FM-2.6" in modes

    def test_t8_wrong_verification(self):
        modes = get_mast_modes("T8")
        assert "FM-3.3" in modes

    def test_unknown_type(self):
        assert get_mast_modes("T99") == []


class TestDeriveMastModes:
    def test_t4_world_only(self):
        # T4 has FM-1.3 based on world type alone (no overlay needed)
        modes = derive_mast_modes("T4", [])
        assert "FM-1.3" in modes

    def test_t5_world_only(self):
        # T5 has FM-2.2 based on world type alone
        modes = derive_mast_modes("T5", [])
        assert "FM-2.2" in modes

    def test_t6_with_calibrated_overlay(self):
        # T6 + calibrated_argmax → FM-2.6
        modes = derive_mast_modes("T6", ["pred:calibrated_argmax"])
        assert "FM-2.6" in modes

    def test_t6_without_overlay(self):
        # T6 without the overlay doesn't get FM-2.6
        modes = derive_mast_modes("T6", [])
        assert "FM-2.6" not in modes

    def test_t8_with_wrong_guess(self):
        # T8 + wrong_guess → FM-3.3
        modes = derive_mast_modes("T8", ["term:wrong_guess"])
        assert "FM-3.3" in modes

    def test_t3_with_premature_stop(self):
        # T3 + premature_stop → FM-3.1
        modes = derive_mast_modes("T3", ["term:premature_stop"])
        assert "FM-3.1" in modes

    def test_t3_with_unaware(self):
        # T3 + unaware → FM-1.5
        modes = derive_mast_modes("T3", ["term:unaware"])
        assert "FM-1.5" in modes

    def test_t2_with_wrong_guess(self):
        # T2 + wrong_guess → FM-1.1
        modes = derive_mast_modes("T2", ["term:wrong_guess"])
        assert "FM-1.1" in modes

    def test_modes_are_sorted_and_unique(self):
        # Multiple overlays shouldn't create duplicates
        modes = derive_mast_modes("T3", ["term:premature_stop", "term:unaware"])
        assert modes == sorted(set(modes))
