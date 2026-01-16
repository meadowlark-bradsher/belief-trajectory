"""Overlay chain for composing termination and prediction overlays.

Two overlay families are applied in order:
1. Termination overlays (ACTION channel): stop/guess decisions
2. Prediction overlays (BELIEF-REPORT channel): oracle answer predictions

Termination is applied first because stopping may truncate the trace.
"""

from typing import Optional

from ..models import Prediction, Trajectory, TrajectoryTurn, Guess, derive_mast_modes
from .base import PredictionOverlay, OverlayContext
from .calibrated import CalibratedOverlay
from .termination import (
    TerminationOverlay,
    TerminationContext,
    RationalTerminationOverlay,
)


class OverlayChain:
    """Chain of overlays applied in priority order.

    Handles both termination and prediction overlays:
    - Termination overlays decide stop/guess actions
    - Prediction overlays decide oracle answer predictions

    Termination is applied first, then predictions.
    """

    def __init__(
        self,
        prediction_overlays: Optional[list[PredictionOverlay]] = None,
        termination_overlays: Optional[list[TerminationOverlay]] = None,
        items: Optional[list[str]] = None,
        secret_index: Optional[int] = None,
        skip_prediction: bool = False,
        guess_at_end: bool = True
    ):
        """Initialize the overlay chain.

        Args:
            prediction_overlays: List of prediction overlays
            termination_overlays: List of termination overlays
            items: List of item names (needed for termination context)
            secret_index: The actual secret index (needed for guess scoring)
            skip_prediction: If True, don't apply any prediction overlay
            guess_at_end: If True, guess on final turn if no prior guess
        """
        self.prediction_overlays = prediction_overlays or []
        self.termination_overlays = termination_overlays or []
        self._sort_overlays()

        self.prediction_fallback = CalibratedOverlay()
        self.termination_fallback = RationalTerminationOverlay()
        self.skip_prediction = skip_prediction
        self.guess_at_end = guess_at_end

    def get_tags(self, final_feasible_size: int = 1, has_verification_claim: bool = False) -> list[str]:
        """Collect tags from all overlays.

        Args:
            final_feasible_size: Size of feasible set at end (for guess_at_budget tag)
            has_verification_claim: Whether any guess included a verification_claim
        """
        tags = []
        # Prediction tags
        if self.skip_prediction:
            tags.append("pred:none")
        else:
            for overlay in self.prediction_overlays:
                if hasattr(overlay, 'tag'):
                    tags.append(overlay.tag)
            if not self.prediction_overlays:
                tags.append(self.prediction_fallback.tag if hasattr(self.prediction_fallback, 'tag') else "pred:calibrated_argmax")
        # Termination tags
        for overlay in self.termination_overlays:
            if hasattr(overlay, 'tag'):
                tags.append(overlay.tag)
        if not self.termination_overlays:
            tags.append(self.termination_fallback.tag if hasattr(self.termination_fallback, 'tag') else "term:rational")
        if self.guess_at_end:
            # If guess happened at |S| > 1, it's budget-based not rational
            if final_feasible_size > 1:
                tags.append("term:guess_at_budget")
            else:
                tags.append("term:guess_at_end")
        # Verification claim tag (for FM-3.3 audit automation)
        if has_verification_claim:
            tags.append("verify:claim_present")
        return tags

        self.prediction_context = OverlayContext()
        self.termination_context = TerminationContext(
            items=items or [],
            secret_index=secret_index or 0
        )

    def _sort_overlays(self):
        """Sort overlays by descending priority."""
        self.prediction_overlays.sort(key=lambda o: -o.priority)
        self.termination_overlays.sort(key=lambda o: -o.priority)

    def add_prediction(self, overlay: PredictionOverlay):
        """Add a prediction overlay."""
        self.prediction_overlays.append(overlay)
        self._sort_overlays()

    def add_termination(self, overlay: TerminationOverlay):
        """Add a termination overlay."""
        self.termination_overlays.append(overlay)
        self._sort_overlays()

    def decide_termination(
        self,
        turn: TrajectoryTurn
    ) -> tuple[str, Optional[Guess], Optional[str]]:
        """Decide termination action using the overlay chain.

        Args:
            turn: The turn to decide for

        Returns:
            Tuple of (action, guess, stop_reason)
        """
        for overlay in self.termination_overlays:
            action, guess, reason = overlay.decide_action(
                turn, self.termination_context
            )
            if action != "continue":
                return action, guess, reason

        # Fallback to rational termination
        return self.termination_fallback.decide_action(
            turn, self.termination_context
        )

    def predict(self, turn: TrajectoryTurn) -> Prediction:
        """Generate prediction using the overlay chain.

        Args:
            turn: The turn to predict for

        Returns:
            Prediction from highest-priority matching overlay
        """
        for overlay in self.prediction_overlays:
            prediction = overlay.predict(turn, self.prediction_context)
            if prediction is not None:
                return prediction

        # Fallback to calibrated
        return self.prediction_fallback.predict(turn, self.prediction_context)

    def reset(self, items: Optional[list[str]] = None, secret_index: Optional[int] = None):
        """Reset all overlays and contexts.

        Args:
            items: List of item names for new trajectory
            secret_index: Secret index for new trajectory
        """
        self.prediction_context = OverlayContext()
        self.termination_context = TerminationContext(
            items=items or self.termination_context.items,
            secret_index=secret_index if secret_index is not None else self.termination_context.secret_index
        )

        for overlay in self.prediction_overlays:
            overlay.reset()
        for overlay in self.termination_overlays:
            overlay.reset()

    def apply_to_trajectory(
        self,
        trajectory: Trajectory,
        items: Optional[list[str]] = None
    ) -> Trajectory:
        """Apply termination and predictions to all turns.

        Order: termination first, then predictions.

        Args:
            trajectory: Trajectory to process
            items: List of item names (for termination context)

        Returns:
            Same trajectory with actions and predictions added
        """
        self.reset(
            items=items,
            secret_index=trajectory.secret_index
        )

        has_guessed = False
        has_stopped = False
        for i, turn in enumerate(trajectory.turns):
            is_last_turn = (i == len(trajectory.turns) - 1)

            # 1. Decide termination action
            action, guess, reason = self.decide_termination(turn)

            # Force guess on last turn if enabled and no prior guess/stop
            if is_last_turn and self.guess_at_end and not has_guessed and not has_stopped and action == "continue":
                action = "guess"
                guess = Guess(
                    secret_index=trajectory.secret_index,
                    secret=self.termination_context.items[trajectory.secret_index] if self.termination_context.items else trajectory.secret,
                    confidence=1.0 if turn.feasible_set_size_after == 1 else 1.0 / turn.feasible_set_size_after
                )
                reason = None

            turn.model_action = action
            turn.guess = guess
            turn.stop_reason = reason

            # Handle stop_accepted semantics
            if action in ("stop", "guess"):
                if is_last_turn:
                    turn.stop_accepted = True
                else:
                    # Stop/guess attempted but episode continues = rejected
                    turn.stop_accepted = False
                has_stopped = True

            # Score guess correctness
            if guess is not None:
                turn.guess_correct = (guess.secret_index == trajectory.secret_index)
                has_guessed = True

            # 2. Add prediction (unless skipped)
            if not self.skip_prediction:
                prediction = self.predict(turn)
                turn.prediction = prediction
            else:
                prediction = None
                turn.prediction = None

            # Update contexts
            self.termination_context.add_turn(turn)
            self.prediction_context.add_turn(turn, prediction)

        # Get final feasible set size for tag determination
        final_feasible_size = trajectory.turns[-1].feasible_set_size_after if trajectory.turns else 1

        # Check if any guess has a verification_claim
        has_verification_claim = any(
            t.guess is not None and t.guess.verification_claim is not None
            for t in trajectory.turns
        )

        # Set overlay tags on trajectory
        trajectory.overlay_tags = self.get_tags(final_feasible_size, has_verification_claim)

        # Derive MAST modes from world + overlays
        trajectory.target_mast_modes = derive_mast_modes(
            trajectory.trajectory_type,
            trajectory.overlay_tags
        )

        return trajectory


def apply_overlays(
    trajectory: Trajectory,
    prediction_overlays: Optional[list[PredictionOverlay]] = None,
    termination_overlays: Optional[list[TerminationOverlay]] = None,
    items: Optional[list[str]] = None
) -> Trajectory:
    """Convenience function to apply overlays to a trajectory.

    Args:
        trajectory: Trajectory to process
        prediction_overlays: List of prediction overlays
        termination_overlays: List of termination overlays
        items: List of item names

    Returns:
        Trajectory with actions and predictions applied
    """
    chain = OverlayChain(
        prediction_overlays=prediction_overlays,
        termination_overlays=termination_overlays,
        items=items,
        secret_index=trajectory.secret_index
    )
    return chain.apply_to_trajectory(trajectory, items)
