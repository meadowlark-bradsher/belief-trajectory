"""Refuses-revise prediction overlay.

After a contradiction (prediction != actual answer), keeps
predicting the wrong answer for K turns.
"""

from typing import Optional

from ..models import Prediction, TrajectoryTurn
from .base import PredictionOverlay, OverlayContext


class RefusesReviseOverlay(PredictionOverlay):
    """Refuses-revise predictions that resist updating.

    - Tracks when predictions are contradicted by actual answers
    - After contradiction, keeps predicting the same (wrong) answer
    - Persists for a specified number of turns
    """

    def __init__(
        self,
        persist_turns: int = 3,
        priority: int = 10
    ):
        """Initialize refuses-revise overlay.

        Args:
            persist_turns: Number of turns to persist wrong answer
            priority: Overlay priority
        """
        super().__init__(priority)
        self.persist_turns = persist_turns
        self.refused_answer: Optional[bool] = None
        self.turns_remaining: int = 0

    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate refuses-revise prediction.

        Args:
            turn: The turn to predict for
            context: Overlay context

        Returns:
            Refused-to-revise Prediction or None
        """
        # Check if we're in refused-to-revise mode
        if self.turns_remaining > 0:
            self.turns_remaining -= 1
            return Prediction(
                predicted_answer=self.refused_answer,
                confidence=0.7 - (0.1 * (self.persist_turns - self.turns_remaining))
            )

        # Check for contradiction in last turn
        if context.turn_history and context.prediction_history:
            last_turn = context.turn_history[-1]
            last_pred = context.prediction_history[-1]

            # Contradiction: predicted one thing, got the opposite
            if last_pred.predicted_answer != last_turn.answer:
                self.refused_answer = last_pred.predicted_answer
                self.turns_remaining = self.persist_turns

                return Prediction(
                    predicted_answer=self.refused_answer,
                    confidence=0.8  # Still confident despite contradiction
                )

        # No contradiction, defer to next overlay
        return None

    def reset(self):
        """Reset refused state between trajectories."""
        self.refused_answer = None
        self.turns_remaining = 0
