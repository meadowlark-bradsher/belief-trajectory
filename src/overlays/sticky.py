"""Sticky prediction overlay.

Persists previous high-confidence predictions even when the
evidence changes. Tests belief update failures.
"""

from typing import Optional

from ..models import Prediction, TrajectoryTurn
from .base import PredictionOverlay, OverlayContext


class StickyOverlay(PredictionOverlay):
    """Sticky predictions that persist after high confidence.

    - If no stuck state: predict like calibrated
    - If previous prediction had confidence >= threshold: keep that answer
    - Once "stuck", continues predicting the same answer
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        priority: int = 10  # Higher priority to override calibrated
    ):
        """Initialize sticky overlay.

        Args:
            confidence_threshold: Confidence level that triggers stickiness
            priority: Overlay priority
        """
        super().__init__(priority)
        self.confidence_threshold = confidence_threshold
        self.stuck_answer: Optional[bool] = None
        self.stuck_confidence: float = 0.0

    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate sticky prediction.

        Args:
            turn: The turn to predict for
            context: Overlay context

        Returns:
            Sticky Prediction
        """
        # Check if we're already stuck
        if self.stuck_answer is not None:
            # Keep predicting the stuck answer with decaying confidence
            return Prediction(
                predicted_answer=self.stuck_answer,
                confidence=self.stuck_confidence * 0.95  # Slight decay
            )

        # Check if last prediction triggered stickiness
        last_pred = context.last_prediction
        if last_pred and last_pred.confidence >= self.confidence_threshold:
            self.stuck_answer = last_pred.predicted_answer
            self.stuck_confidence = last_pred.confidence

            # Return the stuck answer for this turn too
            return Prediction(
                predicted_answer=self.stuck_answer,
                confidence=self.stuck_confidence
            )

        # Not stuck yet, defer to next overlay
        return None

    def reset(self):
        """Reset sticky state between trajectories."""
        self.stuck_answer = None
        self.stuck_confidence = 0.0
