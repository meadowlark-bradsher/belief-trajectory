"""Calibrated prediction overlay.

Predicts YES if p_yes > 0.5, NO if p_yes < 0.5, random at exactly 0.5.
Confidence proportional to distance from 0.5.
"""

import random
from typing import Optional

from ..models import Prediction, TrajectoryTurn
from .base import PredictionOverlay, OverlayContext


class CalibratedOverlay(PredictionOverlay):
    """Calibrated predictions based on split ratio.

    - Predicts YES if split_ratio > 0.5, NO if < 0.5, random at exactly 0.5
    - Confidence = |split_ratio - 0.5| * 2 (0 at 50%, 1 at 0% or 100%)

    Tag: pred:calibrated_argmax
    """

    def __init__(self, priority: int = 0, seed: Optional[int] = None):
        super().__init__(priority)
        self.rng = random.Random(seed)

    @property
    def tag(self) -> str:
        return "pred:calibrated_argmax"

    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate calibrated prediction.

        Args:
            turn: The turn to predict for
            context: Overlay context (unused for calibrated)

        Returns:
            Calibrated Prediction
        """
        p_yes = turn.split_ratio

        # Random tie-break at exactly 0.5
        if p_yes == 0.5:
            predicted_answer = self.rng.random() < 0.5
        else:
            predicted_answer = p_yes > 0.5

        confidence = abs(p_yes - 0.5) * 2

        return Prediction(
            predicted_answer=predicted_answer,
            confidence=confidence
        )
