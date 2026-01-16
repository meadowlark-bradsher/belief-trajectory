"""Overconfident prediction overlays.

Variants that always report high confidence regardless of actual split.
"""

from typing import Optional

from ..models import Prediction, TrajectoryTurn
from .base import PredictionOverlay, OverlayContext


class OverconfidentOverlay(PredictionOverlay):
    """Overconfident predictions with calibrated argmax.

    - Same argmax as calibrated (YES if split_ratio > 0.5)
    - Always reports high confidence regardless of actual split

    Tag: pred:overconfident_argmax
    """

    def __init__(
        self,
        confidence: float = 0.95,
        priority: int = 0
    ):
        """Initialize overconfident overlay.

        Args:
            confidence: Fixed confidence value to always report
            priority: Overlay priority
        """
        super().__init__(priority)
        self.fixed_confidence = confidence

    @property
    def tag(self) -> str:
        return "pred:overconfident_argmax"

    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate overconfident prediction.

        Args:
            turn: The turn to predict for
            context: Overlay context (unused)

        Returns:
            Overconfident Prediction
        """
        p_yes = turn.split_ratio
        predicted_answer = p_yes > 0.5  # Strict inequality like calibrated

        return Prediction(
            predicted_answer=predicted_answer,
            confidence=self.fixed_confidence
        )


class AlwaysYesOverlay(PredictionOverlay):
    """Always predicts YES with high confidence.

    - Always predicts YES regardless of split ratio
    - Confidence either fixed or derived from split ratio

    Tag: pred:always_yes
    """

    def __init__(
        self,
        confidence: Optional[float] = 0.95,
        use_calibrated_confidence: bool = False,
        priority: int = 0
    ):
        """Initialize always-yes overlay.

        Args:
            confidence: Fixed confidence (used if use_calibrated_confidence=False)
            use_calibrated_confidence: If True, use |p_yes - 0.5| * 2
            priority: Overlay priority
        """
        super().__init__(priority)
        self.fixed_confidence = confidence
        self.use_calibrated_confidence = use_calibrated_confidence

    @property
    def tag(self) -> str:
        if self.use_calibrated_confidence:
            return "pred:always_yes_calibrated_conf"
        return "pred:always_yes"

    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate always-yes prediction.

        Args:
            turn: The turn to predict for
            context: Overlay context (unused)

        Returns:
            Always-yes Prediction
        """
        if self.use_calibrated_confidence:
            confidence = abs(turn.split_ratio - 0.5) * 2
        else:
            confidence = self.fixed_confidence

        return Prediction(
            predicted_answer=True,
            confidence=confidence
        )
