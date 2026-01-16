"""Commit-early prediction overlay.

Locks to MAP hypothesis after entropy drops below threshold,
ignoring subsequent evidence.
"""

from typing import Optional

from ..models import Prediction, TrajectoryTurn
from .base import PredictionOverlay, OverlayContext


class CommitEarlyOverlay(PredictionOverlay):
    """Commit-early predictions that lock to a hypothesis.

    - Monitors entropy and locks to MAP when entropy < threshold
    - After locking, always predicts consistent with MAP item
    - Ignores evidence that contradicts the committed hypothesis
    """

    def __init__(
        self,
        entropy_threshold: float = 3.0,  # bits (~8 items remaining)
        priority: int = 10
    ):
        """Initialize commit-early overlay.

        Args:
            entropy_threshold: Entropy level that triggers commitment
            priority: Overlay priority
        """
        super().__init__(priority)
        self.entropy_threshold = entropy_threshold
        self.committed: bool = False
        self.committed_pattern: Optional[bool] = None  # Pattern of predictions

    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate commit-early prediction.

        Args:
            turn: The turn to predict for
            context: Overlay context

        Returns:
            Committed Prediction or None
        """
        # Check if we should commit
        if not self.committed and turn.entropy_before <= self.entropy_threshold:
            self.committed = True
            # Commit to the likely answer
            self.committed_pattern = turn.split_ratio >= 0.5

        # If committed, always predict the committed pattern
        if self.committed:
            return Prediction(
                predicted_answer=self.committed_pattern,
                confidence=0.9  # High confidence after commitment
            )

        # Not yet committed, defer to next overlay
        return None

    def reset(self):
        """Reset commitment state between trajectories."""
        self.committed = False
        self.committed_pattern = None
