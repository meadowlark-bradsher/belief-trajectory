"""Base class for prediction overlays."""

from abc import ABC, abstractmethod
from typing import Optional

from ..models import Prediction, TrajectoryTurn


class OverlayContext:
    """Context passed to overlays for stateful predictions.

    Contains information about the trajectory state needed for
    overlays that depend on history.
    """

    def __init__(self):
        self.turn_history: list[TrajectoryTurn] = []
        self.prediction_history: list[Prediction] = []
        self.committed_answer: Optional[bool] = None  # For commit-early
        self.stuck_answer: Optional[bool] = None  # For sticky
        self.stuck_confidence: float = 0.0
        self.contradiction_turn: Optional[int] = None  # For refuses-revise

    def add_turn(self, turn: TrajectoryTurn, prediction: Prediction):
        """Record a turn and its prediction."""
        self.turn_history.append(turn)
        self.prediction_history.append(prediction)

    @property
    def current_turn(self) -> int:
        """Get the current turn number (1-indexed)."""
        return len(self.turn_history) + 1

    @property
    def last_prediction(self) -> Optional[Prediction]:
        """Get the most recent prediction."""
        if self.prediction_history:
            return self.prediction_history[-1]
        return None


class PredictionOverlay(ABC):
    """Abstract base class for prediction overlays.

    Overlays generate predictions for each turn based on the
    split ratio and trajectory context. They can be chained
    together with priority ordering.
    """

    def __init__(self, priority: int = 0):
        """Initialize the overlay.

        Args:
            priority: Higher priority overlays are applied first.
                      If an overlay returns a prediction, lower
                      priority overlays are skipped.
        """
        self.priority = priority

    @abstractmethod
    def predict(
        self,
        turn: TrajectoryTurn,
        context: OverlayContext
    ) -> Optional[Prediction]:
        """Generate a prediction for the turn.

        Args:
            turn: The turn to predict for
            context: Overlay context with history

        Returns:
            Prediction or None (to defer to next overlay)
        """
        pass

    def reset(self):
        """Reset any internal state (called between trajectories)."""
        pass
