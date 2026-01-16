"""Termination overlays for controlling stop/guess behavior.

These overlays operate on the ACTION channel (model_action, guess),
separate from prediction overlays which operate on the BELIEF-REPORT channel.

Apply termination overlays BEFORE prediction overlays, since stopping
may truncate or deactivate later turns.
"""

from abc import ABC, abstractmethod
from typing import Optional
import random

from ..models import TrajectoryTurn, Guess


class TerminationContext:
    """Context for termination overlays."""

    def __init__(self, items: list[str], secret_index: int):
        """Initialize context.

        Args:
            items: List of all item names
            secret_index: The actual secret's index
        """
        self.items = items
        self.secret_index = secret_index
        self.turn_history: list[TrajectoryTurn] = []
        self.has_stopped: bool = False
        self.stop_turn: Optional[int] = None

    def add_turn(self, turn: TrajectoryTurn):
        """Record a turn."""
        self.turn_history.append(turn)
        if turn.model_action in ("guess", "stop"):
            self.has_stopped = True
            self.stop_turn = turn.turn

    @property
    def current_turn(self) -> int:
        return len(self.turn_history) + 1


class TerminationOverlay(ABC):
    """Abstract base class for termination overlays."""

    def __init__(self, priority: int = 0):
        self.priority = priority

    @abstractmethod
    def decide_action(
        self,
        turn: TrajectoryTurn,
        context: TerminationContext
    ) -> tuple[str, Optional[Guess], Optional[str]]:
        """Decide the action for this turn.

        Args:
            turn: The turn to decide for
            context: Termination context with history

        Returns:
            Tuple of (action, guess, stop_reason)
            action: "continue", "guess", or "stop"
            guess: Guess object if action is "guess"
            stop_reason: Reason string if action is "stop"
        """
        pass

    def reset(self):
        """Reset any internal state."""
        pass


class RationalTerminationOverlay(TerminationOverlay):
    """Rational termination: guess when |S|=1, continue otherwise.

    This is the baseline "correct" behavior.

    Tag: term:rational
    """

    def __init__(self, priority: int = 0):
        super().__init__(priority)

    @property
    def tag(self) -> str:
        return "term:rational"

    def decide_action(
        self,
        turn: TrajectoryTurn,
        context: TerminationContext
    ) -> tuple[str, Optional[Guess], Optional[str]]:
        # If feasible set is singleton, guess
        if turn.feasible_set_size_after == 1:
            # In a real implementation, we'd identify which item
            # For now, we know the secret from context
            return "guess", Guess(
                secret_index=context.secret_index,
                secret=context.items[context.secret_index],
                confidence=1.0
            ), None

        return "continue", None, None


class PrematureStopOverlay(TerminationOverlay):
    """Force premature stop when entropy is still high.

    For FM-3.1 (Premature termination): stop before objectives are met.

    Triggers stop when:
    - feasible_set_size >= threshold, OR
    - entropy >= entropy_threshold
    - AND turn >= min_turn (to allow some game progress)

    Tag: term:premature_stop
    """

    def __init__(
        self,
        feasible_threshold: int = 16,
        entropy_threshold: float = 4.0,
        min_turn: int = 3,
        stop_probability: float = 1.0,
        priority: int = 10,
        seed: Optional[int] = None
    ):
        """Initialize premature stop overlay.

        Args:
            feasible_threshold: Stop if |S| >= this
            entropy_threshold: Stop if H >= this (bits)
            min_turn: Don't stop before this turn
            stop_probability: Probability of stopping when triggered
            priority: Overlay priority
            seed: Random seed
        """
        super().__init__(priority)
        self.feasible_threshold = feasible_threshold
        self.entropy_threshold = entropy_threshold
        self.min_turn = min_turn
        self.stop_probability = stop_probability
        self.rng = random.Random(seed)
        self.triggered = False

    @property
    def tag(self) -> str:
        return "term:premature_stop"

    def decide_action(
        self,
        turn: TrajectoryTurn,
        context: TerminationContext
    ) -> tuple[str, Optional[Guess], Optional[str]]:
        if self.triggered or context.has_stopped:
            return "continue", None, None  # Already handled

        if turn.turn < self.min_turn:
            return "continue", None, None

        # Check premature stop conditions (high entropy = early in game)
        is_premature = (
            turn.feasible_set_size_after >= self.feasible_threshold or
            turn.entropy_after >= self.entropy_threshold
        )

        if is_premature and self.rng.random() < self.stop_probability:
            self.triggered = True
            return "stop", None, f"Premature stop at entropy {turn.entropy_after:.2f} bits, |S|={turn.feasible_set_size_after}"

        return "continue", None, None

    def reset(self):
        self.triggered = False


class UnawareTerminationOverlay(TerminationOverlay):
    """Force continued questioning even when should stop.

    For FM-1.5 (Unaware of termination conditions): keep going past
    when the agent should guess/stop.

    Triggers when |S| <= threshold, but forces "continue" for N extra turns.

    Tag: term:unaware
    """

    def __init__(
        self,
        feasible_threshold: int = 2,
        extra_turns: int = 3,
        priority: int = 10
    ):
        """Initialize unaware termination overlay.

        Args:
            feasible_threshold: Should stop when |S| <= this
            extra_turns: Continue for this many extra turns after threshold
            priority: Overlay priority
        """
        super().__init__(priority)
        self.feasible_threshold = feasible_threshold
        self.extra_turns = extra_turns
        self.turns_past_threshold = 0
        self.threshold_reached = False

    @property
    def tag(self) -> str:
        return "term:unaware"

    def decide_action(
        self,
        turn: TrajectoryTurn,
        context: TerminationContext
    ) -> tuple[str, Optional[Guess], Optional[str]]:
        # Check if we've reached the "should stop" threshold
        if turn.feasible_set_size_after <= self.feasible_threshold:
            if not self.threshold_reached:
                self.threshold_reached = True
                self.turns_past_threshold = 0

            self.turns_past_threshold += 1

            # Force continue for extra_turns past threshold
            if self.turns_past_threshold <= self.extra_turns:
                return "continue", None, None

        return "continue", None, None  # Defer to other overlays

    def reset(self):
        self.turns_past_threshold = 0
        self.threshold_reached = False


class WrongGuessOverlay(TerminationOverlay):
    """Force an incorrect guess with false verification claim.

    For FM-3.3 (Incorrect verification): guess a secret not in the feasible set
    and provide a verification claim that incorrectly confirms the guess.

    Tag: term:wrong_guess
    """

    def __init__(
        self,
        trigger_turn: Optional[int] = None,
        trigger_entropy: Optional[float] = 2.0,  # Default: trigger when |S| <= 4
        priority: int = 10,
        seed: Optional[int] = None,
        generate_verification: bool = True
    ):
        """Initialize wrong guess overlay.

        Args:
            trigger_turn: Trigger on this specific turn
            trigger_entropy: Trigger when entropy drops below this (default 2.0 = |S|~4)
            priority: Overlay priority
            seed: Random seed
            generate_verification: If True, generate a false verification claim
        """
        super().__init__(priority)
        self.trigger_turn = trigger_turn
        self.trigger_entropy = trigger_entropy
        self.rng = random.Random(seed)
        self.triggered = False
        self.generate_verification = generate_verification

    @property
    def tag(self) -> str:
        return "term:wrong_guess"

    def _generate_verification_claim(
        self,
        wrong_secret: str,
        turn_history: list[TrajectoryTurn]
    ) -> str:
        """Generate a false verification claim for the wrong guess.

        The claim incorrectly asserts that the guess is consistent with
        the question-answer history, when it actually isn't.
        """
        # Find a question where the wrong guess would give a different answer
        for t in turn_history:
            # The claim asserts consistency with history (falsely)
            answer_word = "Yes" if t.answer else "No"
            return (
                f"The secret must be '{wrong_secret}' because "
                f"the answer to '{t.question}' was {answer_word}, "
                f"which is consistent with '{wrong_secret}'."
            )
        return f"Based on the answers so far, the secret is '{wrong_secret}'."

    def decide_action(
        self,
        turn: TrajectoryTurn,
        context: TerminationContext
    ) -> tuple[str, Optional[Guess], Optional[str]]:
        if self.triggered or context.has_stopped:
            return "continue", None, None

        should_trigger = False

        if self.trigger_turn is not None and turn.turn == self.trigger_turn:
            should_trigger = True
        elif self.trigger_entropy is not None and turn.entropy_after <= self.trigger_entropy:
            should_trigger = True

        if should_trigger:
            self.triggered = True
            # Guess a wrong secret (not the actual one)
            wrong_index = context.secret_index
            while wrong_index == context.secret_index:
                wrong_index = self.rng.randint(0, len(context.items) - 1)

            wrong_secret = context.items[wrong_index]

            # Generate false verification claim
            verification = None
            if self.generate_verification:
                verification = self._generate_verification_claim(
                    wrong_secret, context.turn_history
                )

            return "guess", Guess(
                secret_index=wrong_index,
                secret=wrong_secret,
                confidence=0.9,  # Confident but wrong
                verification_claim=verification
            ), None

        return "continue", None, None

    def reset(self):
        self.triggered = False
