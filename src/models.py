"""Data models for belief trajectories."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Prediction:
    """Model prediction for a turn."""
    predicted_answer: bool  # YES=True, NO=False
    confidence: float  # 0.0 to 1.0


@dataclass
class Guess:
    """A guess action by the agent."""
    secret_index: int
    secret: str
    confidence: float  # 0.0 to 1.0
    verification_claim: Optional[str] = None  # For FM-3.3: incorrect verification statement


@dataclass
class TrajectoryTurn:
    """Record of a single turn in a trajectory."""
    turn: int
    question_id: int
    question: str
    answer: bool  # True = YES, False = NO

    # Feasible set tracking (number of remaining possible secrets)
    feasible_set_size_before: int
    feasible_set_size_after: int

    # Entropy tracking (in bits)
    entropy_before: float
    entropy_after: float

    # Split characteristics for this question on the feasible set
    split_ratio: float  # proportion of feasible set that would answer YES
    branch_taken: Literal["yes", "no"]
    branch_probability: float  # probability of the branch that was taken

    # Action channel (termination overlay)
    model_action: Literal["continue", "guess", "stop"] = "continue"
    guess: Optional[Guess] = None
    guess_correct: Optional[bool] = None
    stop_reason: Optional[str] = None
    stop_accepted: Optional[bool] = None  # None if no stop attempted, True/False if attempted

    # Belief-report channel (prediction overlay)
    prediction: Optional[Prediction] = None

    # Bitmask state (stored as hex string for JSON serialization)
    state_before_hex: Optional[str] = None
    question_bitmask_hex: Optional[str] = None


@dataclass
class Trajectory:
    """Complete belief trajectory for a 20 Questions game."""
    trajectory_id: str
    trajectory_type: str  # "T1" through "T8"
    target_mast_modes: list[str]  # e.g., ["FM-1.1", "FM-2.6", "FM-3.3"]
    generation_mode: str  # "secret_first" or "path_first"

    secret: str
    secret_index: int

    turns: list[TrajectoryTurn] = field(default_factory=list)

    # Overlay tags for audit (e.g., ["pred:calibrated_argmax", "term:rational"])
    overlay_tags: list[str] = field(default_factory=list)

    # Generation metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate trajectory type."""
        valid_types = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"}
        if self.trajectory_type not in valid_types:
            raise ValueError(f"Invalid trajectory type: {self.trajectory_type}")

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def final_entropy(self) -> float:
        if not self.turns:
            return 0.0
        return self.turns[-1].entropy_after

    @property
    def initial_entropy(self) -> float:
        if not self.turns:
            return 0.0
        return self.turns[0].entropy_before


# MAST mode mappings for each trajectory type
TRAJECTORY_MAST_MAPPING: dict[str, list[str]] = {
    "T1": [],  # baseline/control
    "T2": ["FM-1.1", "FM-2.6", "FM-3.3"],  # Early collapse → rare-branch reversal
    "T3": ["FM-1.5", "FM-3.1"],  # Plateau → forced resolution
    "T4": ["FM-1.3"],  # Redundant loop
    "T5": ["FM-2.2"],  # Multi-modal ambiguity
    "T6": ["FM-2.6"],  # Prediction-belief mismatch
    "T7": ["FM-3.1", "FM-3.2"],  # Late shock after confidence
    "T8": ["FM-3.3"],  # Wrong-way update
}


def get_mast_modes(trajectory_type: str) -> list[str]:
    """Get the MAST failure modes targeted by a trajectory type (legacy)."""
    return TRAJECTORY_MAST_MAPPING.get(trajectory_type, [])


def derive_mast_modes(trajectory_type: str, overlay_tags: list[str]) -> list[str]:
    """Derive actually-instantiated MAST modes from world type + overlays.

    This is more accurate than get_mast_modes() because it checks
    whether the required overlay behavior is actually present.

    Args:
        trajectory_type: "T1" through "T8"
        overlay_tags: List of overlay tags (e.g., ["pred:calibrated_argmax", "term:rational"])

    Returns:
        List of MAST failure modes that are actually instantiated
    """
    modes = []
    tags_set = set(overlay_tags)

    # World-only modes (no overlay required)
    if trajectory_type == "T4":
        modes.append("FM-1.3")  # Step repetition (world-driven)
    if trajectory_type == "T5":
        modes.append("FM-2.2")  # Multi-modal ambiguity (world-driven)

    # Prediction-belief mismatch (FM-2.6): world creates mismatch opportunity, calibrated reveals it
    if trajectory_type in ("T2", "T6") and "pred:calibrated_argmax" in tags_set:
        modes.append("FM-2.6")

    # Premature termination (FM-3.1): requires premature_stop overlay
    if trajectory_type in ("T3", "T7") and "term:premature_stop" in tags_set:
        modes.append("FM-3.1")

    # Unaware of termination (FM-1.5): requires unaware overlay
    if trajectory_type == "T3" and "term:unaware" in tags_set:
        modes.append("FM-1.5")

    # Disobey spec / wrong guess (FM-1.1): T2 with wrong_guess
    if trajectory_type == "T2" and "term:wrong_guess" in tags_set:
        modes.append("FM-1.1")

    # Incorrect verification (FM-3.3): T8 with wrong_guess + verification_claim
    # The wrong_guess overlay generates a false verification claim
    if trajectory_type == "T8" and "term:wrong_guess" in tags_set:
        modes.append("FM-3.3")

    return sorted(set(modes))
