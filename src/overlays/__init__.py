"""Overlays for generating model behavior on trajectories.

Two overlay families:
1. Termination overlays: Control stop/guess behavior (ACTION channel)
2. Prediction overlays: Control oracle answer predictions (BELIEF-REPORT channel)

Apply termination overlays BEFORE prediction overlays.
"""

from .base import PredictionOverlay
from .calibrated import CalibratedOverlay
from .overconfident import OverconfidentOverlay, AlwaysYesOverlay
from .sticky import StickyOverlay
from .commit_early import CommitEarlyOverlay
from .refuses_revise import RefusesReviseOverlay
from .termination import (
    TerminationOverlay,
    TerminationContext,
    RationalTerminationOverlay,
    PrematureStopOverlay,
    UnawareTerminationOverlay,
    WrongGuessOverlay,
)
from .chain import OverlayChain, apply_overlays

__all__ = [
    # Prediction overlays (belief-report channel)
    "PredictionOverlay",
    "CalibratedOverlay",
    "OverconfidentOverlay",
    "AlwaysYesOverlay",
    "StickyOverlay",
    "CommitEarlyOverlay",
    "RefusesReviseOverlay",
    # Termination overlays (action channel)
    "TerminationOverlay",
    "TerminationContext",
    "RationalTerminationOverlay",
    "PrematureStopOverlay",
    "UnawareTerminationOverlay",
    "WrongGuessOverlay",
    # Chain
    "OverlayChain",
    "apply_overlays",
]
