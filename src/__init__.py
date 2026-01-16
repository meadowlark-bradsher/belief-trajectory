"""Belief trajectory generator for 20 Questions stress testing."""

from .models import Trajectory, TrajectoryTurn, Prediction
from .config import GeneratorConfig
from .loader import load_cuq_dataset, CUQDataset
from .bitmask import (
    popcount,
    split_ratio,
    information_gain,
    update_state,
    entropy,
)

__all__ = [
    "Trajectory",
    "TrajectoryTurn",
    "Prediction",
    "GeneratorConfig",
    "load_cuq_dataset",
    "CUQDataset",
    "popcount",
    "split_ratio",
    "information_gain",
    "update_state",
    "entropy",
]
