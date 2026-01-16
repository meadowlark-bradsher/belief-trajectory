"""JSON output persistence for trajectories."""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Trajectory
from .config import OutputConfig


def trajectory_to_dict(
    trajectory: Trajectory,
    include_bitmasks: bool = False
) -> dict:
    """Convert trajectory to JSON-serializable dict.

    Args:
        trajectory: Trajectory to convert
        include_bitmasks: Whether to include hex bitmask fields

    Returns:
        Dictionary representation
    """
    data = asdict(trajectory)

    # Remove bitmask fields if not needed
    if not include_bitmasks:
        for turn in data["turns"]:
            turn.pop("state_before_hex", None)
            turn.pop("question_bitmask_hex", None)

    return data


def save_trajectory(
    trajectory: Trajectory,
    output_dir: str | Path,
    config: Optional[OutputConfig] = None
) -> Path:
    """Save a single trajectory to JSON.

    Args:
        trajectory: Trajectory to save
        output_dir: Directory to save to
        config: Output configuration

    Returns:
        Path to saved file
    """
    config = config or OutputConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{trajectory.trajectory_type}_{trajectory.trajectory_id}.json"
    filepath = output_dir / filename

    data = trajectory_to_dict(trajectory, config.include_bitmasks)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=config.indent)

    return filepath


def save_batch(
    trajectories: list[Trajectory],
    output_dir: str | Path,
    config: Optional[OutputConfig] = None,
    batch_name: Optional[str] = None
) -> Path:
    """Save a batch of trajectories to a single JSONL file.

    Args:
        trajectories: List of trajectories
        output_dir: Directory to save to
        config: Output configuration
        batch_name: Optional name for the batch file

    Returns:
        Path to saved file
    """
    config = config or OutputConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if batch_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"batch_{timestamp}"

    filepath = output_dir / f"{batch_name}.jsonl"

    with open(filepath, "w") as f:
        for trajectory in trajectories:
            data = trajectory_to_dict(trajectory, config.include_bitmasks)
            f.write(json.dumps(data) + "\n")

    return filepath


def load_trajectory(filepath: str | Path) -> Trajectory:
    """Load a single trajectory from JSON.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded Trajectory
    """
    from .models import TrajectoryTurn, Prediction, Guess

    with open(filepath) as f:
        data = json.load(f)

    turns = []
    for turn_data in data["turns"]:
        # Handle prediction
        prediction = None
        if turn_data.get("prediction"):
            prediction = Prediction(**turn_data["prediction"])
        turn_data["prediction"] = prediction

        # Handle guess
        guess = None
        if turn_data.get("guess"):
            guess = Guess(**turn_data["guess"])
        turn_data["guess"] = guess

        turns.append(TrajectoryTurn(**turn_data))

    data["turns"] = turns
    return Trajectory(**data)


def load_batch(filepath: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of Trajectory objects
    """
    from .models import TrajectoryTurn, Prediction, Guess

    trajectories = []

    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            turns = []
            for turn_data in data["turns"]:
                # Handle prediction
                prediction = None
                if turn_data.get("prediction"):
                    prediction = Prediction(**turn_data["prediction"])
                turn_data["prediction"] = prediction

                # Handle guess
                guess = None
                if turn_data.get("guess"):
                    guess = Guess(**turn_data["guess"])
                turn_data["guess"] = guess

                turns.append(TrajectoryTurn(**turn_data))

            data["turns"] = turns
            trajectories.append(Trajectory(**data))

    return trajectories


def save_batch_summary(
    trajectories: list[Trajectory],
    output_dir: str | Path,
    batch_name: Optional[str] = None
) -> Path:
    """Save a summary of batch statistics.

    Args:
        trajectories: List of trajectories
        output_dir: Directory to save to
        batch_name: Optional name for the summary file

    Returns:
        Path to saved summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if batch_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"summary_{timestamp}"

    # Compute summary statistics
    type_counts = {}
    mode_counts = {}
    avg_turns_by_type = {}

    for traj in trajectories:
        # Count by type
        type_counts[traj.trajectory_type] = type_counts.get(traj.trajectory_type, 0) + 1

        # Count by mode
        mode_counts[traj.generation_mode] = mode_counts.get(traj.generation_mode, 0) + 1

        # Track turns by type
        if traj.trajectory_type not in avg_turns_by_type:
            avg_turns_by_type[traj.trajectory_type] = []
        avg_turns_by_type[traj.trajectory_type].append(len(traj.turns))

    # Compute averages
    for ttype, turn_counts in avg_turns_by_type.items():
        avg_turns_by_type[ttype] = sum(turn_counts) / len(turn_counts)

    summary = {
        "total_trajectories": len(trajectories),
        "by_type": type_counts,
        "by_mode": mode_counts,
        "avg_turns_by_type": avg_turns_by_type,
        "generated_at": datetime.now().isoformat(),
    }

    filepath = output_dir / f"{batch_name}.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    return filepath
