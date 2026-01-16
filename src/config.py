"""Configuration for the belief trajectory generator."""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class DataConfig:
    """Paths to CUQ dataset files."""
    questions_path: str = "data/questions.jsonl"
    items_path: str = "data/items.txt"


@dataclass
class GeneratorConfig:
    """Configuration for trajectory generation."""
    # Number of items (secrets) in the dataset
    num_items: int = 128

    # Default trajectory length
    max_turns: int = 20
    min_turns: int = 5

    # Split ratio thresholds
    balanced_min: float = 0.4
    balanced_max: float = 0.6
    near_balanced_min: float = 0.45
    near_balanced_max: float = 0.55
    extreme_threshold: float = 0.1  # <0.1 or >0.9 is extreme

    # Entropy thresholds
    low_entropy_threshold: float = 2.0  # bits

    # Random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class OverlayConfig:
    """Configuration for prediction overlays."""
    # Overconfident overlay
    overconfident_confidence: float = 0.95

    # Sticky overlay
    sticky_threshold: float = 0.8  # confidence threshold to become sticky

    # Commit-early overlay
    commit_early_entropy_threshold: float = 3.0  # bits

    # Refuses-revise overlay
    refuses_revise_turns: int = 3  # turns to keep wrong answer


@dataclass
class OutputConfig:
    """Configuration for output."""
    output_dir: str = "belief-trajectory/outputs"
    indent: int = 2
    include_bitmasks: bool = False  # Include hex bitmasks in output


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from JSON file or return defaults."""
    if path is None:
        return Config()

    with open(path) as f:
        data = json.load(f)

    config = Config()

    if "data" in data:
        config.data = DataConfig(**data["data"])
    if "generator" in data:
        config.generator = GeneratorConfig(**data["generator"])
    if "overlay" in data:
        config.overlay = OverlayConfig(**data["overlay"])
    if "output" in data:
        config.output = OutputConfig(**data["output"])

    return config


def save_config(config: Config, path: str) -> None:
    """Save configuration to JSON file."""
    from dataclasses import asdict

    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
