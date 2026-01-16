"""Base class for trajectory generators."""

from abc import ABC, abstractmethod
from typing import Optional
import uuid

from ..models import Trajectory, TrajectoryTurn, get_mast_modes
from ..config import GeneratorConfig
from ..loader import CUQDataset
from ..index import QuestionIndex


class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators.

    Subclasses implement either secret-first or path-first generation modes.
    """

    def __init__(
        self,
        dataset: CUQDataset,
        config: Optional[GeneratorConfig] = None,
        seed: Optional[int] = None
    ):
        """Initialize the generator.

        Args:
            dataset: The CUQ dataset
            config: Generator configuration
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.config = config or GeneratorConfig()
        self.index = QuestionIndex(dataset, seed=seed)
        self.seed = seed

    @property
    @abstractmethod
    def generation_mode(self) -> str:
        """Return 'secret_first' or 'path_first'."""
        pass

    @abstractmethod
    def generate(
        self,
        trajectory_type: str,
        secret: Optional[str] = None,
        secret_index: Optional[int] = None,
        max_turns: Optional[int] = None
    ) -> Trajectory:
        """Generate a trajectory of the specified type.

        Args:
            trajectory_type: "T1" through "T8"
            secret: Optional secret to use (for secret-first mode)
            secret_index: Optional secret index (alternative to secret)
            max_turns: Maximum number of turns

        Returns:
            Generated Trajectory
        """
        pass

    def _generate_id(self) -> str:
        """Generate a unique trajectory ID."""
        return str(uuid.uuid4())[:8]

    def _create_trajectory(
        self,
        trajectory_type: str,
        secret: str,
        secret_index: int,
        turns: list[TrajectoryTurn]
    ) -> Trajectory:
        """Create a Trajectory object with metadata.

        Args:
            trajectory_type: "T1" through "T8"
            secret: The secret item
            secret_index: Index of the secret
            turns: List of turns

        Returns:
            Complete Trajectory object
        """
        return Trajectory(
            trajectory_id=self._generate_id(),
            trajectory_type=trajectory_type,
            target_mast_modes=get_mast_modes(trajectory_type),
            generation_mode=self.generation_mode,
            secret=secret,
            secret_index=secret_index,
            turns=turns,
            metadata={
                "seed": self.seed,
                "num_items": self.dataset.num_items,
            }
        )
