"""Load CUQ dataset (questions and items)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Question:
    """A single question with its bitmask."""
    question_id: int
    question: str
    bitmask: int  # 128-bit integer


@dataclass
class CUQDataset:
    """The CUQ dataset containing questions and items."""
    items: list[str]  # 128 items (secrets)
    questions: list[Question]  # 122,913 questions

    # Lookup indices
    _item_to_index: dict[str, int] = field(default_factory=dict, repr=False)
    _question_id_to_index: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Build lookup indices."""
        self._item_to_index = {item: i for i, item in enumerate(self.items)}
        self._question_id_to_index = {q.question_id: i for i, q in enumerate(self.questions)}

    @property
    def num_items(self) -> int:
        return len(self.items)

    @property
    def num_questions(self) -> int:
        return len(self.questions)

    def get_item_index(self, item: str) -> Optional[int]:
        """Get the index of an item (secret)."""
        return self._item_to_index.get(item)

    def get_item(self, index: int) -> str:
        """Get an item by index."""
        return self.items[index]

    def get_question_by_id(self, question_id: int) -> Optional[Question]:
        """Get a question by its ID."""
        idx = self._question_id_to_index.get(question_id)
        if idx is None:
            return None
        return self.questions[idx]

    def get_question(self, index: int) -> Question:
        """Get a question by its list index."""
        return self.questions[index]


def load_items(items_path: str | Path) -> list[str]:
    """Load items (secrets) from items.txt.

    Args:
        items_path: Path to items.txt file

    Returns:
        List of 128 item names
    """
    items_path = Path(items_path)
    with open(items_path, "r") as f:
        items = [line.strip() for line in f if line.strip()]
    return items


def load_questions(questions_path: str | Path) -> list[Question]:
    """Load questions from questions.jsonl.

    Args:
        questions_path: Path to questions.jsonl file

    Returns:
        List of Question objects
    """
    questions_path = Path(questions_path)
    questions = []

    with open(questions_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            questions.append(Question(
                question_id=data["question_id"],
                question=data["question"],
                bitmask=data["bitmask"]
            ))

    return questions


def load_cuq_dataset(
    questions_path: str | Path = "flan-oracle/cuq/release/questions.jsonl",
    items_path: str | Path = "flan-oracle/cuq/release/items.txt",
    base_path: Optional[str | Path] = None
) -> CUQDataset:
    """Load the complete CUQ dataset.

    Args:
        questions_path: Path to questions.jsonl (relative or absolute)
        items_path: Path to items.txt (relative or absolute)
        base_path: Optional base path to prepend

    Returns:
        CUQDataset with questions and items loaded
    """
    if base_path is not None:
        base_path = Path(base_path)
        questions_path = base_path / questions_path
        items_path = base_path / items_path

    items = load_items(items_path)
    questions = load_questions(questions_path)

    return CUQDataset(items=items, questions=questions)
