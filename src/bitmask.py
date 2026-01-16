"""Integer-based bitmask operations for 128-item feasible sets.

This module uses Python integers as bitmasks, where bit i represents item i.
The CUQ dataset stores bitmasks as large integers (128 bits for 128 items).
"""

import math
from typing import Optional


def popcount(state: int) -> int:
    """Count number of 1s in the state bitmask.

    Args:
        state: Integer bitmask

    Returns:
        Number of set bits
    """
    return bin(state).count('1')


def full_state(num_items: int = 128) -> int:
    """Create a state with all items possible.

    Args:
        num_items: Number of items in the game

    Returns:
        Integer with first num_items bits set
    """
    return (1 << num_items) - 1


def item_to_bit(item_index: int) -> int:
    """Convert item index to single-bit mask.

    Args:
        item_index: 0-based index of the item

    Returns:
        Integer with only that bit set
    """
    return 1 << item_index


def get_yes_set(state: int, question_mask: int) -> int:
    """Get the set of items that would answer YES.

    Args:
        state: Current feasible set
        question_mask: Question's bitmask (1 = YES for that item)

    Returns:
        Intersection of state and question_mask
    """
    return state & question_mask


def get_no_set(state: int, question_mask: int) -> int:
    """Get the set of items that would answer NO.

    Args:
        state: Current feasible set
        question_mask: Question's bitmask

    Returns:
        Items in state but not in question_mask
    """
    return state & ~question_mask


def split_ratio(state: int, question_mask: int) -> float:
    """Compute the YES/total split ratio.

    Args:
        state: Current feasible set
        question_mask: Question's bitmask

    Returns:
        Proportion of feasible items that would answer YES (0.0 to 1.0)
    """
    total = popcount(state)
    if total == 0:
        return 0.0
    yes_count = popcount(get_yes_set(state, question_mask))
    return yes_count / total


def entropy(state: int) -> float:
    """Compute entropy of the feasible set (uniform prior).

    Args:
        state: Current feasible set

    Returns:
        Entropy in bits (log2 of feasible set size)
    """
    k = popcount(state)
    if k <= 1:
        return 0.0
    return math.log2(k)


def information_gain(state: int, question_mask: int) -> float:
    """Compute expected information gain for a question.

    Args:
        state: Current feasible set
        question_mask: Question's bitmask

    Returns:
        Expected information gain in bits
    """
    k = popcount(state)
    if k <= 1:
        return 0.0

    k_yes = popcount(get_yes_set(state, question_mask))
    k_no = k - k_yes

    if k_yes == 0 or k_no == 0:
        return 0.0  # No information gained

    p_yes = k_yes / k
    p_no = k_no / k

    H_before = math.log2(k)
    H_after = p_yes * math.log2(k_yes) + p_no * math.log2(k_no)

    return H_before - H_after


def update_state(state: int, question_mask: int, answer: bool) -> int:
    """Update state based on question answer.

    Args:
        state: Current feasible set
        question_mask: Question's bitmask
        answer: True for YES, False for NO

    Returns:
        New state after filtering
    """
    if answer:
        return get_yes_set(state, question_mask)
    else:
        return get_no_set(state, question_mask)


def is_singleton(state: int) -> bool:
    """Check if state contains exactly one item.

    Args:
        state: Current feasible set

    Returns:
        True if exactly one bit is set
    """
    return state > 0 and (state & (state - 1)) == 0


def get_singleton_index(state: int) -> Optional[int]:
    """Get the index of the single item in a singleton state.

    Args:
        state: State that should be a singleton

    Returns:
        Item index, or None if not a singleton
    """
    if not is_singleton(state):
        return None
    return state.bit_length() - 1


def is_item_in_state(state: int, item_index: int) -> bool:
    """Check if an item is in the feasible set.

    Args:
        state: Current feasible set
        item_index: 0-based index of the item

    Returns:
        True if the item is possible
    """
    return bool(state & item_to_bit(item_index))


def get_answer_for_secret(question_mask: int, secret_index: int) -> bool:
    """Determine the answer for a specific secret.

    Args:
        question_mask: Question's bitmask
        secret_index: 0-based index of the secret

    Returns:
        True if the secret answers YES, False for NO
    """
    return bool(question_mask & item_to_bit(secret_index))


def bitmask_overlap(mask1: int, mask2: int) -> float:
    """Compute Jaccard similarity between two bitmasks.

    Args:
        mask1: First bitmask
        mask2: Second bitmask

    Returns:
        Jaccard similarity (intersection / union), 0.0 to 1.0
    """
    intersection = popcount(mask1 & mask2)
    union = popcount(mask1 | mask2)
    if union == 0:
        return 0.0
    return intersection / union


def to_hex(state: int) -> str:
    """Convert state to hex string for JSON serialization.

    Args:
        state: Integer bitmask

    Returns:
        Hex string (without '0x' prefix)
    """
    return hex(state)[2:]


def from_hex(hex_str: str) -> int:
    """Convert hex string back to integer.

    Args:
        hex_str: Hex string (with or without '0x' prefix)

    Returns:
        Integer bitmask
    """
    if hex_str.startswith('0x'):
        return int(hex_str, 16)
    return int(hex_str, 16)


def items_in_state(state: int) -> list[int]:
    """Get list of item indices in the feasible set.

    Args:
        state: Current feasible set

    Returns:
        List of 0-based item indices
    """
    indices = []
    i = 0
    s = state
    while s:
        if s & 1:
            indices.append(i)
        s >>= 1
        i += 1
    return indices
