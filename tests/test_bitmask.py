"""Tests for bitmask operations."""

import math
import pytest
from src.bitmask import (
    popcount,
    full_state,
    item_to_bit,
    get_yes_set,
    get_no_set,
    split_ratio,
    entropy,
    information_gain,
    update_state,
    is_singleton,
    get_singleton_index,
    is_item_in_state,
    get_answer_for_secret,
    bitmask_overlap,
    to_hex,
    from_hex,
    items_in_state,
)


class TestPopcount:
    def test_zero(self):
        assert popcount(0) == 0

    def test_one(self):
        assert popcount(1) == 1

    def test_powers_of_two(self):
        for i in range(10):
            assert popcount(1 << i) == 1

    def test_all_ones(self):
        assert popcount(0b1111) == 4
        assert popcount(0xFF) == 8

    def test_mixed(self):
        assert popcount(0b10101010) == 4
        assert popcount(0b11001100) == 4


class TestFullState:
    def test_default_128(self):
        state = full_state()
        assert popcount(state) == 128

    def test_custom_size(self):
        assert popcount(full_state(8)) == 8
        assert popcount(full_state(16)) == 16
        assert popcount(full_state(1)) == 1

    def test_value(self):
        assert full_state(4) == 0b1111
        assert full_state(8) == 0xFF


class TestItemToBit:
    def test_index_zero(self):
        assert item_to_bit(0) == 1

    def test_index_one(self):
        assert item_to_bit(1) == 2

    def test_higher_indices(self):
        assert item_to_bit(7) == 128
        assert item_to_bit(10) == 1024


class TestGetYesSet:
    def test_full_overlap(self):
        state = 0b1111
        question = 0b1111
        assert get_yes_set(state, question) == 0b1111

    def test_no_overlap(self):
        state = 0b1111
        question = 0b11110000
        assert get_yes_set(state, question) == 0

    def test_partial_overlap(self):
        state = 0b1111
        question = 0b1100
        assert get_yes_set(state, question) == 0b1100


class TestGetNoSet:
    def test_full_question(self):
        state = 0b1111
        question = 0b1111
        assert get_no_set(state, question) == 0

    def test_no_question(self):
        state = 0b1111
        question = 0
        assert get_no_set(state, question) == 0b1111

    def test_partial(self):
        state = 0b1111
        question = 0b1100
        assert get_no_set(state, question) == 0b0011


class TestSplitRatio:
    def test_empty_state(self):
        assert split_ratio(0, 0b1111) == 0.0

    def test_balanced(self):
        state = 0b1111  # 4 items
        question = 0b1100  # 2 yes, 2 no
        assert split_ratio(state, question) == 0.5

    def test_all_yes(self):
        state = 0b1111
        question = 0b1111
        assert split_ratio(state, question) == 1.0

    def test_all_no(self):
        state = 0b1111
        question = 0
        assert split_ratio(state, question) == 0.0

    def test_skewed(self):
        state = 0b1111  # 4 items
        question = 0b0001  # 1 yes, 3 no
        assert split_ratio(state, question) == 0.25


class TestEntropy:
    def test_empty(self):
        assert entropy(0) == 0.0

    def test_singleton(self):
        assert entropy(1) == 0.0
        assert entropy(0b1000) == 0.0

    def test_two_items(self):
        assert entropy(0b11) == 1.0

    def test_four_items(self):
        assert entropy(0b1111) == 2.0

    def test_128_items(self):
        assert entropy(full_state(128)) == 7.0


class TestInformationGain:
    def test_empty_state(self):
        assert information_gain(0, 0b1111) == 0.0

    def test_singleton(self):
        assert information_gain(1, 0b1111) == 0.0

    def test_no_information(self):
        # All YES or all NO gives no information
        state = 0b1111
        assert information_gain(state, 0b1111) == 0.0  # all YES
        assert information_gain(state, 0) == 0.0  # all NO

    def test_balanced_split(self):
        # Balanced split gives maximum information gain
        state = 0b1111  # 4 items, H=2
        question = 0b1100  # 2 yes, 2 no
        ig = information_gain(state, question)
        # H_before = 2, H_after = 0.5*1 + 0.5*1 = 1, IG = 1
        assert ig == pytest.approx(1.0)

    def test_skewed_split(self):
        state = 0b1111  # 4 items
        question = 0b0001  # 1 yes, 3 no
        ig = information_gain(state, question)
        # H_before = 2
        # H_after = 0.25*0 + 0.75*log2(3) = 0.75*1.585 = 1.189
        # IG = 2 - 1.189 = 0.811
        assert 0 < ig < 1.0


class TestUpdateState:
    def test_yes_answer(self):
        state = 0b1111
        question = 0b1100
        new_state = update_state(state, question, answer=True)
        assert new_state == 0b1100

    def test_no_answer(self):
        state = 0b1111
        question = 0b1100
        new_state = update_state(state, question, answer=False)
        assert new_state == 0b0011


class TestIsSingleton:
    def test_zero(self):
        assert is_singleton(0) == False

    def test_single_bit(self):
        assert is_singleton(1) == True
        assert is_singleton(2) == True
        assert is_singleton(128) == True

    def test_multiple_bits(self):
        assert is_singleton(3) == False
        assert is_singleton(0b1111) == False


class TestGetSingletonIndex:
    def test_not_singleton(self):
        assert get_singleton_index(0) is None
        assert get_singleton_index(0b11) is None

    def test_singleton(self):
        assert get_singleton_index(1) == 0
        assert get_singleton_index(2) == 1
        assert get_singleton_index(8) == 3
        assert get_singleton_index(128) == 7


class TestIsItemInState:
    def test_in_state(self):
        state = 0b1010
        assert is_item_in_state(state, 1) == True
        assert is_item_in_state(state, 3) == True

    def test_not_in_state(self):
        state = 0b1010
        assert is_item_in_state(state, 0) == False
        assert is_item_in_state(state, 2) == False


class TestGetAnswerForSecret:
    def test_yes(self):
        question = 0b1010
        assert get_answer_for_secret(question, 1) == True
        assert get_answer_for_secret(question, 3) == True

    def test_no(self):
        question = 0b1010
        assert get_answer_for_secret(question, 0) == False
        assert get_answer_for_secret(question, 2) == False


class TestBitmaskOverlap:
    def test_identical(self):
        assert bitmask_overlap(0b1111, 0b1111) == 1.0

    def test_no_overlap(self):
        assert bitmask_overlap(0b1111, 0b11110000) == 0.0

    def test_partial(self):
        # 0b1100 and 0b1010: intersection=0b1000 (1), union=0b1110 (3)
        assert bitmask_overlap(0b1100, 0b1010) == pytest.approx(1/3)

    def test_empty(self):
        assert bitmask_overlap(0, 0) == 0.0


class TestHexConversion:
    def test_to_hex(self):
        assert to_hex(255) == "ff"
        assert to_hex(0) == "0"
        assert to_hex(16) == "10"

    def test_from_hex(self):
        assert from_hex("ff") == 255
        assert from_hex("0") == 0
        assert from_hex("0xff") == 255

    def test_roundtrip(self):
        for val in [0, 1, 255, 12345, full_state(128)]:
            assert from_hex(to_hex(val)) == val


class TestItemsInState:
    def test_empty(self):
        assert items_in_state(0) == []

    def test_single(self):
        assert items_in_state(1) == [0]
        assert items_in_state(8) == [3]

    def test_multiple(self):
        assert items_in_state(0b1010) == [1, 3]
        assert items_in_state(0b1111) == [0, 1, 2, 3]
