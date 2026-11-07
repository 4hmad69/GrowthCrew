"""Unit tests for the text rules the Word and PDF exporters share."""

import pytest

from backend.app.reports.text import (
    EMPTY,
    LABEL_KEYS,
    clean_text,
    is_scalar,
    item_label,
    paragraph_texts,
    scalar_text,
)


def test_the_empty_marker_is_an_em_dash() -> None:
    assert EMPTY == "\u2014"


def test_names_label_an_item_before_titles() -> None:
    assert LABEL_KEYS == ("name", "title")


# --- clean_text -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("plain", "plain"),
        ("", ""),
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\n\nb", "a\n\nb"),
        ("tab\there", "tab\there"),
        ("a\x00b", "ab"),
        ("a\x08b\x0bc\x0cd\x0ee\x1ff", "abcdef"),
        ("\ud800\udfff", ""),
        ("\ufffe\uffff", ""),
        ("\ufffd stays", "\ufffd stays"),
        ("Café ☕ 日本語", "Café ☕ 日本語"),
    ],
)
def test_clean_text(raw: str, cleaned: str) -> None:
    assert clean_text(raw) == cleaned


# --- paragraph_texts --------------------------------------------------------------------


def test_one_paragraph_stays_one() -> None:
    assert paragraph_texts("Just one.") == ["Just one."]


def test_blank_lines_split_paragraphs() -> None:
    assert paragraph_texts("One.\n\nTwo.\n\n\nThree.") == ["One.", "Two.", "Three."]


def test_a_blank_line_may_hold_spaces_and_tabs() -> None:
    assert paragraph_texts("One.\n \t \nTwo.") == ["One.", "Two."]


def test_single_newlines_stay_inside_a_paragraph() -> None:
    assert paragraph_texts("a\nb\nc") == ["a\nb\nc"]


def test_windows_line_endings_are_normalised_before_splitting() -> None:
    assert paragraph_texts("One.\r\n\r\nTwo.") == ["One.", "Two."]


def test_leading_whitespace_is_kept() -> None:
    assert paragraph_texts("  indented\n\n") == ["  indented"]


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", "\t \n \n"])
def test_text_with_nothing_in_it_has_no_paragraphs(blank: str) -> None:
    assert paragraph_texts(blank) == []


def test_unwritable_characters_are_dropped_before_splitting() -> None:
    assert paragraph_texts("a\x00b\n\nc") == ["ab", "c"]


# --- is_scalar and scalar_text ----------------------------------------------------------


@pytest.mark.parametrize("value", [None, True, False, 0, 3.5, "text", ""])
def test_plain_values_are_scalars(value: object) -> None:
    assert is_scalar(value) is True  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[], [1], {}, {"a": 1}])
def test_lists_and_objects_are_not_scalars(value: object) -> None:
    assert is_scalar(value) is False  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, EMPTY),
        (True, "Yes"),
        (False, "No"),
        (0, "0"),
        (7, "7"),
        (-5, "-5"),
        (2.5, "2.5"),
        ("text", "text"),
        ("", EMPTY),
        ("   ", EMPTY),
        ("\n\t", EMPTY),
        ("  kept  ", "  kept  "),
        ("a\x00b", "ab"),
        ("a\r\nb", "a\nb"),
    ],
)
def test_scalar_text(value: object, text: str) -> None:
    assert scalar_text(value) == text  # type: ignore[arg-type]


def test_a_boolean_is_not_written_as_a_number() -> None:
    assert scalar_text(True) == "Yes"
    assert scalar_text(1) == "1"


# --- item_label -------------------------------------------------------------------------


def test_a_named_item_is_labelled_by_its_name() -> None:
    assert item_label({"name": "Weekend Wen", "age": 3}, 1) == ("Weekend Wen", "name")


def test_a_title_is_used_when_there_is_no_name() -> None:
    assert item_label({"title": "Launch plan"}, 2) == ("Launch plan", "title")


def test_a_name_wins_over_a_title() -> None:
    assert item_label({"title": "T", "name": "N"}, 1) == ("N", "name")


def test_the_label_is_trimmed() -> None:
    assert item_label({"name": "  Wen \n"}, 1) == ("Wen", "name")


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"age": 3},
        {"name": ""},
        {"name": "   "},
        {"name": 5},
        {"name": None},
        {"name": ["x"]},
        "plain text",
        7,
        None,
        ["a"],
    ],
)
def test_an_item_with_no_usable_name_is_numbered(item: object) -> None:
    assert item_label(item, 4) == ("Item 4", None)  # type: ignore[arg-type]


def test_a_blank_name_falls_through_to_a_usable_title() -> None:
    assert item_label({"name": " ", "title": "Real"}, 1) == ("Real", "title")


def test_unwritable_characters_are_dropped_from_a_label() -> None:
    assert item_label({"name": "Wen\x00"}, 1) == ("Wen", "name")
