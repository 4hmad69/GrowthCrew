"""Text rules shared by the document exporters.

Generated content is generic JSON, and the Word and PDF exporters have to
make the same decisions about it: which characters a document can carry,
where a paragraph ends, what a missing value looks like, and what to call
an item in a list. Those decisions live here once, so the two formats
cannot drift apart.
"""

import re

from pydantic import JsonValue

# What a missing or empty value looks like in a document.
EMPTY = "\u2014"

# Characters XML 1.0 cannot represent. Tab, newline and carriage return are allowed.
_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_BLANK_LINES = re.compile(r"\n\s*\n")

# Keys whose text names a list item, in the order they are tried.
LABEL_KEYS = ("name", "title")


def clean_text(text: str) -> str:
    """Normalise line endings and drop characters a document cannot hold."""

    return _INVALID_XML.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def paragraph_texts(text: str) -> list[str]:
    """Text split into paragraphs at blank lines; empty pieces are dropped."""

    return [piece for piece in _BLANK_LINES.split(clean_text(text)) if piece.strip()]


def is_scalar(value: JsonValue) -> bool:
    """True for anything that is not a list or an object."""

    return not isinstance(value, dict | list)


def scalar_text(value: JsonValue) -> str:
    """A plain value as document text: a dash for nothing, Yes or No for a flag."""

    if value is None:
        return EMPTY
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text = clean_text(str(value))
    return text if text.strip() else EMPTY


def item_label(item: JsonValue, number: int) -> tuple[str, str | None]:
    """A heading for a list item, and the key it came from (so it is not repeated)."""

    if isinstance(item, dict):
        for key in LABEL_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return clean_text(value.strip()), key
    return f"Item {number}", None
