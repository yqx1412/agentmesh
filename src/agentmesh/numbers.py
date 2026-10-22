"""Numbers in agent text, compared by value: the orchestrator checks that the final reply
states the values the steps looked up, and the Research guard checks that every number in
an answer is in a page it fetched."""

from __future__ import annotations

import re
from decimal import Decimal

_URL_TEXT = re.compile(r"https?://\S+")
# A number on its own: not part of a name like "Qwen3-8B", a version or a path.
_NUMBER = re.compile(r"(?<![\w.,/-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w-]|[.,]\d)")
# A numbered list item's marker ("1. ", "2) ").
_LIST_MARKER = re.compile(r"^[ \t>*_-]*\d+[.)][ \t]", re.MULTILINE)
# Counts and factors ("1 window", "x2 for K and V", "2 bytes") far more often than values.
TRIVIAL = frozenset({Decimal(0), Decimal(1), Decimal(2)})


def numbers(text: str) -> dict[Decimal, str]:
    """The numbers in ``text`` (URLs and list markers ignored), keyed by value so
    32,768 == 32768 and 8.0 == 8, each with the first way it was written."""
    found: dict[Decimal, str] = {}
    text = _LIST_MARKER.sub(" ", _URL_TEXT.sub(" ", text))
    for m in _NUMBER.finditer(text):
        found.setdefault(Decimal(m.group(1).replace(",", "") + (m.group(2) or "")), m.group(0))
    return found
