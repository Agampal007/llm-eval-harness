"""Tokenisation and sentence splitting shared by the systems and the metrics."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been being by can could did do does for from had has have how i if in into
    is it its many may might not of on or our s so some such than that the their then there these they this
    to was were what when where which while who whom whose why will with would you your""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens, stop words included."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning: no stop words and no single letters."""
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BREAK.split(text.strip()) if s.strip()]


def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    """True if `needle` occurs as a contiguous run of tokens inside `haystack`."""
    if not needle:
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))
