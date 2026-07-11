"""Deterministic, dependency-free splitting of pathway layers into substeps.

Processed ChatPathway layers may contain several graph-grounded event strings
joined into one ``text`` field.  This module recovers sentence-level events
without requiring a general-purpose NLP sentence tokenizer.  When the original
``source_items`` are available they are authoritative boundaries and their
indices are retained as provenance.

The splitter is deliberately conservative: it handles the punctuation used by
the generated KEGG text while protecting decimals, dotted identifiers, common
abbreviations, initials, and enzyme commission numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence


_ABBREVIATIONS = frozenset(
    {
        "approx.",
        "ca.",
        "cf.",
        "dept.",
        "dr.",
        "e.g.",
        "eq.",
        "etc.",
        "fig.",
        "hr.",
        "i.e.",
        "inc.",
        "max.",
        "min.",
        "mol.",
        "mr.",
        "mrs.",
        "ms.",
        "no.",
        "prof.",
        "ref.",
        "sec.",
        "st.",
        "vol.",
        "vs.",
        "wt.",
    }
)
_DOTTED_ABBREVIATION_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")
_WORD_RE = re.compile(r"[A-Za-z]+")
_CLOSING_PUNCTUATION = frozenset("\"'\u2019\u201d)]}")


@dataclass(frozen=True)
class PathwaySubstep:
    """One sentence-level biological event with stable source provenance.

    ``index`` and ``source_item_index`` are zero-based.  The latter is ``None``
    when parsing only an aggregate ``text`` value.
    """

    index: int
    text: str
    source_item_index: Optional[int]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "substep": self.index,
            "text": self.text,
            "source_item_index": self.source_item_index,
        }


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _token_ending_at(text: str, period_index: int) -> str:
    start = period_index
    while start > 0 and not text[start - 1].isspace():
        if text[start - 1] in "([{\"'\u2018\u201c":
            break
        start -= 1
    return text[start : period_index + 1]


def _next_word(text: str, start: int) -> str:
    match = _WORD_RE.search(text, start)
    return match.group(0) if match else ""


def _is_protected_period(text: str, index: int) -> bool:
    """Return whether ``text[index]`` is internal punctuation, not a boundary."""

    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""

    # Decimal values and EC/accession-number components: 3.5, EC 1.2.3.4.
    if previous.isdigit() and following.isdigit():
        return True

    # Dots embedded in a token cannot be sentence boundaries: e.g., p.R175H.
    if following and not following.isspace() and following not in _CLOSING_PUNCTUATION:
        return True

    token = _token_ending_at(text, index)
    lowered = token.casefold()
    if lowered in _ABBREVIATIONS or _DOTTED_ABBREVIATION_RE.fullmatch(token):
        return True

    # Scientific-name initials such as E. coli.  Restrict this protection to a
    # lowercase following word so ``Protein A. Gene B ...`` still splits.
    if len(token) == 2 and token[0].isalpha() and token[1] == ".":
        next_word = _next_word(text, index + 1)
        if next_word and next_word[0].islower():
            return True

    return False


def _split_one_source(text: str) -> list[str]:
    """Split one source item while preserving terminal punctuation."""

    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if character not in ".!?":
            index += 1
            continue
        if character == "." and _is_protected_period(normalized, index):
            index += 1
            continue

        # Keep ellipses/repeated terminal punctuation and closing quotes or
        # brackets attached to the preceding substep.
        end = index + 1
        while end < len(normalized) and normalized[end] in ".!?":
            end += 1
        while end < len(normalized) and normalized[end] in _CLOSING_PUNCTUATION:
            end += 1

        if end == len(normalized) or normalized[end].isspace():
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            while end < len(normalized) and normalized[end].isspace():
                end += 1
            start = end
            index = end
            continue
        index = end

    tail = normalized[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def parse_substeps(
    text: str,
    *,
    source_items: Optional[Sequence[str]] = None,
) -> tuple[PathwaySubstep, ...]:
    """Parse an aggregate pathway step into stable, ordered substeps.

    Args:
        text: Aggregate layer text.  It is used when ``source_items`` is absent
            or contains no non-empty entries.
        source_items: Original event strings from the processed JSON layer.
            Non-empty items are authoritative boundaries and their original
            zero-based indices are retained.  Repeated items are not removed.

    Returns:
        An immutable tuple of :class:`PathwaySubstep` records.

    Raises:
        TypeError: If ``text`` or an entry in ``source_items`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(source_items, (str, bytes)):
        raise TypeError("source_items must be a sequence of strings, not a string")

    sources: list[tuple[Optional[int], str]] = []
    if source_items is not None:
        for source_index, item in enumerate(source_items):
            if not isinstance(item, str):
                raise TypeError("every source_items entry must be a string")
            if _normalize_text(item):
                sources.append((source_index, item))
    if not sources and _normalize_text(text):
        sources.append((None, text))

    parsed: list[PathwaySubstep] = []
    for source_index, source_text in sources:
        for chunk in _split_one_source(source_text):
            parsed.append(
                PathwaySubstep(
                    index=len(parsed),
                    text=chunk,
                    source_item_index=source_index,
                )
            )
    return tuple(parsed)


def split_substeps(
    text: str,
    *,
    source_items: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Convenience API returning only ordered substep strings."""

    return tuple(
        substep.text for substep in parse_substeps(text, source_items=source_items)
    )


__all__ = ["PathwaySubstep", "parse_substeps", "split_substeps"]
