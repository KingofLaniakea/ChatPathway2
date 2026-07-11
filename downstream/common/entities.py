"""Conservative biochemical entity parsing used by tasks 1 and 2.

The parser is intentionally transparent rather than pretending to be a full
gene-name resolver. For publishable evaluation, pass a curated synonym map and
an external pathway library. Untagged all-caps symbols are accepted as a useful
fallback for the ChatPathway text format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


TAGGED_ENTITY = re.compile(
    r"\b(?:gene|protein|metabolite|compound|component|enzyme)\s+"
    r"([A-Za-z0-9][A-Za-z0-9._/\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9._/\-]*){0,4})",
    flags=re.IGNORECASE,
)
SYMBOL = re.compile(r"\b[A-Z][A-Z0-9\-]{1,14}\b")
STOP = re.compile(
    r"\s+(?:activates?|inhibits?|binds?|regulates?|phosphorylates?|"
    r"dephosphorylates?|ubiquitinates?|methylates?|acetylates?|degrades?|"
    r"cleaves?|forms?|produces?|(?:is\s+)?converts?(?:ed)?(?:\s+to)?|cataly[sz]es?|induces?|represses?|"
    r"expresses?|transports?|associates?|dissociates?|causes?|leads?\s+to|"
    r"results?\s+in|mediates?(?:\s+a\s+functional\s+link)?|(?:is\s+)?shared(?:\s+in)?|"
    r"and\s+(?:gene|protein|metabolite|compound|component)|"
    r"or\s+(?:gene|protein|metabolite|compound|component)|via|through|by)\b.*$",
    flags=re.IGNORECASE,
)
NOISE = {
    "AND", "OR", "THE", "THIS", "THAT", "WITH", "FROM", "GENE", "PROTEIN",
    "COMPONENT", "PATHWAY", "KEGG", "GO", "DNA", "RNA", "JSON", "LLM",
}


def normalize_entity(value: str, synonyms: dict[str, str] | None = None) -> str:
    value = STOP.sub("", str(value)).strip(" \t\n,;:.")
    value = re.sub(r"\s+", " ", value).upper()
    value = re.sub(r"^(?:GENE|PROTEIN|METABOLITE|COMPOUND|COMPONENT|ENZYME)\s+", "", value)
    if synonyms:
        value = synonyms.get(value, value)
    return value


def load_synonyms(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Synonym JSON must be an object of alias -> canonical entity.")
    return {normalize_entity(alias): normalize_entity(canonical) for alias, canonical in raw.items()}


def extract_entities(text: str, synonyms: dict[str, str] | None = None) -> set[str]:
    """Extract tagged phrases and untagged uppercase biological symbols."""
    candidates = [match.group(1) for match in TAGGED_ENTITY.finditer(str(text or ""))]
    candidates.extend(match.group(0) for match in SYMBOL.finditer(str(text or "")))
    return {
        entity
        for candidate in candidates
        if (entity := normalize_entity(candidate, synonyms)) and entity not in NOISE
    }


def precision_recall_f1(predicted: Iterable[str], target: Iterable[str]) -> dict[str, float | int]:
    predicted_set, target_set = set(predicted), set(target)
    tp = len(predicted_set & target_set)
    fp = len(predicted_set - target_set)
    fn = len(target_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else float(not target_set)
    recall = tp / (tp + fn) if tp + fn else float(not predicted_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": float(predicted_set == target_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
