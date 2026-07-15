#!/usr/bin/env python3
"""Task 1: conservative substep-level Conditional Step Prediction (CSP).

An atomic substep is one directed relation such as ``A activates B``.  The
preferred v4 gold/prediction schema is ``remaining_layers/events`` with
canonical participants, mediators, and a lossless structured action, so no
sentence parser is used. Historical ``remaining_substeps`` and v2/v3 outputs remain
readable; only the latter uses a conservative sentence/semicolon parser.
Ambiguous clauses are rejected and counted, never silently converted into
causal labels.

Phenotype is outside this task.  A missing phenotype is neither a negative
label nor an invalid CSP record.

Atomic events inside one graph layer are permutation-invariant by default.
Flat substep-order metrics require a strict structured payload plus independent
causal ordering provenance in the task manifest; sentence order is not enough.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from downstream.common.entities import extract_entities
from downstream.common.io import load_records, mean, write_json, write_rows
from downstream.common.pathway_json import parse_pathway_payload, record_id
from downstream.new_tasks.schemas import (
    SchemaError,
    load_json_object,
    require_choice,
    require_fields,
    require_integer,
    require_mapping,
    require_sequence,
    require_text,
)


RELATION_FORMS = {
    "regulates the expression of": "regulate",
    "can affect the activity of": "affect_activity",
    "affects the activity of": "affect_activity",
    "has an indirect link with": "indirect_link",
    "have an indirect link with": "indirect_link",
    "are converted to": "convert",
    "activates": "activate",
    "activate": "activate",
    "activated": "activate",
    "inhibits": "inhibit",
    "inhibit": "inhibit",
    "inhibited": "inhibit",
    "regulates": "regulate",
    "regulate": "regulate",
    "regulated": "regulate",
    "represses": "repress",
    "repress": "repress",
    "induces": "induce",
    "induce": "induce",
    "binds": "bind",
    "bind": "bind",
    "recruits": "recruit",
    "recruit": "recruit",
    "phosphorylates": "phosphorylate",
    "phosphorylate": "phosphorylate",
    "dephosphorylates": "dephosphorylate",
    "dephosphorylate": "dephosphorylate",
    "ubiquitinates": "ubiquitinate",
    "ubiquitinate": "ubiquitinate",
    "methylates": "methylate",
    "methylate": "methylate",
    "acetylates": "acetylate",
    "acetylate": "acetylate",
    "degrades": "degrade",
    "degrade": "degrade",
    "cleaves": "cleave",
    "cleave": "cleave",
    "produces": "produce",
    "produce": "produce",
    "forms": "form",
    "form": "form",
    "transports": "transport",
    "transport": "transport",
    "catalyzes": "catalyze",
    "catalyses": "catalyze",
    "converts": "convert",
    "converts to": "convert",
    "is converted to": "convert",
    "associates with": "associate",
    "dissociates from": "dissociate",
    "dissociates": "dissociate",
    "dissociate": "dissociate",
    "associates": "associate",
    "associate": "associate",
    "expresses": "express",
    "express": "express",
    "causes": "cause",
    "cause": "cause",
    "leads to": "lead_to",
    "results in": "result_in",
    "mediates a functional link with": "functional_link",
    "mediates a functional link": "functional_link",
    "is shared in successive reactions with": "successive_reaction_link",
    "is shared in successive reactions": "successive_reaction_link",
}
STRUCTURED_RELATION_FORMS = {
    "activation": "activate",
    "inhibition": "inhibit",
    "expression": "express",
    "repression": "repress",
    "phosphorylation": "phosphorylate",
    "dephosphorylation": "dephosphorylate",
    "glycosylation": "glycosylate",
    "ubiquitination": "ubiquitinate",
    "methylation": "methylate",
    "binding/association": "associate",
    "dissociation": "dissociate",
    "indirect effect": "indirect_effect",
    "state change": "state_change",
    "irreversible_conversion": "irreversible_conversion",
    "reversible_conversion": "reversible_conversion",
}
PARSER_VERSION = "atomic_relation_v2"
RELATION_RE = re.compile(
    r"(?<![\w-])(" + "|".join(sorted((re.escape(value) for value in RELATION_FORMS), key=len, reverse=True)) + r")(?![\w-])",
    flags=re.IGNORECASE,
)
CLAUSE_SPLIT_RE = re.compile(r"(?:\s*[.;!?]\s+|\s*;\s*|\n+)")
STEP_PREFIX_RE = re.compile(r"^\s*(?:step|substep)\s*\d+(?:\.\d+)?\s*[:.)-]\s*", re.IGNORECASE)
TAGGED_ARGUMENT_RE = re.compile(
    r"^(?:gene|protein|metabolite|compound|component|enzyme|pathway)\s+(.+)$",
    flags=re.IGNORECASE,
)
TAGGED_ARGUMENT_SPLIT_RE = re.compile(
    r"\s+(?:and|or)\s+(?=(?:gene|protein|metabolite|compound|component|enzyme|pathway)\s+)",
    flags=re.IGNORECASE,
)
ARGUMENT_LEADING_RE = re.compile(
    r"^(?:(?:the\s+)?(?:expression|activity)\s+of|between|of|to)\s+",
    flags=re.IGNORECASE,
)
ARGUMENT_TRAILING_RE = re.compile(
    r"\s+(?:"
    r"in\s+an?\s+(?:ir)?reversible\s+way"
    r"|at\s+the\s+pathway\s+level"
    r"|via\s+(?:gene|protein|metabolite|compound|component|enzyme|pathway)\s+"
    r"|and\s+exerts?\s+an?\s+indirect\s+effect"
    r"|in\s+this\s+pathway"
    r").*$",
    flags=re.IGNORECASE,
)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\n,;:.").casefold()


def _argument_terms(value: str) -> tuple[str, ...]:
    cleaned = STEP_PREFIX_RE.sub("", value).strip(" \t\n,;:.")
    cleaned = ARGUMENT_LEADING_RE.sub("", cleaned)
    cleaned = ARGUMENT_TRAILING_RE.sub("", cleaned).strip(" \t\n,;:.")
    tagged_terms: list[str] = []
    parts = TAGGED_ARGUMENT_SPLIT_RE.split(cleaned)
    for part in parts:
        match = TAGGED_ARGUMENT_RE.match(part.strip(" \t\n,;:."))
        if match is None:
            tagged_terms = []
            break
        normalized = _normalized_text(match.group(1))
        if not normalized:
            tagged_terms = []
            break
        tagged_terms.append(normalized)
    if tagged_terms:
        return tuple(sorted(set(tagged_terms)))
    entities = extract_entities(cleaned)
    if entities:
        return tuple(sorted(entity.casefold() for entity in entities))
    normalized = _normalized_text(cleaned)
    return (normalized,) if normalized else ()


def _structured_terms(value: Any, path: str) -> tuple[str, ...]:
    raw = [value] if isinstance(value, str) else list(require_sequence(value, path, nonempty=True))
    terms = tuple(_normalized_text(require_text(item, f"{path}[{index}]")) for index, item in enumerate(raw))
    return tuple(sorted(set(terms)))


@dataclass(frozen=True)
class AtomicSubstep:
    step: int
    substep: int
    source: tuple[str, ...]
    relation: str
    target: tuple[str, ...]
    text: str
    mediators: tuple[str, ...] = ()

    @property
    def key(
        self,
    ) -> tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]:
        return self.source, self.relation, self.mediators, self.target


@dataclass(frozen=True)
class SubstepPayload:
    substeps: tuple[AtomicSubstep, ...]
    strict_schema_valid: bool
    parser_valid: bool
    total_clauses: int
    unparsed_clauses: tuple[str, ...]
    errors: tuple[str, ...]


def parse_atomic_clause(text: str, step: int, substep: int) -> AtomicSubstep | None:
    clause = STEP_PREFIX_RE.sub("", str(text)).strip(" \t\n,;:.")
    matches = list(RELATION_RE.finditer(clause))
    if len(matches) != 1:
        return None
    match = matches[0]
    source = _argument_terms(clause[: match.start()])
    target = _argument_terms(clause[match.end() :])
    if not source or not target:
        return None
    relation = RELATION_FORMS[match.group(1).casefold()]
    return AtomicSubstep(step, substep, source, relation, target, clause)


def _structured_substep(value: Any, index: int) -> AtomicSubstep:
    item = require_mapping(value, f"remaining_substeps[{index}]")
    required = ("step", "substep", "source", "relation", "target")
    missing = [key for key in required if key not in item]
    if missing:
        raise SchemaError(f"remaining_substeps[{index}] is missing: {', '.join(missing)}.")
    step = require_integer(item["step"], f"remaining_substeps[{index}].step", minimum=0)
    substep = require_integer(item["substep"], f"remaining_substeps[{index}].substep", minimum=0)
    relation_value = require_text(item["relation"], f"remaining_substeps[{index}].relation").casefold()
    relation = RELATION_FORMS.get(relation_value, relation_value)
    if relation not in set(RELATION_FORMS.values()):
        raise SchemaError(f"remaining_substeps[{index}].relation is unsupported: {relation_value!r}.")
    source = _structured_terms(item["source"], f"remaining_substeps[{index}].source")
    target = _structured_terms(item["target"], f"remaining_substeps[{index}].target")
    text = str(item.get("text") or f"{' + '.join(source)} {relation} {' + '.join(target)}").strip()
    return AtomicSubstep(step, substep, source, relation, target, text)


def _load_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _structured_atomic_payload(loaded: dict[str, Any]) -> SubstepPayload:
    parsed = parse_pathway_payload(loaded, allow_text_fallback=False)
    if not parsed.schema_valid:
        return SubstepPayload(
            (),
            False,
            False,
            0,
            (),
            (
                parsed.error
                or "structured pathway continuation failed strict schema validation",
            ),
        )
    output: list[AtomicSubstep] = []
    raw_layers = loaded["remaining_layers"]
    for layer in raw_layers:
        step = int(layer["layer_index"])
        for substep, event in enumerate(layer["events"]):
            source = tuple(
                sorted(
                    _normalized_text(entity["canonical_id"])
                    for entity in event["source"]
                )
            )
            target = tuple(
                sorted(
                    _normalized_text(entity["canonical_id"])
                    for entity in event["target"]
                )
            )
            if loaded.get("schema_version") == "pathway_continuation_v4":
                mediators = tuple(
                    sorted(
                        _normalized_text(entity["canonical_id"])
                        for entity in event["mediators"]
                    )
                )
                action = event["action"]
                if action["kind"] == "conversion":
                    relation = (
                        "conversion:"
                        + _normalized_text(action["reversibility"])
                    )
                else:
                    relation_class = _normalized_text(action["relation_class"])
                    subtypes = "+".join(
                        _normalized_text(value) for value in action["subtypes"]
                    ) or "unspecified"
                    relation = f"{relation_class}:{subtypes}"
            else:
                mediators = ()
                raw_relation = _normalized_text(event["relation"])
                relation = STRUCTURED_RELATION_FORMS.get(raw_relation, raw_relation)
            output.append(
                AtomicSubstep(
                    step=step,
                    substep=substep,
                    source=source,
                    relation=relation,
                    target=target,
                    text=str(event["text"]).strip(),
                    mediators=mediators,
                )
            )
    return SubstepPayload(
        tuple(output),
        True,
        True,
        len(output),
        (),
        (),
    )


def parse_substeps(value: Any) -> SubstepPayload:
    loaded = _load_json(value)
    if isinstance(loaded, dict) and "remaining_layers" in loaded:
        return _structured_atomic_payload(loaded)
    if isinstance(loaded, dict) and "remaining_substeps" in loaded:
        try:
            raw = require_sequence(loaded["remaining_substeps"], "remaining_substeps")
            substeps = tuple(_structured_substep(item, index) for index, item in enumerate(raw))
        except SchemaError as exc:
            return SubstepPayload((), False, False, len(loaded.get("remaining_substeps") or []), (), (str(exc),))
        ordered = all(
            (left.step, left.substep) < (right.step, right.substep)
            for left, right in zip(substeps, substeps[1:])
        )
        error = () if ordered else ("remaining_substeps must be strictly ordered by (step, substep).",)
        return SubstepPayload(substeps, ordered, ordered, len(substeps), (), error)

    parsed_pathway = parse_pathway_payload(loaded)
    substeps: list[AtomicSubstep] = []
    unparsed: list[str] = []
    total = 0
    for step_position, step_value in enumerate(parsed_pathway.steps):
        # Historical layer_set_v1 preserves serialized source-item boundaries.
        # Sentence splitting is only a v2 fallback for aggregate ``text``.
        clauses = (
            list(step_value.substeps)
            if step_value.substeps
            else [
                clause.strip()
                for clause in CLAUSE_SPLIT_RE.split(step_value.text)
                if clause.strip()
            ]
        )
        for clause_position, clause in enumerate(clauses):
            total += 1
            parsed = parse_atomic_clause(clause, step_position, clause_position)
            if parsed is None:
                unparsed.append(clause)
            else:
                substeps.append(parsed)
    errors = tuple([parsed_pathway.error] if parsed_pathway.error else [])
    if not parsed_pathway.steps:
        errors += ("no pathway steps were found",)
    return SubstepPayload(
        tuple(substeps),
        False,
        bool(parsed_pathway.steps) and not unparsed and bool(substeps),
        total,
        tuple(unparsed),
        errors,
    )


def _set_f1(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    overlap = len(left_set & right_set)
    precision = overlap / len(left_set) if left_set else 0.0
    recall = overlap / len(right_set) if right_set else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _lcs_count(predicted: Sequence[AtomicSubstep], target: Sequence[AtomicSubstep]) -> int:
    row = [0] * (len(target) + 1)
    for guess in predicted:
        previous = row[:]
        for index, truth in enumerate(target, 1):
            row[index] = previous[index - 1] + 1 if guess.key == truth.key else max(previous[index], row[index - 1])
    return row[-1]


def _counter_scores(predicted: Counter[Any], target: Counter[Any]) -> tuple[float, float, float, int]:
    overlap = sum((predicted & target).values())
    predicted_count, target_count = sum(predicted.values()), sum(target.values())
    precision = overlap / predicted_count if predicted_count else float(target_count == 0)
    recall = overlap / target_count if target_count else float(predicted_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, overlap


def _layers(substeps: Sequence[AtomicSubstep]) -> list[list[AtomicSubstep]]:
    result: list[list[AtomicSubstep]] = []
    current_step: int | None = None
    for substep in substeps:
        if current_step != substep.step:
            result.append([])
            current_step = substep.step
        result[-1].append(substep)
    return result


def _layer_key(layer: Sequence[AtomicSubstep]) -> tuple[tuple[Any, int], ...]:
    return tuple(sorted(Counter(substep.key for substep in layer).items(), key=repr))


def _generic_lcs_count(left: Sequence[Any], right: Sequence[Any]) -> int:
    row = [0] * (len(right) + 1)
    for left_value in left:
        previous = row[:]
        for index, right_value in enumerate(right, 1):
            row[index] = previous[index - 1] + 1 if left_value == right_value else max(previous[index], row[index - 1])
    return row[-1]


def _f1_from_lcs(overlap: int, predicted_count: int, target_count: int) -> tuple[float, float, float]:
    precision = overlap / predicted_count if predicted_count else float(target_count == 0)
    recall = overlap / target_count if target_count else float(predicted_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _argument_union(layer: Sequence[AtomicSubstep], side: str) -> tuple[str, ...]:
    return tuple(sorted({term for substep in layer for term in getattr(substep, side)}))


def evaluate_pair(
    target_value: Any,
    predicted_value: Any,
    *,
    ordering_mode: str = "layer_set",
) -> dict[str, Any]:
    if ordering_mode not in {"layer_set", "causal_substep_sequence"}:
        raise SchemaError("ordering_mode must be 'layer_set' or 'causal_substep_sequence'.")
    target = parse_substeps(target_value)
    predicted = parse_substeps(predicted_value)
    strict_required = ordering_mode == "causal_substep_sequence"
    eligible = bool(
        target.substeps
        and target.parser_valid
        and predicted.parser_valid
        and (not strict_required or (target.strict_schema_valid and predicted.strict_schema_valid))
    )

    predicted_counter = Counter(substep.key for substep in predicted.substeps)
    target_counter = Counter(substep.key for substep in target.substeps)
    atomic_precision, atomic_recall, atomic_f1, _ = _counter_scores(predicted_counter, target_counter)
    predicted_layers, target_layers = _layers(predicted.substeps), _layers(target.substeps)
    predicted_layer_keys = [_layer_key(layer) for layer in predicted_layers]
    target_layer_keys = [_layer_key(layer) for layer in target_layers]
    layer_overlap = _generic_lcs_count(predicted_layer_keys, target_layer_keys)
    layer_precision, layer_recall, layer_f1 = _f1_from_lcs(
        layer_overlap, len(predicted_layers), len(target_layers)
    )

    if target_layers and predicted_layers:
        first_truth, first_guess = target_layers[0], predicted_layers[0]
        next_layer_exact = float(_layer_key(first_truth) == _layer_key(first_guess))
        _, _, next_layer_f1, _ = _counter_scores(
            Counter(substep.key for substep in first_guess),
            Counter(substep.key for substep in first_truth),
        )
        next_source = _set_f1(_argument_union(first_guess, "source"), _argument_union(first_truth, "source"))
        next_target = _set_f1(_argument_union(first_guess, "target"), _argument_union(first_truth, "target"))
        _, _, next_relation, _ = _counter_scores(
            Counter(substep.relation for substep in first_guess),
            Counter(substep.relation for substep in first_truth),
        )
    else:
        next_layer_exact = next_layer_f1 = next_source = next_relation = next_target = 0.0

    global_source = _set_f1(
        _argument_union(predicted.substeps, "source"), _argument_union(target.substeps, "source")
    )
    global_target = _set_f1(
        _argument_union(predicted.substeps, "target"), _argument_union(target.substeps, "target")
    )
    _, _, global_relation, _ = _counter_scores(
        Counter(substep.relation for substep in predicted.substeps),
        Counter(substep.relation for substep in target.substeps),
    )

    if ordering_mode == "causal_substep_sequence":
        exact_count = _lcs_count(predicted.substeps, target.substeps)
        exact_precision, exact_recall, ordered_f1 = _f1_from_lcs(
            exact_count, len(predicted.substeps), len(target.substeps)
        )
        next_exact: float | None = (
            float(target.substeps[0].key == predicted.substeps[0].key)
            if target.substeps and predicted.substeps
            else 0.0
        )
    else:
        exact_precision = exact_recall = ordered_f1 = next_exact = None
    return {
        "eligible": int(eligible),
        "ordering_mode": ordering_mode,
        "target_substeps": len(target.substeps),
        "predicted_substeps": len(predicted.substeps),
        "target_total_clauses": target.total_clauses,
        "predicted_total_clauses": predicted.total_clauses,
        "target_unparsed_clauses": len(target.unparsed_clauses),
        "predicted_unparsed_clauses": len(predicted.unparsed_clauses),
        "target_parser_validity": float(target.parser_valid),
        "prediction_parser_validity": float(predicted.parser_valid),
        "target_strict_schema_validity": float(target.strict_schema_valid),
        "prediction_strict_schema_validity": float(predicted.strict_schema_valid),
        "next_layer_event_set_exact": next_layer_exact,
        "next_layer_event_multiset_f1": next_layer_f1,
        "next_layer_source_f1": next_source,
        "next_layer_relation_f1": next_relation,
        "next_layer_target_f1": next_target,
        "atomic_event_multiset_precision": atomic_precision,
        "atomic_event_multiset_recall": atomic_recall,
        "atomic_event_multiset_f1": atomic_f1,
        "global_source_f1": global_source,
        "global_relation_f1": global_relation,
        "global_target_f1": global_target,
        "ordered_layer_precision": layer_precision,
        "ordered_layer_recall": layer_recall,
        "ordered_layer_f1": layer_f1,
        "layer_sequence_exact": float(predicted_layer_keys == target_layer_keys),
        "next_substep_exact": next_exact,
        "ordered_substep_precision": exact_precision,
        "ordered_substep_recall": exact_recall,
        "ordered_substep_f1": ordered_f1,
        "causal_substep_sequence_exact": (
            float(len(target.substeps) == len(predicted.substeps) and all(
                left.key == right.key for left, right in zip(target.substeps, predicted.substeps)
            ))
            if ordering_mode == "causal_substep_sequence"
            else None
        ),
        "target_parse_errors": " | ".join((*target.errors, *target.unparsed_clauses)),
        "prediction_parse_errors": " | ".join((*predicted.errors, *predicted.unparsed_clauses)),
    }


COMMON_METRIC_FIELDS = (
    "next_layer_event_set_exact",
    "next_layer_event_multiset_f1",
    "next_layer_source_f1",
    "next_layer_relation_f1",
    "next_layer_target_f1",
    "atomic_event_multiset_precision",
    "atomic_event_multiset_recall",
    "atomic_event_multiset_f1",
    "global_source_f1",
    "global_relation_f1",
    "global_target_f1",
    "ordered_layer_precision",
    "ordered_layer_recall",
    "ordered_layer_f1",
    "layer_sequence_exact",
)
CAUSAL_ORDER_METRIC_FIELDS = (
    "next_substep_exact",
    "ordered_substep_precision",
    "ordered_substep_recall",
    "ordered_substep_f1",
    "causal_substep_sequence_exact",
)


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    require_fields(
        value,
        ("schema_version", "dataset_id", "split", "model_checkpoint", "parser_version", "ordering_mode"),
        "manifest",
    )
    if require_integer(value["schema_version"], "manifest.schema_version") != 1:
        raise SchemaError("manifest.schema_version must be 1.")
    require_text(value["dataset_id"], "manifest.dataset_id")
    require_choice(value["split"], ("validation", "test"), "manifest.split")
    require_text(value["model_checkpoint"], "manifest.model_checkpoint")
    parser_version = require_text(value["parser_version"], "manifest.parser_version")
    if parser_version != PARSER_VERSION:
        raise SchemaError(f"manifest.parser_version must be {PARSER_VERSION!r} for this evaluator.")
    ordering_mode = require_choice(
        value["ordering_mode"], ("layer_set", "causal_substep_sequence"), "manifest.ordering_mode"
    )
    if ordering_mode == "causal_substep_sequence":
        provenance = require_mapping(value.get("ordering_provenance"), "manifest.ordering_provenance")
        require_fields(provenance, ("source", "version"), "manifest.ordering_provenance")
        require_text(provenance["source"], "manifest.ordering_provenance.source")
        require_text(provenance["version"], "manifest.ordering_provenance.version")
    return dict(value)


def evaluate_records(
    records: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    target_column: str = "answer",
    predicted_column: str = "predicted_answer",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = validate_manifest(manifest)
    source_records = list(records)
    if not source_records:
        raise SchemaError("Task 1 input contains no records.")
    rows = []
    for index, record in enumerate(source_records):
        if target_column not in record or predicted_column not in record:
            raise SchemaError(f"record[{index}] requires {target_column!r} and {predicted_column!r}.")
        rows.append({
            "sample_id": record_id(record, index),
            **evaluate_pair(
                record[target_column],
                record[predicted_column],
                ordering_mode=str(metadata["ordering_mode"]),
            ),
        })
    eligible = [row for row in rows if row["eligible"]]
    summary = {
        "task": "task1_substep_csp",
        "manifest": metadata,
        "num_records": len(rows),
        "eligible_records": len(eligible),
        "excluded_parser_invalid_records": len(rows) - len(eligible),
        "metrics": {
            field: (
                mean([float(row[field]) for row in eligible if row[field] is not None])
                if eligible and any(row[field] is not None for row in eligible)
                else None
            )
            for field in (
                *COMMON_METRIC_FIELDS,
                *CAUSAL_ORDER_METRIC_FIELDS,
            )
        },
        "coverage": {
            "target_parser_validity": mean([float(row["target_parser_validity"]) for row in rows]),
            "prediction_parser_validity": mean([float(row["prediction_parser_validity"]) for row in rows]),
            "target_strict_schema_validity": mean([float(row["target_strict_schema_validity"]) for row in rows]),
            "prediction_strict_schema_validity": mean([float(row["prediction_strict_schema_validity"]) for row in rows]),
        },
        "phenotype_policy": "Phenotype is not scored in Task 1; missing phenotype is not a negative label.",
        "ordering_policy": (
            "layer_set treats events within a graph layer as an unordered multiset. causal_substep_sequence is allowed "
            "only with explicit ordering provenance and strict structured substeps."
        ),
        "warning": "Parser-invalid clauses are excluded and counted. Audit natural-language parsing before reporting biological CSP.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL inference records.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True, help="Held-out dataset, checkpoint, and parser provenance JSON.")
    parser.add_argument("--target-column", default="answer")
    parser.add_argument("--predicted-column", default="predicted_answer")
    args = parser.parse_args()
    rows, summary = evaluate_records(
        load_records(args.input),
        load_json_object(args.manifest),
        target_column=args.target_column,
        predicted_column=args.predicted_column,
    )
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
