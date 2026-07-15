"""Strict parsing helpers for the maintained multi-step pathway JSON contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class PathwayStepValue:
    position: int
    step: int | None
    layer: str
    text: str
    substeps: tuple[str, ...] = ()

    def candidate_object(self, position: int) -> dict[str, object]:
        """Return a shuffle-safe object without original order metadata."""

        values = self.substeps or (self.text,)
        return {
            "step": position,
            "layer": f"candidate layer {position}",
            "substeps": [
                {"substep": index, "text": text}
                for index, text in enumerate(values)
            ],
        }


@dataclass(frozen=True)
class ParsedPathwayPayload:
    steps: tuple[PathwayStepValue, ...]
    phenotype: Any
    json_valid: bool
    schema_valid: bool
    error: str = ""

    @property
    def step_text(self) -> str:
        return "\n".join(step.text for step in self.steps)


def _strip_fence(value: str) -> str:
    text = value.strip()
    match = CODE_FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _step_from_value(value: Any, position: int) -> tuple[PathwayStepValue | None, bool]:
    if isinstance(value, dict):
        text = value.get("text")
        raw_substeps = value.get("substeps")
        substeps: tuple[str, ...] = ()
        if isinstance(raw_substeps, list):
            values = [
                str(item.get("text", "")).strip()
                for item in raw_substeps
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            substeps = tuple(values)
            if text is None and substeps:
                text = " ".join(substeps)
        if text is None:
            left = value.get("reactant") or value.get("source") or value.get("e1")
            relation = value.get("relation") or value.get("reaction") or value.get("r")
            right = value.get("product") or value.get("target") or value.get("e2")
            if left and relation and right:
                text = f"{left} {relation} {right}"
            else:
                return None, False
        text = str(text).strip()
        if not text:
            return None, False
        step_number = _as_int(value.get("step", value.get("step_index")))
        layer = str(value.get("layer", value.get("layer_id", ""))).strip()
        schema_valid = (
            step_number is not None
            and bool(layer)
            and isinstance(raw_substeps, list)
            and bool(substeps)
            and len(substeps) == len(raw_substeps)
        )
        return PathwayStepValue(position, step_number, layer, text, substeps), schema_valid
    if isinstance(value, str) and value.strip():
        return PathwayStepValue(position, None, "", value.strip()), False
    return None, False


def _v3_entity_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"canonical_id", "name"}
        and isinstance(value.get("canonical_id"), str)
        and bool(value["canonical_id"].strip())
        and isinstance(value.get("name"), str)
        and bool(value["name"].strip())
    )


def _v4_entity_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "canonical_id",
        "aliases",
        "name",
    }:
        return False
    canonical_id = value.get("canonical_id")
    aliases = value.get("aliases")
    return (
        isinstance(canonical_id, str)
        and bool(canonical_id.strip())
        and isinstance(aliases, list)
        and all(isinstance(alias, str) and alias.strip() for alias in aliases)
        and len(set(aliases)) == len(aliases)
        and canonical_id not in aliases
        and isinstance(value.get("name"), str)
        and bool(value["name"].strip())
    )


def _v4_action_valid(value: Any, event_type: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "relation_class",
        "subtypes",
        "reversibility",
    }:
        return False
    subtypes = value.get("subtypes")
    if (
        not isinstance(subtypes, list)
        or not all(isinstance(item, str) and item.strip() for item in subtypes)
        or len(set(subtypes)) != len(subtypes)
    ):
        return False
    if event_type == "reaction":
        return (
            value.get("kind") == "conversion"
            and value.get("relation_class") is None
            and subtypes == []
            and value.get("reversibility") in {"reversible", "irreversible"}
        )
    return (
        event_type == "relation"
        and value.get("kind") == "relation"
        and isinstance(value.get("relation_class"), str)
        and bool(value["relation_class"].strip())
        and value.get("reversibility") is None
    )


def _v4_layer_from_value(value: Any, position: int) -> tuple[PathwayStepValue | None, bool]:
    if not isinstance(value, dict):
        return None, False
    layer_index = value.get("layer_index")
    events = value.get("events")
    shape_valid = (
        set(value) == {"layer_index", "events"}
        and isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        and isinstance(events, list)
        and bool(events)
    )
    if not isinstance(events, list):
        return None, False
    texts: list[str] = []
    events_valid = True
    for event in events:
        if not isinstance(event, dict):
            events_valid = False
            continue
        event_type = event.get("event_type")
        sources = event.get("source")
        mediators = event.get("mediators")
        targets = event.get("target")
        text = str(event.get("text", "")).strip()
        event_valid = (
            set(event)
            == {"event_type", "source", "action", "mediators", "target", "text"}
            and event_type in {"relation", "reaction"}
            and isinstance(sources, list)
            and bool(sources)
            and all(_v4_entity_valid(item) for item in sources)
            and isinstance(mediators, list)
            and all(_v4_entity_valid(item) for item in mediators)
            and isinstance(targets, list)
            and bool(targets)
            and all(_v4_entity_valid(item) for item in targets)
            and _v4_action_valid(event.get("action"), event_type)
            and isinstance(event.get("text"), str)
            and bool(text)
        )
        events_valid = events_valid and event_valid
        if text:
            texts.append(text)
    if not texts:
        return None, False
    step_number = (
        layer_index
        if isinstance(layer_index, int) and not isinstance(layer_index, bool)
        else None
    )
    layer = f"layer {step_number}" if step_number is not None else ""
    return (
        PathwayStepValue(
            position=position,
            step=step_number,
            layer=layer,
            text=" ".join(texts),
            substeps=tuple(texts),
        ),
        shape_valid and events_valid and len(texts) == len(events),
    )


def _v3_layer_from_value(value: Any, position: int) -> tuple[PathwayStepValue | None, bool]:
    if not isinstance(value, dict):
        return None, False
    layer_index = value.get("layer_index")
    events = value.get("events")
    shape_valid = (
        set(value) == {"layer_index", "events"}
        and isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        and isinstance(events, list)
        and bool(events)
    )
    if not isinstance(events, list):
        return None, False
    texts: list[str] = []
    events_valid = True
    for event in events:
        if not isinstance(event, dict):
            events_valid = False
            continue
        text = str(event.get("text", "")).strip()
        sources = event.get("source")
        targets = event.get("target")
        event_valid = (
            set(event) == {"source", "relation", "target", "text"}
            and isinstance(sources, list)
            and bool(sources)
            and all(_v3_entity_valid(item) for item in sources)
            and isinstance(targets, list)
            and bool(targets)
            and all(_v3_entity_valid(item) for item in targets)
            and isinstance(event.get("relation"), str)
            and bool(event["relation"].strip())
            and isinstance(event.get("text"), str)
            and bool(text)
        )
        events_valid = events_valid and event_valid
        if text:
            texts.append(text)
    if not texts:
        return None, False
    step_number = layer_index if isinstance(layer_index, int) and not isinstance(layer_index, bool) else None
    layer = f"layer {step_number}" if step_number is not None else ""
    return (
        PathwayStepValue(
            position=position,
            step=step_number,
            layer=layer,
            text=" ".join(texts),
            substeps=tuple(texts),
        ),
        shape_valid and events_valid and len(texts) == len(events),
    )


def parse_pathway_payload(value: Any, *, allow_text_fallback: bool = True) -> ParsedPathwayPayload:
    """Parse v4/v3 ``remaining_layers``, v2 steps, and historical shapes.

    ``json_valid`` is true only for syntactically valid JSON. ``schema_valid``
    additionally requires the maintained object shape and structured step
    objects. Text fallback is retained for auditing old predictions, but never
    counts as valid JSON or valid schema.
    """

    loaded = value
    json_valid = not isinstance(value, str)
    error = ""
    if isinstance(value, str):
        text = _strip_fence(value)
        try:
            loaded = json.loads(text)
            json_valid = True
        except json.JSONDecodeError as exc:
            error = f"invalid_json:{exc.msg}"
            if not allow_text_fallback:
                return ParsedPathwayPayload((), None, False, False, error)
            lines = tuple(
                PathwayStepValue(index, None, "", line.strip())
                for index, line in enumerate(text.splitlines())
                if line.strip()
            )
            return ParsedPathwayPayload(lines, None, False, False, error)

    phenotype = None
    schema_valid = True
    if isinstance(loaded, dict):
        if "remaining_layers" in loaded:
            raw_layers = loaded.get("remaining_layers")
            if not isinstance(raw_layers, list):
                return ParsedPathwayPayload(
                    (),
                    None,
                    json_valid,
                    False,
                    "remaining_layers_is_not_list",
                )
            schema_version = loaded.get("schema_version")
            schema_valid = (
                set(loaded) == {"schema_version", "remaining_layers"}
                and schema_version
                in {"pathway_continuation_v4", "pathway_continuation_v3"}
                and bool(raw_layers)
            )
            parsed_layers: list[PathwayStepValue] = []
            previous_layer_index: int | None = None
            for position, raw_layer in enumerate(raw_layers):
                layer_parser = (
                    _v4_layer_from_value
                    if schema_version == "pathway_continuation_v4"
                    else _v3_layer_from_value
                )
                layer, layer_schema_valid = layer_parser(raw_layer, position)
                if layer is None:
                    schema_valid = False
                    continue
                if previous_layer_index is not None and (
                    layer.step is None or layer.step != previous_layer_index + 1
                ):
                    layer_schema_valid = False
                previous_layer_index = layer.step
                parsed_layers.append(layer)
                schema_valid = schema_valid and layer_schema_valid
            return ParsedPathwayPayload(
                tuple(parsed_layers),
                None,
                json_valid,
                schema_valid,
                error,
            )
        if "remaining_steps" in loaded:
            raw_steps = loaded.get("remaining_steps")
            phenotype = loaded.get("predicted_phenotype")
        elif "steps" in loaded:
            raw_steps = loaded.get("steps")
            phenotype = loaded.get("phenotype")
            schema_valid = False
        elif "pathway" in loaded:
            raw_steps = loaded.get("pathway")
            phenotype = loaded.get("phenotype")
            schema_valid = False
        else:
            return ParsedPathwayPayload((), loaded.get("predicted_phenotype"), json_valid, False, "missing_remaining_steps")
    elif isinstance(loaded, list):
        raw_steps = loaded
        schema_valid = False
    else:
        return ParsedPathwayPayload((), None, json_valid, False, "payload_is_not_object")

    if not isinstance(raw_steps, list):
        return ParsedPathwayPayload((), phenotype, json_valid, False, "remaining_steps_is_not_list")

    parsed: list[PathwayStepValue] = []
    for position, raw_step in enumerate(raw_steps):
        step, step_schema_valid = _step_from_value(raw_step, position)
        if step is None:
            schema_valid = False
            continue
        parsed.append(step)
        schema_valid = schema_valid and step_schema_valid
    return ParsedPathwayPayload(tuple(parsed), phenotype, json_valid, schema_valid, error)


def phenotype_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        text = value.get("text")
        return str(text).strip() if text is not None and str(text).strip() else None
    text = str(value).strip()
    return text or None


def canonical_candidate_json(steps: Iterable[PathwayStepValue]) -> str:
    """Serialize an order candidate without leaking original step/layer ids."""

    payload = {
        "remaining_steps": [step.candidate_object(index) for index, step in enumerate(steps)],
        "predicted_phenotype": None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_id(record: dict[str, Any], fallback: Any) -> Any:
    return record.get("sample_id", record.get("id", record.get("entry_id", fallback)))
