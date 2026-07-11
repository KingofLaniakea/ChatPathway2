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


def parse_pathway_payload(value: Any, *, allow_text_fallback: bool = True) -> ParsedPathwayPayload:
    """Parse current ``remaining_steps`` JSON and selected historical shapes.

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
