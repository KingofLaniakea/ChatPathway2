"""Deterministic, provenance-pinned text rendering for v4 graph events.

The historical ``processed/*.json`` files are paragraph views.  They cannot be
joined back to event IDs after layer-level text deduplication.  The archived
Step 12 producer is nevertheless deterministic: its relation template table is
pinned under :mod:`dataprocess.assets` and can be applied directly to each
``processed_graph`` event before paragraph assembly.

``legacy_*`` functions reproduce that producer surface form.  ``audited_*``
functions preserve the same action content while fixing grammar and the one
known direction-reversing GErel template.  No language model is used to create
gold text.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


TEMPLATE_ASSET = (
    Path(__file__).resolve().parent
    / "assets"
    / "kegg_step12_relation_templates_v2.json"
)

TEXT_SOURCE_LEGACY_CORRECTED = "kegg_step12_v2_template_corrected"
TEXT_SOURCE_CANONICAL_FALLBACK = "canonical_action_fallback_v4"


@lru_cache(maxsize=1)
def template_asset() -> dict[str, Any]:
    payload = json.loads(TEMPLATE_ASSET.read_text(encoding="utf-8"))
    if set(payload) != {"provenance", "relation_templates"}:
        raise ValueError("Step 12 template asset has unexpected top-level keys")
    if not isinstance(payload["relation_templates"], dict):
        raise ValueError("Step 12 relation_templates must be an object")
    return payload


def template_provenance() -> dict[str, str]:
    provenance = template_asset()["provenance"]
    if not isinstance(provenance, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in provenance.items()
    ):
        raise ValueError("Step 12 template provenance is invalid")
    return dict(provenance)


def _name(entity: Mapping[str, Any]) -> str:
    value = str(entity.get("name") or "").strip()
    if not value:
        raise ValueError("event entity lacks a display name")
    return value


def joined_names(entities: Sequence[Mapping[str, Any]]) -> str:
    if not entities:
        raise ValueError("event entity collection is empty")
    return " and ".join(_name(entity) for entity in entities)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _missing_interaction_context(subtypes: Sequence[str]) -> str:
    context = [value for value in subtypes if value != "missing interaction"]
    if not context:
        return "interaction"
    key = ", ".join(context)
    return {
        "activation": "activation interaction",
        "binding/association": "binding or association interaction",
        "dissociation": "dissociation interaction",
        "expression": "expression-related interaction",
        "inhibition": "inhibition interaction",
        "repression": "repression-related interaction",
    }.get(key, f"{key}-related interaction")


def legacy_relation_text(
    *,
    relation_class: str,
    subtypes: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    mediators: Sequence[Mapping[str, Any]],
) -> str | None:
    """Reproduce the event-level Step 12 text before paragraph deduplication."""

    source = joined_names(sources)
    target = joined_names(targets)
    normalized_subtypes = tuple(str(value).strip().casefold() for value in subtypes)
    if "missing interaction" in normalized_subtypes:
        context = _missing_interaction_context(normalized_subtypes)
        return _capitalize(
            f"KEGG marks the {context} between {source} and {target} as missing in this pathway."
        )
    subtype_key = ", ".join(normalized_subtypes) if normalized_subtypes else "None"
    templates = template_asset()["relation_templates"]
    template = templates.get(relation_class, {}).get(subtype_key)
    if template is None:
        return None
    mediator = joined_names(mediators) if mediators else None
    return _capitalize(
        str(template).format(entry1=source, entry2=target, compound=mediator)
    )


_AUDITED_TEMPLATE_OVERRIDES: dict[tuple[str, tuple[str, ...]], str] = {
    # The archived template reversed entry1/entry2 despite the direction-bearing
    # expression annotation.  v4 keeps KGML entry1 -> entry2 here.
    ("GErel", ("expression", "indirect")): (
        "{entry1} indirectly influences the expression of {entry2}."
    ),
    # Preserve both annotated actions without introducing an unsupported claim
    # specifically about gene expression.
    ("PPrel", ("activation", "dephosphorylation")): (
        "{entry1} dephosphorylates and activates {entry2}."
    ),
}


def _clean_template_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = value.replace(" by triggers its dissociation", " by triggering its dissociation")
    value = value.replace(" are not yet reported", " is not yet reported")
    return value


def _fallback_relation_text(
    *,
    relation_class: str,
    subtypes: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    mediators: Sequence[Mapping[str, Any]],
) -> str:
    source = joined_names(sources)
    target = joined_names(targets)
    annotations = ", ".join(subtypes) if subtypes else "no subtype annotation"
    mediator_clause = f" via {joined_names(mediators)}" if mediators else ""
    return (
        f"KEGG records a {relation_class} relation from {source} to {target}"
        f"{mediator_clause}, with annotations {annotations}."
    )


def audited_relation_text(
    *,
    relation_class: str,
    subtypes: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    mediators: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, str]:
    """Return ``(model_text, legacy_text, text_source)`` for one relation."""

    normalized = tuple(str(value).strip().casefold() for value in subtypes)
    legacy = legacy_relation_text(
        relation_class=relation_class,
        subtypes=normalized,
        sources=sources,
        targets=targets,
        mediators=mediators,
    )
    override = _AUDITED_TEMPLATE_OVERRIDES.get((relation_class, normalized))
    if override is not None:
        model_text = override.format(
            entry1=joined_names(sources),
            entry2=joined_names(targets),
            compound=joined_names(mediators) if mediators else None,
        )
        return _clean_template_text(_capitalize(model_text)), legacy, TEXT_SOURCE_LEGACY_CORRECTED
    if legacy is not None:
        return _clean_template_text(legacy), legacy, TEXT_SOURCE_LEGACY_CORRECTED
    return (
        _fallback_relation_text(
            relation_class=relation_class,
            subtypes=normalized,
            sources=sources,
            targets=targets,
            mediators=mediators,
        ),
        None,
        TEXT_SOURCE_CANONICAL_FALLBACK,
    )


def legacy_reaction_text(
    *,
    reversibility: str,
    sources: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> str:
    source = joined_names(sources)
    target = joined_names(targets)
    verb = "is" if len(sources) == 1 and len(targets) == 1 else "are"
    return _capitalize(
        f"{source} {verb} converted to {target} in a {reversibility} way."
    )


def audited_reaction_text(
    *,
    reversibility: str,
    sources: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    if reversibility not in {"reversible", "irreversible"}:
        raise ValueError(f"unsupported reaction reversibility={reversibility!r}")
    source = joined_names(sources)
    target = joined_names(targets)
    verb = "is" if len(sources) == 1 and len(targets) == 1 else "are"
    adverb = "reversibly" if reversibility == "reversible" else "irreversibly"
    legacy = legacy_reaction_text(
        reversibility=reversibility,
        sources=sources,
        targets=targets,
    )
    return (
        _capitalize(f"{source} {verb} {adverb} converted to {target}."),
        legacy,
        TEXT_SOURCE_LEGACY_CORRECTED,
    )


__all__ = [
    "TEMPLATE_ASSET",
    "TEXT_SOURCE_CANONICAL_FALLBACK",
    "TEXT_SOURCE_LEGACY_CORRECTED",
    "audited_reaction_text",
    "audited_relation_text",
    "joined_names",
    "legacy_reaction_text",
    "legacy_relation_text",
    "template_asset",
    "template_provenance",
]
