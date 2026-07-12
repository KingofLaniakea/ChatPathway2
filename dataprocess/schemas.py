"""Typed records for ChatPathway2 pathway trajectory CSV generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

try:
    from dataprocess.substeps import parse_substeps
except ImportError:  # Allows direct script execution from dataprocess/.
    from substeps import parse_substeps  # type: ignore


CSV_FIELDNAMES = [
    "sample_id",
    "record_id",
    "question",
    "answer",
    "question_type",
    "given_step",
    "total_step",
    "pathway_id",
    "pathway_family_id",
    "entry_id",
    "phenotype",
    "phenotype_status",
    "phenotype_source",
    "organism",
    "pathway_block",
    "pathway_title",
    "source_json",
    "source_graph_json",
    "prefix_step_count",
    "target_step_count",
    "has_empty_prefix",
    "substep_schema_version",
    "substep_source",
]


QUESTION_TYPE = "remaining_pathway_json"
PATHWAY_FAMILY_RE = re.compile(r"(\d{5})$")


def canonical_pathway_family_id(pathway_id: object) -> str:
    """Return the cross-organism KEGG pathway family identifier.

    Organism-specific KEGG IDs such as ``hsa04010`` and ``mmu04010`` share
    the same five-digit reference-map suffix.  Keeping that suffix explicit
    lets split/audit code prevent the same KEGG pathway family from crossing
    train and a strict held-out evaluation.  Non-standard IDs remain visible
    under a ``raw:`` namespace instead of being silently discarded.
    """

    normalized = str(pathway_id or "").strip()
    match = PATHWAY_FAMILY_RE.search(normalized)
    if match:
        return match.group(1)
    return f"raw:{normalized.casefold()}" if normalized else "raw:<missing>"


@dataclass(frozen=True)
class PathwayStep:
    """One ordered text layer from a processed KEGG pathway block."""

    step_index: int
    layer_id: str
    text: str
    source_items: Sequence[str] = field(default_factory=tuple)

    def prompt_line(self) -> str:
        events = parse_substeps(self.text, source_items=self.source_items)
        rendered = "\n".join(f"  - Event {event.index}: {event.text}" for event in events)
        return f"Step {self.step_index} ({self.layer_id}; same-depth events):\n{rendered}"

    def answer_object(self) -> dict[str, object]:
        return {
            "step": self.step_index,
            "layer": self.layer_id,
            "substeps": [
                event.as_dict()
                for event in parse_substeps(self.text, source_items=self.source_items)
            ],
        }


@dataclass(frozen=True)
class PhenotypeTarget:
    """Explicit block-level phenotype supervision, if present."""

    text: Optional[str] = None
    status: str = "not_annotated"
    source: str = ""

    @property
    def has_target(self) -> bool:
        return bool(self.text and self.text.strip())

    def answer_value(self) -> Optional[dict[str, str]]:
        if not self.has_target:
            return None
        return {
            "text": self.text.strip(),
            "status": self.status,
        }


@dataclass(frozen=True)
class PathwayRecord:
    """A single sink-rooted pathway block from one processed JSON file."""

    source_json: str
    source_graph_json: str
    organism: str
    pathway_id: str
    entry_id: str
    pathway_block: str
    pathway_title: str
    steps: Sequence[PathwayStep]
    phenotype: PhenotypeTarget

    @property
    def pathway_family_id(self) -> str:
        return canonical_pathway_family_id(self.pathway_id)

    @property
    def record_id(self) -> str:
        identity = "\n".join(
            (self.organism, self.source_json, self.pathway_id, self.pathway_block)
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    @property
    def total_step(self) -> int:
        return max(len(self.steps) - 1, 0)


@dataclass(frozen=True)
class PathwayExample:
    """A supervised prefix-to-remaining-trajectory training example."""

    record: PathwayRecord
    prefix_len: int

    @property
    def observed_steps(self) -> Sequence[PathwayStep]:
        return self.record.steps[: self.prefix_len]

    @property
    def remaining_steps(self) -> Sequence[PathwayStep]:
        return self.record.steps[self.prefix_len :]

    @property
    def given_step(self) -> int:
        if self.prefix_len == 0:
            return -1
        return self.observed_steps[-1].step_index

    @property
    def sample_id(self) -> str:
        return f"{self.record.record_id}:prefix={self.prefix_len}"

    def question(self) -> str:
        lines = [
            "You are an expert in biological pathway reasoning.",
            "The source is a KEGG pathway trajectory converted from KGML into graph-grounded text.",
            "Task: continue the pathway trajectory from the observed prefix.",
            "",
            "Instructions:",
            "- Treat each Step as one ordered graph-layer transition from upstream to downstream.",
            "- Each Step contains one or more substeps/events at the same graph depth; do not invent an order among same-depth events.",
            "- Predict only the remaining downstream Steps; do not repeat observed Steps.",
            '- Return valid JSON only, with keys "remaining_steps" and "predicted_phenotype"; each remaining Step must contain "step", "layer", and a "substeps" list.',
            '- Use null for "predicted_phenotype" when the source graph has no phenotype annotation.',
            "",
            f"Organism: {self.record.organism or 'unknown'}",
            f"KEGG pathway ID: {self.record.pathway_id or 'unknown'}",
            f"Pathway block: {self.record.pathway_block}",
            f"Pathway title: {self.record.pathway_title or 'unknown'}",
            "",
            "Observed pathway prefix:",
        ]
        if self.observed_steps:
            lines.extend(step.prompt_line() for step in self.observed_steps)
        else:
            lines.append("No pathway steps are given.")
        return "\n".join(lines)

    def answer_payload(self) -> dict[str, object]:
        return {
            "remaining_steps": [
                step.answer_object() for step in self.remaining_steps
            ],
            "predicted_phenotype": self.record.phenotype.answer_value(),
        }

    def answer_json(self) -> str:
        return json.dumps(
            self.answer_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def csv_row(self) -> dict[str, object]:
        phenotype_text = self.record.phenotype.text or ""
        return {
            "sample_id": self.sample_id,
            "record_id": self.record.record_id,
            "question": self.question(),
            "answer": self.answer_json(),
            "question_type": QUESTION_TYPE,
            "given_step": self.given_step,
            "total_step": self.record.total_step,
            "pathway_id": self.record.pathway_id,
            "pathway_family_id": self.record.pathway_family_id,
            "entry_id": self.record.entry_id,
            "phenotype": phenotype_text,
            "phenotype_status": self.record.phenotype.status,
            "phenotype_source": self.record.phenotype.source,
            "organism": self.record.organism,
            "pathway_block": self.record.pathway_block,
            "pathway_title": self.record.pathway_title,
            "source_json": self.record.source_json,
            "source_graph_json": self.record.source_graph_json,
            "prefix_step_count": self.prefix_len,
            "target_step_count": len(self.remaining_steps),
            "has_empty_prefix": int(self.prefix_len == 0),
            "substep_schema_version": "layer_set_v1",
            "substep_source": "processed_source_items",
        }
