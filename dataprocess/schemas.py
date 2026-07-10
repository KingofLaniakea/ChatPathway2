"""Typed records for ChatPathway2 pathway trajectory CSV generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence


CSV_FIELDNAMES = [
    "question",
    "answer",
    "question_type",
    "given_step",
    "total_step",
    "pathway_id",
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
]


QUESTION_TYPE = "remaining_pathway_json"


@dataclass(frozen=True)
class PathwayStep:
    """One ordered text layer from a processed KEGG pathway block."""

    step_index: int
    layer_id: str
    text: str
    source_items: Sequence[str] = field(default_factory=tuple)

    def prompt_line(self) -> str:
        return f"Step {self.step_index}: {self.text}"

    def answer_object(self) -> dict[str, object]:
        return {
            "step": self.step_index,
            "layer": self.layer_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class PhenotypeTarget:
    """Phenotype supervision extracted from processed_graph, if present."""

    text: Optional[str] = None
    status: str = "missing"
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

    def question(self) -> str:
        lines = [
            "You are an expert in biological pathway reasoning.",
            "The source is a KEGG pathway trajectory converted from KGML into graph-grounded text.",
            "Task: continue the pathway trajectory from the observed prefix.",
            "",
            "Instructions:",
            "- Treat each Step as one ordered graph-layer transition from upstream to downstream.",
            "- A Step may summarize multiple reaction or relation events that occur in the same graph layer.",
            "- Predict only the remaining downstream Steps; do not repeat observed Steps.",
            '- Return valid JSON only, with keys "remaining_steps" and "predicted_phenotype".',
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
            "question": self.question(),
            "answer": self.answer_json(),
            "question_type": QUESTION_TYPE,
            "given_step": self.given_step,
            "total_step": self.record.total_step,
            "pathway_id": self.record.pathway_id,
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
        }
