"""Shared prompt/answer token budgeting and substep-span alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dataprocess.substeps import split_substeps


class IncompleteSupervisionError(ValueError):
    """Raised when a complete assistant JSON target cannot fit the token budget."""

    def __init__(self, *, prompt_tokens: int, answer_tokens: int, max_length: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.answer_tokens = answer_tokens
        self.max_length = max_length
        super().__init__(
            "complete prompt+assistant JSON does not fit the token budget: "
            f"prompt={prompt_tokens}, closed_answer={answer_tokens}, max_length={max_length}. "
            "Drop or rematerialize this sample; assistant JSON must never be token-truncated."
        )


@dataclass(frozen=True)
class EncodedSupervision:
    input_ids: list[int]
    labels: list[int]
    step_span_groups: tuple[tuple[tuple[int, int], ...], ...]
    prompt_tokens_dropped: int
    answer_tokens_dropped: int
    substeps_total: int
    substeps_retained: int
    semantic_steps_total: int
    semantic_steps_retained: int


def trim_prompt_ids(prompt_ids: list[int], budget: int, *, head_tokens: int = 192) -> list[int]:
    """Keep task/schema instructions plus the most recent observed pathway."""

    if budget <= 0:
        return []
    if len(prompt_ids) <= budget:
        return prompt_ids
    head = min(head_tokens, max(1, budget // 3))
    return prompt_ids[:head] + prompt_ids[-(budget - head) :]


def pathway_step_substep_texts(answer_json: str) -> tuple[tuple[str, ...], ...]:
    try:
        payload = json.loads(answer_json)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, dict):
        return ()
    if isinstance(payload.get("remaining_layers"), list):
        groups: list[tuple[str, ...]] = []
        for layer in payload["remaining_layers"]:
            if not isinstance(layer, dict) or not isinstance(layer.get("events"), list):
                continue
            values = tuple(
                str(event.get("text", "")).strip()
                for event in layer["events"]
                if isinstance(event, dict) and str(event.get("text", "")).strip()
            )
            if values:
                groups.append(values)
        return tuple(groups)
    if not isinstance(payload.get("remaining_steps"), list):
        return ()
    groups: list[tuple[str, ...]] = []
    for step in payload["remaining_steps"]:
        if not isinstance(step, dict):
            continue
        explicit = step.get("substeps")
        if isinstance(explicit, list):
            values: list[str] = []
            for substep in explicit:
                if isinstance(substep, dict) and str(substep.get("text", "")).strip():
                    values.append(str(substep["text"]).strip())
            if values:
                groups.append(tuple(values))
            continue
        text = str(step.get("text", "")).strip()
        if text:
            values = split_substeps(text)
            if values:
                groups.append(values)
    return tuple(groups)


def _find_substep_char_spans(
    answer_json: str,
    groups: tuple[tuple[str, ...], ...],
) -> list[list[tuple[int, int]]]:
    span_groups: list[list[tuple[int, int]]] = []
    cursor = 0
    for texts in groups:
        spans: list[tuple[int, int]] = []
        for text in texts:
            escaped = json.dumps(text, ensure_ascii=False)[1:-1]
            start = answer_json.find(escaped, cursor)
            if start < 0:
                continue
            end = start + len(escaped)
            spans.append((start, end))
            cursor = end
        span_groups.append(spans)
    return span_groups


def _answer_encoding(tokenizer: Any, answer_text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return list(encoded["input_ids"]), [tuple(pair) for pair in encoded["offset_mapping"]]
    except (TypeError, NotImplementedError, KeyError):
        return list(tokenizer.encode(answer_text, add_special_tokens=False)), None


def _token_spans(
    tokenizer: Any,
    answer_text: str,
    answer_ids: list[int],
    char_spans: list[tuple[int, int]],
    offsets: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for char_start, char_end in char_spans:
        if offsets is not None:
            indices = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > char_start and token_start < char_end
            ]
            if indices:
                spans.append((indices[0], indices[-1] + 1))
            continue
        # Slow-tokenizer fallback. Prefix token counts are deterministic for
        # the same serialized answer and avoid guessing token substrings.
        start = len(tokenizer.encode(answer_text[:char_start], add_special_tokens=False))
        end = len(tokenizer.encode(answer_text[:char_end], add_special_tokens=False))
        if 0 <= start < end <= len(answer_ids):
            spans.append((start, end))
    return spans


def encode_supervised(
    tokenizer: Any,
    prompt: str,
    answer_json: str,
    *,
    max_length: int,
    answer_budget_fraction: float = 0.5,
    truncation_policy: str = "error",
) -> EncodedSupervision:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not 0 < answer_budget_fraction < 1:
        raise ValueError("answer_budget_fraction must be between 0 and 1")
    if truncation_policy not in {"error", "measure"}:
        raise ValueError("truncation_policy must be 'error' or 'measure'")
    try:
        payload = json.loads(answer_json)
    except json.JSONDecodeError as exc:
        raise ValueError("assistant supervision must be one complete JSON value") from exc
    if not isinstance(payload, (dict, list)):
        raise ValueError("assistant supervision JSON must be an object or list")

    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    answer_text = f"{answer_json}<|im_end|>"
    answer_ids, offsets = _answer_encoding(tokenizer, answer_text)
    if len(prompt_ids) + len(answer_ids) <= max_length:
        kept_prompt = prompt_ids
        kept_answer = answer_ids
    elif truncation_policy == "error":
        raise IncompleteSupervisionError(
            prompt_tokens=len(prompt_ids),
            answer_tokens=len(answer_ids),
            max_length=max_length,
        )
    else:
        # Retained only for retrospective coverage measurement. Trainers use
        # the default fail-closed policy above.
        minimum_answer_budget = max(1, round(max_length * answer_budget_fraction))
        answer_keep = min(len(answer_ids), minimum_answer_budget)
        prompt_budget = max_length - answer_keep
        kept_prompt = trim_prompt_ids(prompt_ids, prompt_budget)
        remaining_budget = max_length - len(kept_prompt)
        kept_answer = answer_ids[:remaining_budget]

    text_groups = pathway_step_substep_texts(answer_json)
    char_groups = _find_substep_char_spans(answer_json, text_groups)
    answer_groups = [
        _token_spans(tokenizer, answer_text, answer_ids, char_group, offsets)
        for char_group in char_groups
    ]
    full_groups = tuple(
        tuple(
            (len(kept_prompt) + start, len(kept_prompt) + end)
            for start, end in answer_group
        )
        for answer_group, text_group in zip(answer_groups, text_groups)
        if len(answer_group) == len(text_group)
        and answer_group
        and all(end <= len(kept_answer) for _, end in answer_group)
    )
    return EncodedSupervision(
        input_ids=kept_prompt + kept_answer,
        labels=[-100] * len(kept_prompt) + kept_answer,
        step_span_groups=full_groups,
        prompt_tokens_dropped=len(prompt_ids) - len(kept_prompt),
        answer_tokens_dropped=len(answer_ids) - len(kept_answer),
        substeps_total=sum(len(group) for group in text_groups),
        substeps_retained=sum(len(group) for group in full_groups),
        semantic_steps_total=len(text_groups),
        semantic_steps_retained=len(full_groups),
    )


__all__ = [
    "EncodedSupervision",
    "IncompleteSupervisionError",
    "encode_supervised",
    "pathway_step_substep_texts",
    "trim_prompt_ids",
]
