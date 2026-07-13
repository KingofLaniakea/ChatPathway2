"""Pure helpers for strict pathway-JSON generation retries."""

from __future__ import annotations

import json

from downstream.common.pathway_json import parse_pathway_payload


def generation_validity(
    text: str,
    *,
    expected_first_layer: int | None = None,
) -> tuple[bool, bool, str]:
    """Validate one bare, complete v3 JSON object.

    Downstream readers retain compatibility with fenced historical output, but
    maintained generation is stricter: Markdown fences, commentary, trailing
    text, and a continuation that restarts at the wrong layer all trigger a
    retry.
    """

    try:
        loaded = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return False, False, f"invalid_json:{exc.msg}"
    parsed = parse_pathway_payload(loaded, allow_text_fallback=False)
    schema_valid = parsed.schema_valid and bool(parsed.steps)
    if schema_valid and expected_first_layer is not None:
        actual_first_layer = parsed.steps[0].step
        if actual_first_layer != expected_first_layer:
            return (
                True,
                False,
                f"first_layer_index:{actual_first_layer};expected:{expected_first_layer}",
            )
    error = parsed.error or ("invalid_schema" if not schema_valid else "")
    return True, schema_valid, error


def repair_prompt(original_prompt: str, previous: str, error: str, attempt: int) -> str:
    del previous  # Failed output is intentionally omitted so it cannot crowd out the observed prefix.
    assistant_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if not original_prompt.endswith(assistant_suffix):
        raise ValueError("original prompt does not end with the maintained assistant marker")
    user_prompt = original_prompt[: -len(assistant_suffix)]
    return "".join(
        (
            user_prompt,
            "\nA previous generation failed strict JSON validation ",
            f"({error or 'invalid_schema'}). This is repair attempt {attempt} of 3. ",
            "Regenerate the answer from the observed prefix. Return exactly one complete JSON object ",
            "matching the required schema, without Markdown, commentary, or extra keys.",
            assistant_suffix,
        )
    )


def retry_token_budget(
    *,
    max_new_tokens: int,
    retry_max_new_tokens: int,
    max_json_attempts: int,
    attempt: int,
) -> int:
    if attempt >= max_json_attempts:
        return retry_max_new_tokens
    return min(retry_max_new_tokens, max_new_tokens * 2)


__all__ = ["generation_validity", "repair_prompt", "retry_token_budget"]
