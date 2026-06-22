"""Conditional candidate scoring for Tasks VII and VIII.

Scores are mean log probabilities of a candidate continuation conditioned on a
ChatPathway question/prefix. Higher is better. This is ranking, not generation:
it never assumes a candidate is biologically valid merely because the LLM gives
it a high score.
"""

from __future__ import annotations

from typing import Any


def chat_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def load_model(base_model: str, adapter: str | None, device_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device_name if device_name != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    return tokenizer, model, device


def conditional_score(tokenizer: Any, model: Any, device: Any, question: str, candidate: str, max_length: int) -> float:
    """Mean next-token log probability of candidate text given a question."""
    import torch

    prompt_ids = tokenizer.encode(chat_prompt(question), add_special_tokens=False)
    continuation_ids = tokenizer.encode(f"{candidate}<|im_end|>", add_special_tokens=False)
    full_ids = (prompt_ids + continuation_ids)[-max_length:]
    # If truncation removed a portion of the prompt, only retained continuation
    # tokens contribute to the score.
    retained_prompt = max(0, len(full_ids) - len(continuation_ids))
    target_start = retained_prompt
    if len(full_ids) < 2 or target_start >= len(full_ids):
        return float("-inf")
    input_ids = torch.tensor(full_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, :-1].float()
        log_probs = logits.log_softmax(dim=-1)
    target_ids = input_ids[0, 1:]
    token_positions = torch.arange(target_ids.numel(), device=device)
    target_mask = token_positions >= max(target_start - 1, 0)
    values = log_probs[token_positions[target_mask], target_ids[target_mask]]
    return float(values.mean().cpu()) if values.numel() else float("-inf")
