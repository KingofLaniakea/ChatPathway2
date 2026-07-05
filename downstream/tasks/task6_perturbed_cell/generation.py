#!/usr/bin/env python3
"""Generate Task VI Cell2Sentence prediction artifacts.

This moves the server C2S and Gemma comparison generation paths under the
Task VI downstream package. It writes the JSONL contract consumed by
``downstream.tasks.task6_perturbed_cell``:

``instruction``, ``ground_truth``, ``prediction``, and ``control_base``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


QWEN_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
QWEN_C2S_ADAPTER = "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent/checkpoint_epoch_5"
GEMMA_BASE_MODEL = "/root/autodl-tmp/models/C2S-Scale-Gemma-2-2B"
TRAIN_JSONL = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"
TEST_JSONL = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"
QWEN_OUTPUT = "/root/autodl-tmp/runs/c2s/jurkat_ours_results_epoch5.jsonl"
GEMMA_OUTPUT = "/root/autodl-tmp/runs/c2s/jurkat_test_gemma_predictions_result_5percent_500.jsonl"


@dataclass
class GenerationConfig:
    model: str
    base_model: str
    adapter: str | None
    train_jsonl: str
    test_jsonl: str
    output: str
    limit: int | None
    max_new_tokens: int
    device: str
    overwrite: bool
    cuda_visible_devices: str | None


def control_sentence(instruction: str) -> str:
    marker = "Control cell sentence: "
    if marker not in instruction:
        return ""
    return instruction.split(marker, 1)[1].split(".\n\nPerturbed", 1)[0]


def prompt_for(model_name: str, instruction: str) -> str:
    if model_name == "qwen_c2s":
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    return instruction


def read_jsonl(path: str, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def default_config(model_name: str) -> dict[str, Any]:
    if model_name == "qwen_c2s":
        return {
            "base_model": QWEN_BASE_MODEL,
            "adapter": QWEN_C2S_ADAPTER,
            "output": QWEN_OUTPUT,
            "limit": 100,
        }
    return {
        "base_model": GEMMA_BASE_MODEL,
        "adapter": None,
        "output": GEMMA_OUTPUT,
        "limit": 500,
    }


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=("qwen_c2s", "gemma"), required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--adapter", help="LoRA adapter; only used for qwen_c2s unless explicitly set.")
    parser.add_argument("--train-jsonl", default=TRAIN_JSONL, help="Recorded for provenance and shared-vocabulary scoring.")
    parser.add_argument("--test-jsonl", default=TEST_JSONL)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Number of test rows to generate. Defaults preserve legacy scripts.")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override for server scheduling.")
    args = parser.parse_args()

    defaults = default_config(args.model)
    adapter = args.adapter if args.adapter is not None else defaults["adapter"]
    if args.model == "gemma" and args.adapter is None:
        adapter = None
    return GenerationConfig(
        model=args.model,
        base_model=args.base_model or defaults["base_model"],
        adapter=adapter,
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        output=args.output or defaults["output"],
        limit=args.limit if args.limit is not None else defaults["limit"],
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        overwrite=args.overwrite,
        cuda_visible_devices=args.cuda_visible_devices,
    )


def generate(cfg: GenerationConfig) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    device = cfg.device if cfg.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    output_path = Path(cfg.output)
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    if cfg.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, cfg.adapter)
    else:
        model = base_model
    model.eval()

    records = read_jsonl(cfg.test_jsonl, cfg.limit)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc=f"{cfg.model} generation"):
            instruction = str(record["instruction"])
            prompt = prompt_for(cfg.model, instruction)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][input_ids.shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            handle.write(json.dumps({
                "model_label": cfg.model,
                "instruction": instruction,
                "ground_truth": record["output"],
                "prediction": prediction,
                "control_base": control_sentence(instruction),
            }, ensure_ascii=False) + "\n")

    metadata_path = output_path.with_suffix(".run.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote run metadata: {metadata_path}")


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
