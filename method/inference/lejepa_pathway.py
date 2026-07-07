"""Run a trained pathway LeJEPA probe on question/answer records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.training.lejepa_pathway import PathwayTextJEPA, embed_texts


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/checkpoints/pathway_lejepa_sentence/lejepa_epoch_3.pt"
DEFAULT_INPUT = "/root/autodl-tmp/data/test_7_species_dataset.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/runs/lejepa_pathway/test_7_species_lejepa_scores.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_backbone(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    if args.adapter and not args.no_adapter:
        backbone = PeftModel.from_pretrained(backbone, args.adapter)
    backbone.eval()
    return tokenizer, backbone


def records_from_csv(path: str, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    if limit is not None:
        df = df.head(limit)
    return df.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_ckpt = torch.load(args.checkpoint, map_location=args.device)
    hidden_size = int(raw_ckpt["hidden_size"])
    latent_dim = int(raw_ckpt["latent_dim"])
    probe = PathwayTextJEPA(hidden_size, latent_dim).to(args.device)
    probe.load_state_dict(raw_ckpt["model_state_dict"])
    probe.eval()

    tokenizer, backbone = load_backbone(args)
    records = records_from_csv(args.input, args.limit)

    with output_path.open("w", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(records), args.batch_size), desc="LeJEPA inference"):
            batch = records[start : start + args.batch_size]
            questions = [str(record.get("question", "")) for record in batch]
            answers = [
                str(record.get("answer", record.get("formatted_answer_no_phenotype", "")))
                for record in batch
            ]
            prompt_texts = [f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" for text in questions]
            answer_texts = [f"{text}<|im_end|>" for text in answers]
            with torch.no_grad():
                prompt_emb = embed_texts(prompt_texts, tokenizer, backbone, args)
                answer_emb = embed_texts(answer_texts, tokenizer, backbone, args)
                pred, target, context = probe(prompt_emb, answer_emb)
                cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
                l2 = torch.linalg.vector_norm(pred - target, dim=-1)
                context_norm = torch.linalg.vector_norm(context, dim=-1)

            for offset, record in enumerate(batch):
                handle.write(json.dumps({
                    "sample_id": start + offset,
                    "pathway_id": record.get("pathway_id"),
                    "entry_id": record.get("entry_id"),
                    "lejepa_cosine": float(cosine[offset].item()),
                    "lejepa_l2": float(l2[offset].item()),
                    "context_norm": float(context_norm[offset].item()),
                }, ensure_ascii=False) + "\n")

    metadata_path = output_path.with_suffix(".run.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
