"""Generate pathway answers with a latent-rollout residual injected into prompt embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from method.dynamics.latent_teacher import build_dynamics, load_backbone, load_projection, prompt_for, rollout


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4"
DEFAULT_AE = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/checkpoints/latent_dynamics_teachers/neural_ode/neural_ode_epoch_3.pt"
DEFAULT_INPUT = "/root/autodl-tmp/data/test_7_species_dataset.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/runs/latent_dynamics_injection/neural_ode_residual_generation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--ae-ckpt", default=DEFAULT_AE)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--batch-size", type=int, default=1, help="Prototype currently injects one prompt at a time.")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def injected_generate(record: dict, tokenizer, model, projection, dynamics, variant: str, args: argparse.Namespace) -> dict:
    question = str(record.get(args.question_column, ""))
    prompt = prompt_for(question)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].float()
        z_all, _ = projection(hidden)
        control = z_all[0].mean(dim=0)
        z0 = z_all[0, -1]
        z_rollout = rollout(dynamics, variant, z0, control, 1)
        delta_latent = z_rollout[1] - z_rollout[0]
        residual_hidden = projection.up(delta_latent.unsqueeze(0)).squeeze(0)

        embeddings = model.get_input_embeddings()(encoded["input_ids"])
        residual = args.residual_scale * residual_hidden.to(device=embeddings.device, dtype=embeddings.dtype)
        embeddings = embeddings.clone()
        embeddings[:, -1, :] = embeddings[:, -1, :] + residual

        generated = model.generate(
            inputs_embeds=embeddings,
            attention_mask=encoded["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=False)
    if prompt in text:
        text = text.split(prompt, 1)[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return {
        **record,
        "predicted_answer": text.strip(),
        "injection_variant": variant,
        "residual_scale": args.residual_scale,
        "residual_norm": float(torch.linalg.vector_norm(residual_hidden).item()),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    if args.limit is not None:
        df = df.head(args.limit)
    if args.question_column not in df.columns:
        raise ValueError(f"Input must contain question column '{args.question_column}'.")

    raw = torch.load(args.checkpoint, map_location=args.device)
    variant = raw["variant"]
    cfg = raw.get("config", {})
    latent_dim = int(cfg.get("latent_dim", args.latent_dim))
    tokenizer, model = load_backbone(args.base_model, None if args.no_adapter else args.adapter, args.device)
    projection = load_projection(args.ae_ckpt, model.config.hidden_size, latent_dim, args.device)
    dynamics = build_dynamics(variant, latent_dim, latent_dim).to(args.device).float()
    dynamics.load_state_dict(raw["model_state_dict"])
    dynamics.eval()

    rows = []
    for _, record in tqdm(list(df.iterrows()), desc=f"{variant} residual injection"):
        rows.append(injected_generate(record.to_dict(), tokenizer, model, projection, dynamics, variant, args))

    pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    with output_path.with_suffix(".run.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote injected-generation CSV: {output_path}")


if __name__ == "__main__":
    main()
