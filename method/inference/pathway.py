import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from method.inference.csv_io import read_csv_text_rows, select_strided_shard
from method.inference.json_retry import generation_validity, repair_prompt, retry_token_budget
from method.training.common import file_sha256, git_commit, seed_everything
from method.training.sequence import trim_prompt_ids

DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4"
DEFAULT_INPUT = "/root/autodl-tmp/data/test_kegg_pathway_eval.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv"


@dataclass
class InferenceConfig:
    base_model_id: str
    trained_lora_path: str
    test_data_path: str
    output_data_path: str
    progress_data_path: str | None
    batch_size: int
    max_length: int
    max_new_tokens: int
    max_json_attempts: int
    retry_max_new_tokens: int
    limit: int | None
    shard_count: int
    shard_index: int
    seed: int
    device: str
    overwrite: bool
    completion_marker: str | None


@dataclass(frozen=True)
class GenerationAttempt:
    text: str
    generated_token_count: int
    finish_reason: str
    max_new_tokens: int


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(
        description="Run deterministic ChatPathway LoRA generation on a CSV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV; must contain a question column.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Prediction CSV to create.")
    parser.add_argument(
        "--progress-output",
        dest="progress_data_path",
        help="Append one auditable JSON record per completed sample; defaults beside --output.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-json-attempts",
        type=int,
        default=3,
        help="Generate, repair, and fail explicitly if strict JSON is still invalid.",
    )
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=8192,
        help="Maximum output budget used by the final JSON repair attempt.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the post-limit input into this many deterministic strided shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to evaluate; pair with unique output/progress paths.",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output CSV.")
    parser.add_argument(
        "--require-complete",
        dest="completion_marker",
        help="Require a trainer run_complete.json marker with status=completed before inference.",
    )
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.max_json_attempts != 3:
        parser.error("--max-json-attempts must be 3 for the maintained inference contract")
    if args.retry_max_new_tokens < args.max_new_tokens:
        parser.error("--retry-max-new-tokens must be at least --max-new-tokens")
    return InferenceConfig(
        base_model_id=args.base_model,
        trained_lora_path=args.adapter,
        test_data_path=args.input,
        output_data_path=args.output,
        progress_data_path=args.progress_data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_json_attempts=args.max_json_attempts,
        retry_max_new_tokens=args.retry_max_new_tokens,
        limit=args.limit,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        completion_marker=args.completion_marker,
    )


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    prompt_budget: int,
    max_new_tokens: int,
    device: str,
    stop_ids: set[int],
) -> list[GenerationAttempt]:
    batch_ids = [
        trim_prompt_ids(
            list(tokenizer.encode(prompt, add_special_tokens=False)),
            prompt_budget,
        )
        for prompt in prompts
    ]
    inputs = tokenizer.pad(
        {
            "input_ids": batch_ids,
            "attention_mask": [[1] * len(ids) for ids in batch_ids],
        },
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted(stop_ids),
        )

    results: list[GenerationAttempt] = []
    padded_input_length = int(inputs["input_ids"].shape[1])
    for out_ids in outputs:
        raw_gen_ids = out_ids[padded_input_length:]
        raw_values = raw_gen_ids.tolist()
        first_stop = next(
            (
                index
                for index, token in enumerate(raw_values)
                if int(token) in stop_ids
            ),
            None,
        )
        actual_gen_ids = (
            raw_gen_ids[: first_stop + 1]
            if first_stop is not None
            else raw_gen_ids
        )
        generated_count = int(actual_gen_ids.numel())
        finish_reason = (
            "eos"
            if first_stop is not None
            else ("max_new_tokens" if generated_count >= max_new_tokens else "stopped")
        )
        text = tokenizer.decode(actual_gen_ids, skip_special_tokens=False)
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>", 1)[0]
        results.append(
            GenerationAttempt(
                text=text.strip(),
                generated_token_count=generated_count,
                finish_reason=finish_reason,
                max_new_tokens=max_new_tokens,
            )
        )
    return results


def run_inference(cfg: InferenceConfig) -> None:
    seed_everything(cfg.seed)
    if cfg.completion_marker:
        marker_path = Path(cfg.completion_marker)
        if not marker_path.is_file():
            raise FileNotFoundError(f"Required training completion marker is missing: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "completed":
            raise ValueError(f"Training completion marker is not completed: {marker_path}")
    output_path = Path(cfg.output_data_path)
    progress_path = (
        Path(cfg.progress_data_path)
        if cfg.progress_data_path
        else output_path.with_suffix(".progress.jsonl")
    )
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}. "
            "Pass --overwrite only when replacement is intentional."
        )
    if progress_path.exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"Refusing to append to existing progress output: {progress_path}. "
                "Pass --overwrite only when replacement is intentional."
            )
        progress_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.touch()
    print(f"Using device: {cfg.device}")
    
    # 1. 初始化 Tokenizer
    print(f"Loading tokenizer from {cfg.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    
    # 2. 加载基础模型与训练好的LoRA权重 (不需要加载HNN)
    print(f"Loading base model from {cfg.base_model_id}...")
    dtype = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, 
        torch_dtype=dtype,
        device_map={"": cfg.device}, 
        trust_remote_code=True
    )
    
    print(f"Loading trained adapter from {cfg.trained_lora_path}...")
    model = PeftModel.from_pretrained(base_model, cfg.trained_lora_path)
    model.eval()  # 切换至评估模式，关闭 Dropout 
    
    # 3. 读取测试集
    print(f"Reading test dataset from {cfg.test_data_path}...")
    fieldnames, all_rows = read_csv_text_rows(cfg.test_data_path, limit=cfg.limit)
    indexed_rows = select_strided_shard(
        all_rows,
        shard_index=cfg.shard_index,
        shard_count=cfg.shard_count,
    )
    source_indices = [index for index, _ in indexed_rows]
    rows = [row for _, row in indexed_rows]
    df = pd.DataFrame(rows, columns=fieldnames)
    df.insert(0, "dataset_index", source_indices)
    
    # 打印检查，确保读取进来时列结构是完好的
    print(f"Dataset columns detected: {list(df.columns)}")
    print(
        f"Total samples to process: {len(df)} "
        f"(shard {cfg.shard_index + 1}/{cfg.shard_count}; full post-limit input={len(all_rows)})"
    )
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    # 4. 构建标准的 Prompt 格式 (必须与训练时 CSVPathwayDataset 里的格式严格一致)
    prompts = []
    for _, row in df.iterrows():
        prompt_text = f"<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt_text)
        
    predicted_answers = []
    generated_token_counts = []
    total_generated_token_counts = []
    finish_reasons = []
    generation_attempt_counts = []
    json_validity = []
    schema_validity = []

    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end_ids) != 1:
        raise ValueError(
            "The base tokenizer must encode <|im_end|> as exactly one token; "
            f"got {im_end_ids!r}"
        )
    im_end_id = int(im_end_ids[0])
    raw_eos_ids = tokenizer.eos_token_id
    if raw_eos_ids is None:
        eos_ids: set[int] = set()
    elif isinstance(raw_eos_ids, (list, tuple, set)):
        eos_ids = {int(value) for value in raw_eos_ids}
    else:
        eos_ids = {int(raw_eos_ids)}
    stop_ids = eos_ids | {im_end_id}
    
    # 5. 分批次进行自回归推理
    print("Starting batch inference...")
    for i in tqdm(range(0, len(prompts), cfg.batch_size), desc="Inferencing"):
        batch_prompts = prompts[i : i + cfg.batch_size]
        initial_results = generate_batch(
            model,
            tokenizer,
            batch_prompts,
            prompt_budget=cfg.max_length,
            max_new_tokens=cfg.max_new_tokens,
            device=cfg.device,
            stop_ids=stop_ids,
        )

        for j, initial in enumerate(initial_results):
            local_sample_index = i + j
            sample_index = source_indices[local_sample_index]
            source_row = rows[local_sample_index]
            try:
                expected_first_layer = int(source_row.get("prefix_step_count", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Input CSV must contain an integer prefix_step_count for strict v3 inference"
                ) from exc
            attempt_history = [initial]
            current = initial
            json_valid, schema_valid, validation_error = generation_validity(
                current.text,
                expected_first_layer=expected_first_layer,
            )
            for attempt in range(2, cfg.max_json_attempts + 1):
                if schema_valid:
                    break
                repaired_prompt = repair_prompt(
                    batch_prompts[j],
                    current.text,
                    validation_error,
                    attempt,
                )
                current = generate_batch(
                    model,
                    tokenizer,
                    [repaired_prompt],
                    prompt_budget=cfg.max_length,
                    max_new_tokens=retry_token_budget(
                        max_new_tokens=cfg.max_new_tokens,
                        retry_max_new_tokens=cfg.retry_max_new_tokens,
                        max_json_attempts=cfg.max_json_attempts,
                        attempt=attempt,
                    ),
                    device=cfg.device,
                    stop_ids=stop_ids,
                )[0]
                attempt_history.append(current)
                json_valid, schema_valid, validation_error = generation_validity(
                    current.text,
                    expected_first_layer=expected_first_layer,
                )

            progress_record = {
                "sample_index": sample_index,
                "sample_id": source_row.get("sample_id", ""),
                "record_id": source_row.get("record_id", ""),
                "organism": source_row.get("organism", ""),
                "pathway_family_id": source_row.get("pathway_family_id", ""),
                "gold_answer": source_row.get("answer", ""),
                "predicted_answer": current.text,
                "generation_attempts": len(attempt_history),
                "generation_attempt_history": [asdict(value) for value in attempt_history],
                "generated_token_count": current.generated_token_count,
                "total_generated_token_count": sum(
                    value.generated_token_count for value in attempt_history
                ),
                "finish_reason": current.finish_reason,
                "prediction_json_valid": json_valid,
                "prediction_schema_valid": schema_valid,
                "validation_error": validation_error,
                "status": "completed" if schema_valid else "failed_after_three_attempts",
            }
            with progress_path.open("a", encoding="utf-8") as progress_handle:
                progress_handle.write(
                    json.dumps(progress_record, ensure_ascii=False) + "\n"
                )
            if not schema_valid:
                raise RuntimeError(
                    "strict pathway JSON generation failed after three attempts: "
                    f"sample_id={source_row.get('sample_id', '')!r}, error={validation_error}"
                )

            json_validity.append(json_valid)
            schema_validity.append(schema_valid)
            predicted_answers.append(current.text)
            generated_token_counts.append(current.generated_token_count)
            total_generated_token_counts.append(
                sum(value.generated_token_count for value in attempt_history)
            )
            finish_reasons.append(current.finish_reason)
            generation_attempt_counts.append(len(attempt_history))
            
    # 6. 将预测结果追加回 Dataframe，并维持原始列格式输出
    df["predicted_answer"] = predicted_answers
    df["generated_token_count"] = generated_token_counts
    df["total_generated_token_count"] = total_generated_token_counts
    df["finish_reason"] = finish_reasons
    df["generation_attempts"] = generation_attempt_counts
    df["prediction_json_valid"] = json_validity
    df["prediction_schema_valid"] = schema_validity
    
    # 保存至新文件
    # df_final.to_csv(cfg.output_data_path, index=False, quoting=csv.QUOTE_NONE)
    # Preserve every identity, provenance, phenotype-status, and source column.
    df.to_csv(
        output_path,
        index=False, 
        quoting=csv.QUOTE_MINIMAL, 
        escapechar='\\'  # 如果文本内部包含双引号，用反斜杠转义，绝不错位
    )
    metadata_path = output_path.with_suffix(".run.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "git_commit": git_commit(Path(__file__).resolve().parents[2]),
                "input_sha256": file_sha256(cfg.test_data_path),
                "completion_marker_sha256": (
                    file_sha256(cfg.completion_marker) if cfg.completion_marker else None
                ),
                "progress_output": str(progress_path),
                "progress_output_sha256": file_sha256(progress_path),
                "input_rows": len(all_rows),
                "evaluated_rows": len(df),
                "finish_reason_counts": {
                    str(key): int(value)
                    for key, value in df["finish_reason"].value_counts().items()
                },
                "generation_attempt_counts": {
                    str(key): int(value)
                    for key, value in df["generation_attempts"].value_counts().items()
                },
                "total_generated_tokens_including_repairs": int(
                    df["total_generated_token_count"].sum()
                ),
                "prediction_json_valid_count": int(df["prediction_json_valid"].sum()),
                "prediction_schema_valid_count": int(df["prediction_schema_valid"].sum()),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print(f"\nInference completed successfully! Results saved to: {output_path}")
    print(f"Run configuration saved to: {metadata_path}")

if __name__ == "__main__":
    run_inference(parse_args())
