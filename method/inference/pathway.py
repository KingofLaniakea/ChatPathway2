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

DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4"
DEFAULT_INPUT = "/root/autodl-tmp/data/test_7_species_dataset.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv"


@dataclass
class InferenceConfig:
    base_model_id: str
    trained_lora_path: str
    test_data_path: str
    output_data_path: str
    batch_size: int
    max_length: int
    max_new_tokens: int
    device: str
    overwrite: bool


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(
        description="Run deterministic ChatPathway LoRA generation on a CSV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV; must contain a question column.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Prediction CSV to create.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output CSV.")
    args = parser.parse_args()
    return InferenceConfig(
        base_model_id=args.base_model,
        trained_lora_path=args.adapter,
        test_data_path=args.input,
        output_data_path=args.output,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        overwrite=args.overwrite,
    )


def run_inference(cfg: InferenceConfig) -> None:
    output_path = Path(cfg.output_data_path)
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}. "
            "Pass --overwrite only when replacement is intentional."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {cfg.device}")
    
    # 1. 初始化 Tokenizer
    print(f"Loading tokenizer from {cfg.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    
    # 2. 加载基础模型与训练好的LoRA权重 (不需要加载HNN)
    print(f"Loading base model from {cfg.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map={"": cfg.device}, 
        trust_remote_code=True
    )
    
    print(f"Loading trained adapter from {cfg.trained_lora_path}...")
    model = PeftModel.from_pretrained(base_model, cfg.trained_lora_path)
    model.eval()  # 切换至评估模式，关闭 Dropout 
    
    # 3. 读取测试集
    print(f"Reading test dataset from {cfg.test_data_path}...")
    # 使用 quoting=csv.QUOTE_NONE 防范特殊生物信息字符截断
    # df = pd.read_csv(cfg.test_data_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    
    df = pd.read_csv(
        cfg.test_data_path, 
        engine='python',            # 使用强大的 python 解析引擎
        quoting=csv.QUOTE_MINIMAL,  # 允许用双引号包裹含有换行、逗号的单元格
        on_bad_lines='skip'
    )
    
    # 打印检查，确保读取进来时列结构是完好的
    print(f"Dataset columns detected: {list(df.columns)}")
    print(f"Total samples to process: {len(df)}")
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    # 4. 构建标准的 Prompt 格式 (必须与训练时 CSVPathwayDataset 里的格式严格一致)
    prompts = []
    for _, row in df.iterrows():
        prompt_text = f"<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt_text)
        
    predicted_answers = []
    
    # 5. 分批次进行自回归推理
    print("Starting batch inference...")
    for i in tqdm(range(0, len(prompts), cfg.batch_size), desc="Inferencing"):
        batch_prompts = prompts[i : i + cfg.batch_size]
        
        # 编码并Padding
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=cfg.max_length
        ).to(cfg.device)
        
        # 生成配置
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,  # 选用 Greedy Search 确保确定性生物事实输出
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )
            
        # 解码并提取 Assistant 回答部分
        for j, out_ids in enumerate(outputs):
            # 获取输入 Prompt 的 token 长度，防止把 Prompt 重新打印出来
            input_len = inputs["input_ids"][j].shape[0]
            actual_gen_ids = out_ids[input_len:]
            
            # 解码为文本
            gen_text = tokenizer.decode(actual_gen_ids, skip_special_tokens=False)
            
            # 清洗遗留的特殊结束符标签
            if "<|im_end|>" in gen_text:
                gen_text = gen_text.split("<|im_end|>")[0]
            gen_text = gen_text.strip()
            
            predicted_answers.append(gen_text)
            
    # 6. 将预测结果追加回 Dataframe，并维持原始列格式输出
    df["predicted_answer"] = predicted_answers
    
    # 显式规定并检查列的保存顺序，确保符合你的预期要求
    expected_columns = [
        "question", "answer", "question_type", "given_step", 
        "total_step", "pathway_id", "entry_id", "phenotype", "predicted_answer"
    ]
    
    # 如果原始文件有多余或缺少的非核心列，通过此步安全过滤
    final_cols = [col for col in expected_columns if col in df.columns]
    if "predicted_answer" not in final_cols:
        final_cols.append("predicted_answer")
        
    df_final = df[final_cols]
    
    # 保存至新文件
    # df_final.to_csv(cfg.output_data_path, index=False, quoting=csv.QUOTE_NONE)
    df_final.to_csv(
        output_path,
        index=False, 
        quoting=csv.QUOTE_MINIMAL, 
        escapechar='\\'  # 如果文本内部包含双引号，用反斜杠转义，绝不错位
    )
    metadata_path = output_path.with_suffix(".run.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nInference completed successfully! Results saved to: {output_path}")
    print(f"Run configuration saved to: {metadata_path}")

if __name__ == "__main__":
    run_inference(parse_args())
