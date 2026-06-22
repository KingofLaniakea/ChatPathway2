import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class InferenceConfig:
    base_model_id = "/root/autodl-tmp/qwen3_8B"
    # trained_lora_path = "/root/autodl-tmp/qwen3_8b_sft/checkpoint_epoch_5"
    trained_lora_path = "/root/autodl-tmp/qwen3_8b_FrameworkA_1/checkpoint_epoch_4"
    # trained_lora_path = "/root/autodl-tmp/qwen3_8b_FrameworkA_ae_cos/checkpoint_epoch_4"
    # 数据集输入输出路径
    test_data_path = "/root/autodl-tmp/test_7_species_dataset.csv"
    
    output_data_path = "/root/autodl-tmp/test_7_predictions_ae_cos.csv"
    
    # 推理超参数
    batch_size = 8
    max_length = 1072
    max_new_tokens = 768
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

def run_inference():
    cfg = InferenceConfig()
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
    
    print(f"Loading trained SFT+HNN adapter from {cfg.trained_lora_path}...")
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
        cfg.output_data_path, 
        index=False, 
        quoting=csv.QUOTE_MINIMAL, 
        escapechar='\\'  # 如果文本内部包含双引号，用反斜杠转义，绝不错位
    )
    print(f"\nInference completed successfully! Results saved to: {cfg.output_data_path}")

if __name__ == "__main__":
    run_inference()