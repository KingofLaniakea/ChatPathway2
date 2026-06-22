import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class InferenceConfig:
    # 基础模型和训练好的第2代LoRA Checkpoint路径
    base_model_id = "/root/autodl-tmp/models/qwen3_8B"
    trained_lora_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_2"
    
    # 数据集输入输出路径
    test_data_path = "/root/autodl-tmp/data/test_7_species_dataset_small.csv"
    output_data_path = "/root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch2_small.csv"
    
    # 推理超参数（开启大 Batch 提升效率）
    batch_size = 8  # 显存宽裕的话，可以尝试加大到 16 或 32
    max_length = 1072
    max_new_tokens = 768
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

def run_batch_inference():
    cfg = InferenceConfig()
    print(f"Using device: {cfg.device}")
    
    # 1. 初始化 Tokenizer
    print(f"Loading tokenizer from {cfg.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 大模型批量生成时，必须将 Padding 侧设置为左侧（Left Padding）
    # 这样能确保所有样本的 Prompt 末尾（即推理起点）在右侧完美对齐
    tokenizer.padding_side = "left" 
    
    # 2. 加载基础模型与训练好的LoRA权重
    print(f"Loading base model from {cfg.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map={"": cfg.device}, 
        trust_remote_code=True
    )
    
    print(f"Loading trained SFT+HNN adapter from {cfg.trained_lora_path}...")
    model = PeftModel.from_pretrained(base_model, cfg.trained_lora_path)
    model.eval()  # 切换至评估模式
    
    # 3. 读取测试集
    print(f"Reading test dataset from {cfg.test_data_path}...")
    df = pd.read_csv(cfg.test_data_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    
    # 4. 构建标准的 Prompt 格式
    prompts = []
    for _, row in df.iterrows():
        prompt_text = f"<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt_text)
        
    predicted_answers = []
    
    # 5. 分批次进行高性能批量自回归推理
    print(f"Starting batch inference (batch_size={cfg.batch_size})...")
    for i in tqdm(range(0, len(prompts), cfg.batch_size), desc="Inferencing"):
        batch_prompts = prompts[i : i + cfg.batch_size]
        
        # Tokenizer 会自动在较短的 Prompt 左侧填充 [PAD] 符号，实现右侧对齐
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=cfg.max_length
        ).to(cfg.device)
        
        # 获取当前 Batch 编码后的输入矩阵宽度（含左侧 Padding）
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,  # Greedy Search 确保物理确定性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )
            
        # 🌟【核心修改 2】：精准切片截取真正生成的答案
        # 因为采用了左填充，model.generate() 吐出的新 Token 会严格追加在矩阵的右侧
        # 此时每一行的 out_ids 结构为: [左侧数量不等的PAD] + [原始Prompt] + [新生成的Answer]
        # 也就是说，我们只需要直接切掉前 input_length 宽度的内容，剩下的就是纯粹的答案，天然杜绝了 PAD 符号污染！
        for j, out_ids in enumerate(outputs):
            actual_gen_ids = out_ids[input_length:]
            
            # 解码为文本
            gen_text = tokenizer.decode(actual_gen_ids, skip_special_tokens=False)
            
            # 清洗遗留的特殊结束符标签
            if "<|im_end|>" in gen_text:
                gen_text = gen_text.split("<|im_end|>")[0]
            gen_text = gen_text.strip()
            
            predicted_answers.append(gen_text)
            
    # 6. 将预测结果追加回 Dataframe，并维持原始列格式输出
    df["predicted_answer"] = predicted_answers
    
    expected_columns = [
        "question", "answer", "question_type", "given_step", 
        "total_step", "pathway_id", "entry_id", "phenotype", "predicted_answer"
    ]
    
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
    run_batch_inference()
