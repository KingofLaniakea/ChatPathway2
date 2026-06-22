import os
# 🚨 核心修改：在导入 torch 之前，强行指定只看物理 GPU 1
# 这样操作系统会把物理 GPU 1 映射为当前脚本的 "cuda:0"，完全隔离物理 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_zero_shot_inference():
    # ================= 1. 路径配置 =================
    model_path = "/root/autodl-tmp/C2S-Scale-Gemma-2-2B"
    test_dataset_path = "/root/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small.jsonl"
    
    # 此时这里的 "cuda" 实际上指向的就是物理上的 GPU 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 隔离环境成功！当前脚本使用的安全设备: {device} (对应物理 GPU 1)")

    # ================= 2. 加载原生的 Tokenizer 和 2B 模型 =================
    print(f"[*] 正在加载官方 C2S-Gemma-2B 模型与词表...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Gemma-2 的 Pad Token 兜底处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # 使用 bf16 精度，显存占用极小，计算速度快
        attn_implementation="sdpa",   # 启用 PyTorch 自带的缩放点积注意力加速
        trust_remote_code=True
    ).to(device)
    model.eval()

    # ================= 3. 读取测试集的第一条样本 =================
    if not os.path.exists(test_dataset_path):
        print(f"[!] 错误：未找到测试集文件 {test_dataset_path}，请确认路径是否正确。")
        return

    print(f"[*] 正在读取测试集样本...")
    first_sample = None
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                first_sample = json.loads(line.strip())
                break # 只拿第一条
                
    if not first_sample:
        print("[!] 错误：测试集为空文件。")
        return

    inference_prompt = first_sample["instruction"]
    ground_truth_response = first_sample["output"]

    print("\n" + "="*30 + " 1. 输入 Prompt " + "="*30)
    print(inference_prompt)

    # ================= 4. 编码输入 =================
    input_ids = tokenizer.encode(inference_prompt, return_tensors="pt").to(device)

    # ================= 5. 生成预测结果 =================
    print("\n[*] 官方大模型正在 GPU 1 上预测中（Zero-shot 状态）...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1000,       # 对应官方设定的 ~4 tokens per gene 长度
            do_sample=False,          # 关闭采样，采用 Greedy Search 确保预测稳定性
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 截取生成的文本（去掉 Prompt 部分）
    generated_tokens = outputs[0][len(input_ids[0]):]
    predicted_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # ================= 6. 打印终极大比拼 =================
    print("\n" + "="*30 + " 2. Ground Truth (真实观测细胞) " + "="*30)
    print(ground_truth_response)

    print("\n" + "="*30 + " 3. Model Prediction (预训练模型预测) " + "="*30)
    print(predicted_response if predicted_response else "[模型未输出任何基因Token或直接触发了终止符]")
    print("="*75)

if __name__ == "__main__":
    run_zero_shot_inference()