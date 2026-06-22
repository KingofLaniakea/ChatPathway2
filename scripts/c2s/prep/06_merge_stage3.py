import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path = "/root/autodl-tmp/models/qwen3_8B"
stage3_lora_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5"
output_path = "/root/autodl-tmp/models/qwen3_8b_stage3_full_merged"

print("========== 1. 正在加载原始 8B 基座模型 (使用 CPU 稳妥融合) ==========")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    device_map="cpu"  
)

print("========== 2. 正在加载 Stage 3 LoRA 权重 (adapter_model) ==========")
model = PeftModel.from_pretrained(base_model, stage3_lora_path)

print("========== 3. 正在执行矩阵物理合并 (merge_and_unload) ==========")
merged_model = model.merge_and_unload()

print(f"========== 4. 正在写盘保存全新的完整预训练模型 ==========")
merged_model.save_pretrained(output_path)
print(f"\n融合彻底完成！新阶段微调的绝对安全底座已生成在:\n{output_path}\n")
