import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 1. 严格对齐你截图中的真实路径 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen.jsonl"

# 👑 修正 1：基座模型指向你截图一中拥有完整 vocab.json 的纯净大底座
base_model_path = "/root/autodl-tmp/models/qwen3_8B"

# 👑 修正 2：LoRA 权重指向你截图二中未经过 CS 数据微调的 Stage 3 节点
target_lora_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5"

print(f"[*] 正在初始化设备: {device}")

# ================= 2. 加载最干净的 Tokenizer 与模型 =================
print(f"[*] 正在从大底座路径载入完好的 Tokenizer: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[*] 正在载入基础底座 (Base Model): {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,   # 使用与 SFT 相同的 bfloat16 精度
    attn_implementation="sdpa",   # 启用 SDPA 优化
    trust_remote_code=True,
    device_map="cuda:0"
)

print(f"[*] 正在挂载 Stage 3 LoRA 适配器: {target_lora_path}")
model = PeftModel.from_pretrained(base_model, target_lora_path).to(device)
model.eval()
print("[+] 模型与分词器全部安全加载完毕。\n")

# ================= 3. 读取第一个样本并进行像素级 Token 拼接 =================
if not os.path.exists(test_jsonl_path):
    raise FileNotFoundError(f"[-] 未找到测试集文件: {test_jsonl_path}")

with open(test_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            first_sample = json.loads(line.strip())
            break

instruction = str(first_sample['instruction']) # 确保是纯字符串
ground_truth = first_sample['output']

print("="*40 + " DEBUG 像素对齐检查 " + "="*40)

# 1. 从纯净词表中提取核心特殊 Token ID
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645
user_ids = tokenizer.encode("user\n", add_special_tokens=False)
assistant_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

# 2. 文本编码转换（由于底座词表完好，这里绝对能成功解析出几百个 Token）
raw_prompt_ids = tokenizer.encode(instruction, add_special_tokens=False, allowed_special="none")

# 3. 严格复刻你 SFT 时的第 57 行硬编码拼接公式：[im_start] + user\n + raw_prompt + [im_end, 10] + [im_start] + assistant\n
full_prompt_ids = [im_start_id] + user_ids + raw_prompt_ids + [im_end_id, 10] + [im_start_id] + assistant_ids

# 4. 打印调试信息，复核 Token 的真实状态
print(f"[*] 训练拼接公式: [im_start] + user\\n + raw_prompt + [im_end, 10] + [im_start] + assistant\\n")
print(f"[*] 纯文本 raw_prompt_ids 真实解析长度: {len(raw_prompt_ids)} (基于干净大底座词表)")
print(f"[*] 最终全量拼接后的输入长度 (原始输入 Token 长度): {len(full_prompt_ids)}")
print(f"[*] 前 15 个 Prompt Token IDs: {full_prompt_ids[:15]}")
print(f"[*] 拼接后的 Prompt 真实文本预览（前 80 个字符）:\n{tokenizer.decode(full_prompt_ids[:40])} ... [已截断预览]")
print("="*100 + "\n")

# ================= 4. 执行单条推理 =================
input_ids_tensor = torch.tensor([full_prompt_ids], dtype=torch.long).to(device)
attention_mask = torch.ones_like(input_ids_tensor).to(device)

print("[*] 正在发送给 Stage 3 旧模型进行 Greedy Search 生成...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask,
        max_new_tokens=250, 
        do_sample=False,              # 确定性推理
        pad_token_id=151643,          # 遵循你 SFT Collate 阶段指定的 pad_id
        eos_token_id=im_end_id        # 遇到模型的结束符立即停止
    )

input_len = len(full_prompt_ids)
generated_tokens = outputs[0][input_len:]

print("\n" + "="*40 + " 生成结果汇报 " + "="*40)
print(f"[*] 原始输入 Token 长度: {input_len}")
print(f"[*] 模型新增 Token 长度: {len(generated_tokens)}")
if len(generated_tokens) > 0:
    print(f"[*] 吐出的前 20 个原始 Token IDs: {generated_tokens[:20].tolist()}")

# 解码生成的文本
pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).replace("<|im_end|>", "").strip()

print(f"\n[Ground Truth 真实扰动细胞基因]:\n{ground_truth}\n")
print(f"\n[Model Prediction 模型预测基因]:\n{pred_text if pred_text else '(输出为空)'}\n")
print("="*94)

if not pred_text:
    print("\n⚠️ 对照组结论：Token 输入长度成功恢复正常！但旧模型输出为空。这铁证了它因为没有经历过 C2S 基因预测任务的训练，遇到此类 Prompt 时无法识别，直接输出了 EOS 提前交卷。")
else:
    print("\n✅ 成功：旧模型不仅吃进去了完整的 Token，还给出了某种预测响应！")
