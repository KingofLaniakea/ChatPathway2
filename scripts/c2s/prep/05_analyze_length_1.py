import json
import numpy as np
from transformers import AutoTokenizer

def analyze_dataset_context():
    # ================= 🚀 路径配置（已根据你的截图和之前信息对齐） =================
    base_model_path = "/root/autodl-tmp/models/qwen3_8b_stage3_full_merged"
    dataset_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_qa_datasets.jsonl"
    
    print("正在加载本地底座 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 获取 Qwen 控制 Token ID
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645
    
    prompt_token_lens = []
    answer_token_lens = []
    total_token_lens = []
    
    char_lens_prompt = []
    char_lens_answer = []
    
    zero_tokens_error = 0
    total_samples = 0
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_samples += 1
            item = json.loads(line.strip())
            
            instruction = item["instruction"]
            output = item["output"]
            
            # 统计纯字符串字数
            char_lens_prompt.append(len(instruction))
            char_lens_answer.append(len(output))
            
            # 数字化分词
            raw_prompt_ids = tokenizer.encode(instruction, add_special_tokens=False)
            raw_answer_ids = tokenizer.encode(output, add_special_tokens=False)
            
            p_len = len(raw_prompt_ids)
            a_len = len(raw_answer_ids)
            # 算上我们手动硬编码拼上去的：<|im_start|>user\n (4 tokens) 和 <|im_end|>\n<|im_start|>assistant\n (4 tokens)
            t_len = p_len + a_len + 8 
            
            prompt_token_lens.append(p_len)
            answer_token_lens.append(a_len)
            total_token_lens.append(t_len)
            
            if p_len == 0 or a_len == 0:
                zero_tokens_error += 1
            
            # 🌟 打印前 2 条样本的微观切词细节，排查有没有变成一堆无意义重复 ID
            if total_samples <= 2:
                print(f"\n" + "-"*30 + f" 样例 {total_samples} 编码细节核对 " + "-"*30)
                print(f"【Instruction 文本片段】: {instruction[:80]}...")
                print(f" -> 对应 Token 数量: {p_len} | 前 8 个 Token ID 分别为: {raw_prompt_ids[:8]}")
                print(f"【Output 文本片段】: {output[:80]}...")
                print(f" -> 对应 Token 数量: {a_len} | 前 8 个 Token ID 分别为: {raw_answer_ids[:8]}")
                print(f" -> 预期总拼装长度: {t_len}")

    # ================= 📈 生成宏观统计报告 =================
    print("\n" + "="*20 + " 数据集上下文分析报告 " + "="*20)
    print(f"1. 样本总数 (Total Samples)       : {total_samples} 条")
    print(f"2. 分词异常数 (0 Token Samples)   : {zero_tokens_error} 条")
    print("-" * 65)
    
    print("3. 【Prompt 题目部分 Token 长度统计】")
    print(f"   - 最短: {np.min(prompt_token_lens)} | 平均: {np.mean(prompt_token_lens):.1f} | 99% 分位数: {np.percentile(prompt_token_lens, 99):.1f} | 最长: {np.max(prompt_token_lens)}")
    print(f"   - 平均每个基因耗费的 Token 数: {np.mean(prompt_token_lens) / 200:.2f} (理想状态接近 1.0~1.5)")
    print("-" * 65)
    
    print("4. 【Answer 答案部分 Token 长度统计】")
    print(f"   - 最短: {np.min(answer_token_lens)} | 平均: {np.mean(answer_token_lens):.1f} | 99% 分位数: {np.percentile(answer_token_lens, 99):.1f} | 最长: {np.max(answer_token_lens)}")
    print("-" * 65)
    
    print("5. 【全长分布与 max_length (1648) 适配度】")
    print(f"   - 拼装后最短总长: {np.min(total_token_lens)}")
    print(f"   - 拼装后平均总长: {np.mean(total_token_lens):.1f}")
    print(f"   - 拼装后 99% 分位数: {np.percentile(total_token_lens, 99):.1f}")
    print(f"   - 拼装后最长样本: {np.max(total_token_lens)}")
    
    over_limit = sum(1 for l in total_token_lens if l > 1648)
    print(f"   - 超过设定最大长度 1648 的样本数: {over_limit} 条 (占比: {over_limit / total_samples * 100:.2f}%)")
    print("=" * 63)

if __name__ == "__main__":
    analyze_dataset_context()
