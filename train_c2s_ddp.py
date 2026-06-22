import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from tqdm import tqdm

# ================= 1. DDP 初始化 =================
def init_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# ================= 2. config =================
class Config:
    base_model_id = "/root/autodl-tmp/qwen3_8B"
    stage3_lora_path = "/root/autodl-tmp/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5"
    train_dataset_path = "/root/CRISPR_GSE264667_Data/jurkat_c2s_train_seen.jsonl"
    save_path = "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_train"
    
    # 👑 双卡下单卡分配 16 (两张卡实际总 batch 达 32)
    batch_size = 16          
    gradient_accumulation_steps = 4
    lr = 2e-5
    epochs = 1
    max_length = 1648
    
    lora_config = {
        "r": 64,                    
        "lora_alpha": 128,         
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }

class C2SPairingDataset(Dataset):
    def __init__(self, cfg, tokenizer, is_main_process=True):
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.data = []
        
        with open(cfg.train_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        if is_main_process:
            print(f"[+] 成功加载预处理数据集，共 {len(self.data)} 条样本。")

        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645
        self.user_ids = self.tokenizer.encode("user\n", add_special_tokens=False)
        self.assistant_ids = self.tokenizer.encode("assistant\n", add_special_tokens=False)

    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = str(item["instruction"])
        output = str(item["output"])
        
        raw_prompt_ids = self.tokenizer.encode(instruction, add_special_tokens=False, allowed_special="none")
        raw_answer_ids = self.tokenizer.encode(output, add_special_tokens=False, allowed_special="none")
        
        prompt_ids = [self.im_start_id] + self.user_ids + raw_prompt_ids + [self.im_end_id, 10] + [self.im_start_id] + self.assistant_ids
        answer_ids = raw_answer_ids + [self.im_end_id]
        
        full_ids = (prompt_ids + answer_ids)[:self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:self.max_length]
        
        if len(full_ids) == 0 or all(l == -100 for l in labels):
            full_ids = [self.im_start_id] + self.user_ids + [151643] + [self.im_end_id]
            labels = [-100, -100, 151643, -100]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long), 
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    pad_id = 151643 
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids_padded, 
        "labels": labels_padded, 
        "attention_mask": (input_ids_padded != pad_id).long()
    }

# ================= 3. train =================
def train():
    local_rank = init_ddp()
    is_main_process = (local_rank == 0)
    
    cfg = Config()
    if is_main_process:
        os.makedirs(cfg.save_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print(f"1. 正在加载纯净的基础 8B 底座模型: {cfg.base_model_id}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16,   
        attn_implementation="sdpa",   
        trust_remote_code=True,
        device_map=f"cuda:{local_rank}"  # 👑 严格绑定当前进程的显卡
    )
    
    if is_main_process:
        print(f"2. 👑 正在载入 Stage 3 LoRA 权重作为基础矩阵...")
    model = PeftModel.from_pretrained(base_model, cfg.stage3_lora_path, is_trainable=True)
    
    # 显式激活增量 LoRA 梯度配置
    peft_config = LoraConfig(**cfg.lora_config)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    if is_main_process:
        model.print_trainable_parameters()
    
    full_dataset = C2SPairingDataset(cfg, tokenizer, is_main_process=is_main_process)
    
    # 👑 分布式采样器，实现双卡不重复地均分训练集
    sampler = DistributedSampler(full_dataset, shuffle=True)
    train_loader = DataLoader(
        full_dataset, 
        batch_size=cfg.batch_size, 
        sampler=sampler,
        collate_fn=collate_fn
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg.lr)
    
    # 👑 穿上 DDP 外衣
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank,
        find_unused_parameters=False
    )

    if is_main_process:
        print(f"4. 增量热启动与 DDP 结合成功！正式开启【双卡分布式加速】训练。")

    loss_history = []
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        # 仅在主进程打印进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_main_process else train_loader
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            raw_loss_val = outputs.loss.item()
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += raw_loss_val
            if is_main_process:
                pbar.set_postfix({"loss": f"{raw_loss_val:.6f}"})

        # 仅由主卡保存模型权重，防止写盘冲突
        if is_main_process:
            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"[Epoch {epoch+1}] average Loss: {avg_loss:.6f}")
            
            epoch_save_path = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
            model.module.save_pretrained(epoch_save_path) # 注意 DDP 包裹下需调用 .module
            tokenizer.save_pretrained(epoch_save_path)
            print(f"[+] 新权重与完整词表已由主卡成功保存至: {epoch_save_path}")
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', linestyle='--')
            plt.title('ChatPathway Stage 4 (C2S) SFT Loss Curve (DDP)')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.grid(True)
            plt.savefig(os.path.join(cfg.save_path, 'c2s_sft_loss_curve.png'))
            plt.close()
            
    dist.destroy_process_group()

if __name__ == "__main__":
    train()