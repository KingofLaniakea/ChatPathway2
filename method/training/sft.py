import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ================= 1. 配置类 =================
class Config:
    # --- 路径配置 (8B 模型) ---
    base_model_id = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/.cache/hf_models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    train_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/datasets/train_11_species_dataset.csv"
    save_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_sft"
    
    # --- 训练超参数 ---
    batch_size = 4                  
    gradient_accumulation_steps = 4 
    lr = 2e-5
    epochs = 5                     
    max_length = 1072               
    
    # --- LoRA 配置 ---
    lora_config = {
        "r": 64,                    
        "lora_alpha": 128,         
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }

# ================= 2. 数据集类 =================
class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = pd.read_csv(file_path)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row['question'])
        answer = str(row.get('answer', row.get('formatted_answer_no_phenotype', "")))
        prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        answer_text = f"{answer}<|im_end|>"
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        full_ids = (prompt_ids + answer_ids)[:self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:self.max_length]
        return {"input_ids": torch.tensor(full_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

def collate_fn(batch):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    pad_id = 151643
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded, "attention_mask": (input_ids_padded != pad_id).long()}

# ================= 3. 分布式初始化与清理 =================
def setup():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup():
    dist.destroy_process_group()

# ================= 4. 主训练流程 =================
def train():
    setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg = Config()
    
    if local_rank == 0:
        os.makedirs(cfg.save_path, exist_ok=True)
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载 Base Model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map=None # 手动管理设备映射
    )
    
    # 3. 开启梯度检查点 (节省显存，8B 模型必须开启)
    model.gradient_checkpointing_enable()
    
    # 4. 配置并应用 LoRA
    lora_config = LoraConfig(**cfg.lora_config)
    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    # 5. 移动到设备并包装 DDP
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 6. 准备数据
    full_dataset = SFTDataset(cfg.train_path, tokenizer, cfg.max_length)
    sampler = DistributedSampler(full_dataset)
    train_loader = DataLoader(
        full_dataset, 
        batch_size=cfg.batch_size, 
        sampler=sampler, 
        collate_fn=collate_fn
    )

    # 7. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    
    loss_history = []

    # 8. 训练循环
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            
            # 梯度更新
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * cfg.gradient_accumulation_steps
            if local_rank == 0:
                pbar.set_postfix({"loss": f"{loss.item() * cfg.gradient_accumulation_steps:.4f}"})

        # 9. 保存与绘图 (每个 Epoch 结束)
        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")
            
            # 保存模型
            epoch_save_path = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
            model.module.save_pretrained(epoch_save_path)
            
            # 绘制 Loss 曲线
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='g')
            plt.title('Qwen3-8B SFT Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(cfg.save_path, 'sft_loss_curve.png'))
            plt.close()

    cleanup()

if __name__ == "__main__":
    train()