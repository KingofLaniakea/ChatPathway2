import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from tqdm import tqdm

# ================= 1. Configuration =================
class Config:
    base_model_id = "/root/autodl-tmp/qwen3_8B"
    stage3_lora_path = "/root/autodl-tmp/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5"
    train_dataset_path = "/root/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small.jsonl"
    save_path = "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small"
    
    batch_size = 8        
    gradient_accumulation_steps = 8
    lr = 2e-5
    epochs = 5
    max_length = 1648
    
    # LoRA config for aligning target modules during sequential fine-tuning
    lora_config = {
        "r": 64,                    
        "lora_alpha": 128,         
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }

class C2SPairingDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.data = []
        
        with open(cfg.train_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        print(f"[+] Successfully loaded preprocessed dataset with {len(self.data)} samples.")

        # Dynamically extract Qwen control token IDs from the vocabulary
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
        
        # Manually construct Qwen standard Chat template structure
        prompt_ids = [self.im_start_id] + self.user_ids + raw_prompt_ids + [self.im_end_id, 10] + [self.im_start_id] + self.assistant_ids
        answer_ids = raw_answer_ids + [self.im_end_id]
        
        full_ids = (prompt_ids + answer_ids)[:self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:self.max_length]
        
        # Edge case handling fallback
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

# ================= 2. Training Loop =================
def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_path, exist_ok=True)
    
    print(f"[*] Loading tokenizer from base model: {cfg.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(cfg.save_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[1/4] Loading base model: {cfg.base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16,   
        attn_implementation="sdpa",   
        trust_remote_code=True,
        device_map="cuda:0"
    )
    
    print(f"[2/4] Loading Stage 3 LoRA weights as base matrix: {cfg.stage3_lora_path}")
    model = PeftModel.from_pretrained(base_model, cfg.stage3_lora_path, is_trainable=True)
    
    print(f"[3/4] Explicitly activating and validating LoRA gradients...")
    peft_config = LoraConfig(**cfg.lora_config)
    
    # Force older LoRA matrices into a trainable state to prevent frozen gradients
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Print current trainable parameters summary
    model.print_trainable_parameters()
    
    full_dataset = C2SPairingDataset(cfg, tokenizer)
    train_loader = DataLoader(
        full_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg.lr)
    loss_history = []

    print(f"[4/4] Warm start successful. Initiating Stage 4 (C2S) fine-tuning.")
    
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            raw_loss_val = outputs.loss.item()
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += raw_loss_val
            pbar.set_postfix({"loss": f"{raw_loss_val:.6f}"})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.6f}")
        
        epoch_save_path = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        print(f"[+] Weights and tokenizer successfully saved to: {epoch_save_path}")
        
        # Plot and save loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', linestyle='--')
        plt.title('ChatPathway Stage 4 (C2S) SFT Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(cfg.save_path, 'c2s_sft_loss_curve.png'))
        plt.close()

if __name__ == "__main__":
    train()