import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ================= 1. config =================
class Config:
    base_model_id = "/root/autodl-tmp/models/qwen3_8b_stage3_full_merged"
    train_dataset_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"
    save_path = "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft"
    
    batch_size = 2                 
    gradient_accumulation_steps = 8
    lr = 2e-5
    epochs = 3 
    max_length = 1648
    
    top_k_genes = 200
    perturbation_col = 'target_gene'
    control_label = 'non-targeting'
    
    lora_config = {
        "r": 32,                    
        "lora_alpha": 64,         
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }

class C2SPairingDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.top_k_genes = cfg.top_k_genes
        
        raw_dataset = load_from_disk(cfg.train_dataset_path)
        
        self.custom_input_prompt_template = """Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.
Control cell sentence: {control_cell_sentence}.

Perturbed cell sentence:"""
        self.answer_template = "{perturbed_cell_sentence}."
        
        self.control_indices = []
        self.pert_pairs = [] 
        
        for i, sample in enumerate(raw_dataset):
            if sample[cfg.perturbation_col] == cfg.control_label:
                self.control_indices.append(i)
            else:
                self.pert_pairs.append((sample[cfg.perturbation_col], i))
                
        self.raw_dataset = raw_dataset
        assert len(self.control_indices) > 0, "未在数据集中检索到任何 Control 细胞，请检查 label！"

    def _get_cell_sentence_str(self, sample):
        raw_sentence = str(sample.get('cell_sentence', ''))
        words = raw_sentence.split()
        selected_words = words[:self.top_k_genes]
        return " ".join(selected_words), str(len(selected_words))

    def __len__(self): 
        return len(self.pert_pairs)
        
    def __getitem__(self, idx):
        random.seed(idx)
        
        pert_name, perturbed_idx = self.pert_pairs[idx]
        perturbed_sample = self.raw_dataset[perturbed_idx]
        
        control_idx = random.choice(self.control_indices)
        control_sample = self.raw_dataset[control_idx]
        
        control_sentence, num_genes_str = self._get_cell_sentence_str(control_sample)
        perturbed_sentence, _ = self._get_cell_sentence_str(perturbed_sample)
        
        model_input_str = self.custom_input_prompt_template.format(
            num_genes=num_genes_str,
            perturbation_name=pert_name,
            control_cell_sentence=control_sentence
        )
        response_str = self.answer_template.format(
            perturbed_cell_sentence=perturbed_sentence
        )
        
        prompt_text = f"<|im_start|>user\n{model_input_str}<|im_end|>\n<|im_start|>assistant\n"
        answer_text = f"{response_str}<|im_end|>"
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        
        full_ids = (prompt_ids + answer_ids)[:self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:self.max_length]
        
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

# ================= 3. DDP Setup =================
def setup():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup():
    dist.destroy_process_group()

# ================= 4. train =================
def train():
    setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg = Config()
    
    if local_rank == 0:
        os.makedirs(cfg.save_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print(f"1. 正在加载融合了上阶段先验的完整底座: {cfg.base_model_id}")
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map=None        
    )
    
    
    if local_rank == 0:
        print(f"2. 正在初始化 Stage 4 (C2S) 的全新 LoRA 参数...")
        
    lora_config = LoraConfig(**cfg.lora_config)
    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    full_dataset = C2SPairingDataset(cfg, tokenizer)
    sampler = DistributedSampler(full_dataset)
    train_loader = DataLoader(
        full_dataset, 
        batch_size=cfg.batch_size, 
        sampler=sampler, 
        collate_fn=collate_fn
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_history = []

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * cfg.gradient_accumulation_steps
            if local_rank == 0:
                pbar.set_postfix({"loss": f"{loss.item() * cfg.gradient_accumulation_steps:.4f}"})

        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"[Epoch {epoch+1}] average Loss: {avg_loss:.4f}")
            
            epoch_save_path = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
            model.module.save_pretrained(epoch_save_path)
            print(f"Checkpoint 已保存至: {epoch_save_path}")
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', linestyle='--')
            plt.title('ChatPathway Stage 4 (C2S) SFT Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.grid(True)
            plt.savefig(os.path.join(cfg.save_path, 'c2s_sft_loss_curve.png'))
            plt.close()

    cleanup()

if __name__ == "__main__":
    train()
