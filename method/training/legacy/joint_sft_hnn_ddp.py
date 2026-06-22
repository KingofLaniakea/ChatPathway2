import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdiffeq import odeint
import numpy as np

# 环境设置
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

class ThirdStageConfig:
    base_model_id = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/.cache/hf_models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    sft_lora_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_sft/checkpoint_epoch_5"
    ae_ckpt_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_ae_only/checkpoint_epoch_5/ae_proj.pt" 
    
    data_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/datasets/train_11_species_dataset.csv"
    save_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_stage3_sft_hnn"
    
    batch_size = 2  
    gradient_accumulation_steps = 8 
    lr = 2e-5 
    hnn_lr = 5e-5 
    epochs = 5
    max_length = 1072 
    latent_dim = 128 
    
    lambda_align = 0.5 
    lambda_f = 1e-4 
    lambda_d = 1e-3 

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)

class TDHNNFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, hidden_dim),
            Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.f_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, self.half_dim, bias=False)
        )
        self.d1 = nn.Parameter(torch.zeros(1, self.half_dim))
        nn.init.kaiming_normal_(self.d1)
        M = torch.eye(latent_dim)
        M = torch.cat([M[self.half_dim:], -M[:self.half_dim]])
        self.register_buffer('M', M)

    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            H = self.mlp(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            derivs = dH @ self.M.t()
            qdot, pdot = derivs[:, :self.half_dim], derivs[:, self.half_dim:]
            t_tensor = torch.full((x.size(0), 1), t, device=x.device, dtype=x.dtype)
            F = self.f_net(t_tensor)
            return torch.cat([qdot, pdot - torch.abs(self.d1) * pdot + F], dim=1)

class CascadeProjection(nn.Module):
    def __init__(self, high_dim=4096, mid_dim=1024, latent_dim=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(high_dim, mid_dim), nn.LayerNorm(mid_dim), nn.SiLU(), 
            nn.Linear(mid_dim, mid_dim // 2), nn.LayerNorm(mid_dim // 2), nn.SiLU(),
            nn.Linear(mid_dim // 2, latent_dim)
        )
        self.up = nn.Sequential(
            nn.Linear(latent_dim, mid_dim // 2), nn.SiLU(),
            nn.Linear(mid_dim // 2, mid_dim), nn.SiLU(),
            nn.Linear(mid_dim, high_dim)
        )
    def forward(self, x):
        latent = self.down(x)
        reconstructed = self.up(latent)
        return latent, reconstructed

class CSVPathwayDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.df = pd.read_csv(file_path)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt_text = f"<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        answer = str(row.get('answer', row.get('formatted_answer_no_phenotype', "")))
        answer_text = f"{answer}<|im_end|>"
        p_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        a_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        return {"input_ids": torch.tensor((p_ids + a_ids)[:self.max_length]), 
                "labels": torch.tensor(([-100]*len(p_ids) + a_ids)[:self.max_length])}

def collate_fn(batch):
    pad_id = 151643
    input_ids = torch.nn.utils.rnn.pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": (input_ids != pad_id).long()}

def train_third_stage():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    cfg = ThirdStageConfig()
    
    if local_rank == 0: os.makedirs(cfg.save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 1. 加载 SFT LoRA 模型
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.bfloat16, device_map={"": local_rank}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, cfg.sft_lora_path, is_trainable=True)
    model.enable_input_require_grads()

    # 2. 加载并冻结 AE (不包装 DDP)
    proj = CascadeProjection(high_dim=base_model.config.hidden_size, latent_dim=cfg.latent_dim).to(local_rank).bfloat16()
    sd = torch.load(cfg.ae_ckpt_path, map_location=f"cuda:{local_rank}")
    proj.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()})
    for p in proj.parameters(): p.requires_grad = False
    proj.eval()

    # 3. 初始化 HNN 并包装 DDP
    hnn_f = TDHNNFunc(cfg.latent_dim).to(local_rank).bfloat16()
    model = DDP(model, device_ids=[local_rank])
    hnn_f = DDP(hnn_f, device_ids=[local_rank])

    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': cfg.lr}, {'params': hnn_f.parameters(), 'lr': cfg.hnn_lr}])
    dataset = CSVPathwayDataset(cfg.data_path, tokenizer, cfg.max_length)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=DistributedSampler(dataset), collate_fn=collate_fn)

    history = {"total": [], "sft": [], "align": []}
    for epoch in range(cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train(); hnn_f.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        
        for step, batch in enumerate(pbar):
            inputs, mask, labels = batch['input_ids'].to(local_rank), batch['attention_mask'].to(local_rank), batch['labels'].to(local_rank)
            
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels, output_hidden_states=True)
            loss_sft = outputs.loss
            
            h_real = outputs.hidden_states[-1] 
            z_all, _ = proj(h_real) 
            
            last_p_idx = (labels == -100).sum(dim=1) - 1
            z0 = z_all[torch.arange(z_all.size(0)), last_p_idx]
            
            t_steps = torch.linspace(0.0, 1.0, h_real.size(1)).to(local_rank)
            
            hnn_f.module.float()
            z_traj = odeint(hnn_f.module, z0.float(), t_steps, method='rk4').transpose(0, 1).to(torch.bfloat16)
            hnn_f.module.bfloat16()
            
            v_hnn_latent = z_traj[:, 1:, :] - z_traj[:, :-1, :]
            v_hnn_high = proj.up(v_hnn_latent) 
            v_real = h_real[:, 1:, :] - h_real[:, :-1, :]

            cos_sim = torch.nn.functional.cosine_similarity(v_hnn_high, v_real, dim=-1)
            v_mask = mask[:, 1:].float()
            loss_align = ((1.0 - cos_sim) * v_mask).sum() / (v_mask.sum() + 1e-8)
            
            loss_phys_reg = cfg.lambda_f * sum(p.abs().sum() for p in hnn_f.module.f_net.parameters()) + cfg.lambda_d * hnn_f.module.d1.abs().sum()
            
            total_loss = (loss_sft + cfg.lambda_align * loss_align + loss_phys_reg) / cfg.gradient_accumulation_steps
            total_loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(hnn_f.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
                if local_rank == 0:
                    pbar.set_postfix({"SFT": f"{loss_sft.item():.3f}", "Align": f"{loss_align.item():.4f}"})

        if local_rank == 0:
            save_dir = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.module.save_pretrained(save_dir)
            torch.save(hnn_f.module.state_dict(), os.path.join(save_dir, "hnn_func.pt"))

    dist.destroy_process_group()

if __name__ == "__main__":
    train_third_stage()