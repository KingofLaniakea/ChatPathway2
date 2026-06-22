import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
from torchdiffeq import odeint
import numpy as np

class ThirdStageConfig:
    base_model_id = "/root/autodl-tmp/models/qwen3_8B"
    sft_lora_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
    # ae_ckpt_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_new_1/ae_epoch_4/ae_proj.pt"
    ae_ckpt_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"
    data_path = "/root/autodl-tmp/data/train_11_species_dataset.csv"
    save_path = "/root/autodl-tmp/checkpoints/qwen3_8b_FrameworkA_ae_cos"
    
    batch_size = 4
    gradient_accumulation_steps = 12
    lr = 1e-4
    hnn_lr = 2e-4
    epochs = 4
    max_length = 1072 
    latent_dim = 128 
    
    lambda_align = 0.5 
    lambda_f = 1e-4 
    lambda_d = 1e-3 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, file_path, tokenizer, max_length=1072):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.df = pd.read_csv(
            file_path, 
            engine='python',
            quoting=csv.QUOTE_MINIMAL, 
            on_bad_lines='skip'
        )
        
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
    cfg = ThirdStageConfig()
    device = cfg.device
    os.makedirs(cfg.save_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # ---- 【修改 1】：将加载路径直接指向 epoch 2 的 checkpoint ----
    # epoch_2_dir = os.path.join(cfg.save_path, "checkpoint_epoch_2")

    # print(f"Loading base model and resuming LoRA from {epoch_2_dir}...")
    # base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True)
    # # 直接从 epoch_2 加载处于可训练状态的 model
    # model = PeftModel.from_pretrained(base_model, epoch_2_dir, is_trainable=True)
    # model.enable_input_require_grads()


    print(f"Loading base model from {cfg.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True)
    print(f"Loading SFT LoRA adapter from {cfg.sft_lora_path}...")
    model = PeftModel.from_pretrained(base_model, cfg.sft_lora_path, is_trainable=True)
    model.enable_input_require_grads()

    print(f"Loading pre-trained AE projector from {cfg.ae_ckpt_path}...")
    proj = CascadeProjection(high_dim=base_model.config.hidden_size, latent_dim=cfg.latent_dim).to(device).bfloat16()
    sd = torch.load(cfg.ae_ckpt_path, map_location=device)
    proj.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()})
    for p in proj.parameters(): p.requires_grad = False
    proj.eval()

    # ---- 【修改 2】：加载保存的 HNN 权重 ----
    # hnn_f = TDHNNFunc(cfg.latent_dim).to(device).bfloat16()
    # hnn_f_path = os.path.join(epoch_2_dir, "hnn_func.pt")
    # if os.path.exists(hnn_f_path):
    #     print(f"Loading resumed HNN parameters from {hnn_f_path}...")
    #     hnn_f.load_state_dict(torch.load(hnn_f_path, map_location=device))

    hnn_f = TDHNNFunc(cfg.latent_dim).to(device).bfloat16()

    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': cfg.lr}, {'params': hnn_f.parameters(), 'lr': cfg.hnn_lr}])
    
    dataset = CSVPathwayDataset(cfg.data_path, tokenizer, cfg.max_length)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)


    # print("Resuming Joint SFT + HNN Fine-Tuning from Epoch 3...")

    history = {"loss_sft": [], "loss_align": [], "loss_phys_reg": []}

    # print("Starting Joint SFT + HNN Fine-Tuning on single GPU (Optimized Matrix Alignment)...")
    # ---- 【修改 4】：循环范围改为从第三轮到设定的总轮数 ----
    # for epoch in range(2, cfg.epochs):
    for epoch in range(cfg.epochs):
        model.train()
        hnn_f.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        epoch_sft, epoch_align, epoch_phys = 0.0, 0.0, 0.0
        
        for step, batch in enumerate(pbar):
            inputs, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels, output_hidden_states=True)
            loss_sft = outputs.loss
            
            h_real = outputs.hidden_states[-1] 
            z_all, _ = proj(h_real) 
            
            # 1. Pinpoint the last Token position of Prompt as the starting point of HNN Evolution Z 0
            last_p_idx = (labels == -100).sum(dim=1) - 1
            z0 = z_all[torch.arange(z_all.size(0)), last_p_idx]
            
            # 2. dynamic access to the current Batch of all internal samples, the true answer Token maximum length

            max_ans_len = (labels != -100).sum(dim=1).max().item()
            if max_ans_len == 0: 
                max_ans_len = 1
                
            # The time step is only divided into Max + 1 steps
            t_steps = torch.linspace(0.0, 1.0, max_ans_len + 1).to(device)
            
            # 3. Use Ode to solve the HNN potential trajectory (at this time, only the answer span evolution)
            hnn_f.float()
            z_traj = odeint(hnn_f, z0.float(), t_steps, method='rk4').transpose(0, 1).to(torch.bfloat16)
            hnn_f.bfloat16()
            
            # 4. The change velocity corresponding to the low-dimensional trajectory on the HNN side is calculated and projected to the high-dimensional hidden space
            # v_hnn_high : [Batch, max_ans_len, High_dim]
            v_hnn_latent = z_traj[:, 1:, :] - z_traj[:, :-1, :]
            # v_hnn_high = proj.up(v_hnn_latent) 
            with torch.no_grad():
                v_hnn_high = proj.up(v_hnn_latent.detach())
            
            # 5. The true answer velocity vectors are extracted from each sample to filter Padding and misalignment
            cos_sim_total = 0.0
            valid_count = 0
            
            for i in range(inputs.size(0)):
                ans_indices = torch.where(labels[i] != -100)[0]
                if len(ans_indices) == 0: 
                    continue
                
                actual_positions = torch.cat([last_p_idx[i].unsqueeze(0), ans_indices])
                
                v_real_i = h_real[i, actual_positions[1:], :] - h_real[i, actual_positions[:-1], :]
                
                current_ans_len = len(ans_indices)
                v_hnn_high_i = v_hnn_high[i, :current_ans_len, :]
                
                cos_sim_i = torch.nn.functional.cosine_similarity(v_hnn_high_i, v_real_i, dim=-1)
                cos_sim_total += cos_sim_i.mean()
                valid_count += 1
                
            if valid_count > 0:
                loss_align = 1.0 - (cos_sim_total / valid_count)
            else:
                loss_align = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
            
            # A priori regularization term of physics knowledge
            loss_phys_reg = cfg.lambda_f * sum(p.abs().sum() for p in hnn_f.f_net.parameters()) + cfg.lambda_d * hnn_f.d1.abs().sum()
            
            total_loss = (loss_sft + cfg.lambda_align * loss_align + loss_phys_reg) / cfg.gradient_accumulation_steps
            total_loss.backward()
            
            epoch_sft += loss_sft.item()
            epoch_align += loss_align.item()
            epoch_phys += loss_phys_reg.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(hnn_f.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"SFT": f"{loss_sft.item():.3f}", "Align": f"{loss_align.item():.4f}"})

        num_steps = len(train_loader)
        history["loss_sft"].append(epoch_sft / num_steps)
        history["loss_align"].append(epoch_align / num_steps)
        history["loss_phys_reg"].append(epoch_phys / num_steps)

        save_dir = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        torch.save(hnn_f.state_dict(), os.path.join(save_dir, "hnn_func.pt"))
        print(f"\nSaved checkpoint for epoch {epoch+1} to {save_dir}")

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        epochs_range = range(1, len(history["loss_sft"]) + 1)

        axs[0].plot(epochs_range, history["loss_sft"], 'b-o', label='SFT Loss')
        axs[0].set_title('Cross Entropy (SFT)')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True); axs[0].legend()

        axs[1].plot(epochs_range, history["loss_align"], 'g-o', label='Align Loss')
        axs[1].set_title('Manifold Trajectory Alignment')
        axs[1].set_xlabel('Epoch')
        axs[1].grid(True); axs[1].legend()

        axs[2].plot(epochs_range, history["loss_phys_reg"], 'r-o', label='Phys Reg Loss')
        axs[2].set_title('Hamiltonian Regularization')
        axs[2].set_xlabel('Epoch')
        axs[2].grid(True); axs[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.save_path, f'stage3_loss_monitor_epoch_{epoch+1}.png'), dpi=300)
        plt.close()
        print(f"Loss plot updated and saved to {cfg.save_path}/stage3_loss_monitor_epoch_{epoch+1}.png")

if __name__ == "__main__":
    train_third_stage()
