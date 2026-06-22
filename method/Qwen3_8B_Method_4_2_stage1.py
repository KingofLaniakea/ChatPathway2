import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdiffeq import odeint

class HNNTrainConfig:
    base_model_id = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/.cache/hf_models/models--Qwen--Qwen3-8B"
    sft_lora_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_sft/checkpoint_epoch_5"
    ae_ckpt_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_ae_only/checkpoint_epoch_5/ae_proj.pt"
    data_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/datasets/train_11_species_dataset.csv"
    save_path = "/gpfs/projects/AI4D/core-132/neo_desktop/yize/checkpoints/qwen3_8b_stage1_hnn_only"
    
    batch_size = 4
    lr = 1e-4
    epochs = 10
    latent_dim = 128
    max_length = 1072
    
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

def train_stage1():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    cfg = HNNTrainConfig()
    
    if local_rank == 0: os.makedirs(cfg.save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.bfloat16, device_map={"": local_rank})
    model = PeftModel.from_pretrained(base_model, cfg.sft_lora_path).eval()
    for p in model.parameters(): p.requires_grad = False

    proj = CascadeProjection(high_dim=base_model.config.hidden_size, latent_dim=cfg.latent_dim).to(local_rank).bfloat16().eval()
    proj.load_state_dict(torch.load(cfg.ae_ckpt_path, map_location=f"cuda:{local_rank}"))
    for p in proj.parameters(): p.requires_grad = False

    hnn_f = TDHNNFunc(cfg.latent_dim).to(local_rank).bfloat16()
    hnn_f = DDP(hnn_f, device_ids=[local_rank])
    
    optimizer = optim.AdamW(hnn_f.parameters(), lr=cfg.lr)
    dataset = CSVPathwayDataset(cfg.data_path, tokenizer, cfg.max_length)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=DistributedSampler(dataset), collate_fn=collate_fn)

    history = {"total": [], "mse": [], "reg": []}

    for epoch in range(cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        hnn_f.train()
        
        e_total, e_mse, e_reg = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        
        for step, batch in enumerate(pbar):
            inputs = batch['input_ids'].to(local_rank)
            mask = batch['attention_mask'].to(local_rank)
            
            with torch.no_grad():
                outputs = model(input_ids=inputs, attention_mask=mask, output_hidden_states=True)
                h_real = outputs.hidden_states[-1]
                z_gt, _ = proj(h_real) 

            z0 = z_gt[:, 0, :]
            t_steps = torch.linspace(0.0, 1.0, z_gt.size(1)).to(local_rank)
            hnn_f.module.float()
            z_pred = odeint(hnn_f.module, z0.float(), t_steps, method='rk4').transpose(0, 1).to(torch.bfloat16)
            hnn_f.module.bfloat16()

            loss_mse = nn.functional.mse_loss(z_pred * mask.unsqueeze(-1), z_gt * mask.unsqueeze(-1))
            
            loss_f = sum(p.abs().sum() for p in hnn_f.module.f_net.parameters())
            loss_d = hnn_f.module.d1.abs().sum()
            loss_phys_reg = cfg.lambda_f * loss_f + cfg.lambda_d * loss_d
            
            total_loss = (loss_mse + loss_phys_reg) / cfg.gradient_accumulation_steps
            total_loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(hnn_f.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                e_total += total_loss.item() * cfg.gradient_accumulation_steps
                e_mse += loss_mse.item()
                e_reg += loss_phys_reg.item()
                
                if local_rank == 0:
                    pbar.set_postfix({
                        "MSE": f"{loss_mse.item():.5f}",
                        "Reg": f"{loss_phys_reg.item():.2e}"
                    })

        if local_rank == 0:
            avg_factor = len(train_loader) // cfg.gradient_accumulation_steps
            history["total"].append(e_total / avg_factor)
            history["mse"].append(e_mse / avg_factor)
            history["reg"].append(e_reg / avg_factor)
            
            save_dir = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(hnn_f.module.state_dict(), os.path.join(save_dir, "hnn_func.pt"))
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history["mse"], marker='o', color='royalblue')
            plt.title("Trajectory Alignment Loss (MSE)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.subplot(1, 2, 2)
            plt.plot(history["reg"], marker='s', color='forestgreen')
            plt.title("Physics Regularization Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "hnn_training_metrics.png"))
            plt.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    train_stage1()