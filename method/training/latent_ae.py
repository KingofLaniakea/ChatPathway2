import os
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


# ================= 1. Configuration =================
class Config:
    base_model_id = "/root/autodl-tmp/models/qwen3_8B"
    sft_lora_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
    train_path = "/root/autodl-tmp/data/train_11_species_dataset.csv"
    save_path = "/root/autodl-tmp/checkpoints/qwen3_8b_ae_latent_128_cos"

    batch_size = 32     
    gradient_accumulation_steps = 2
    lr = 1e-4                  
    epochs = 5               
    max_length = 1072
    latent_dim = 128         
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ================= 2. AE =================
class CascadeProjection(nn.Module):
    def __init__(self, high_dim=4096, mid_dim=1024, latent_dim=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(high_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(), 
            nn.Linear(mid_dim, mid_dim // 2),
            nn.LayerNorm(mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, latent_dim)
        )
        self.up = nn.Sequential(
            nn.Linear(latent_dim, mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, high_dim)
        )
    def forward(self, x):
        latent = self.down(x)
        reconstructed = self.up(latent)
        return latent, reconstructed

# ================= 3. Data Processing =================
class AEDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1072):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.df = pd.read_csv(
            file_path, 
            engine='python',
            quoting=csv.QUOTE_MINIMAL, 
            on_bad_lines='skip'
        )
        print(f"\n[✓] Successfully loaded AE training dataset with {len(self.df)} clean samples.")
        
        print("="*40 + " [DATASET SELF-CHECK] " + "="*40)
        print("[*] Printing the 1st sample to verify structure:")
        first_row = self.df.iloc[0]
        q_preview = str(first_row['question']).replace('\n', ' [\\n] ') + "..."
        a_preview = str(first_row['answer']).replace('\n', ' [\\n] ') + "..."
        print(f" -> Columns in CSV: {list(self.df.columns)}")
        print(f" -> row[0]['question'] (Preview): {q_preview}")
        print(f" -> row[0]['answer']   (Preview): {a_preview}")
        print("="*102 + "\n")
        
        self.assistant_prefix_ids = self.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        question = str(row['question']) if not pd.isna(row['question']) else ""
        raw_answer = row.get('answer', row.get('formatted_answer_no_phenotype', ""))
        answer = str(raw_answer) if not pd.isna(raw_answer) else ""
        
        prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        
        inputs = self.tokenizer(prompt_text, max_length=self.max_length, truncation=True, padding=False, return_tensors=None)
        input_ids = inputs["input_ids"]
        
        loss_mask = [0] * len(input_ids)
        
        start_idx = 0
        n_prefix = len(self.assistant_prefix_ids)
        for i in range(len(input_ids) - n_prefix + 1):
            if input_ids[i:i+n_prefix] == self.assistant_prefix_ids:
                start_idx = i + n_prefix
                break
        
        if start_idx > 0:
            for i in range(start_idx, len(input_ids)):
                loss_mask[i] = 1
                
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long)
        }

def collate_fn(batch):
    pad_id = 151643
    input_ids = torch.nn.utils.rnn.pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=pad_id)
    loss_mask = torch.nn.utils.rnn.pad_sequence([b['loss_mask'] for b in batch], batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids, 
        "attention_mask": (input_ids != pad_id).long(),
        "loss_mask": loss_mask
    }


# ================= 4. Training =================
def train_ae():
    cfg = Config()
    device = torch.device(cfg.device)
    
    os.makedirs(cfg.save_path, exist_ok=True)
    print(f"[*] Training AE based on SFT model: {cfg.sft_lora_path}")
    print(f"[*] Running on device: {cfg.device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, 
        torch_dtype=torch.float16, 
        trust_remote_code=True, 
        device_map=None
    )
    
    model = PeftModel.from_pretrained(base_model, cfg.sft_lora_path)
    model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    ae_proj = CascadeProjection(high_dim=base_model.config.hidden_size, latent_dim=cfg.latent_dim).to(device)
    ae_proj = ae_proj.to(torch.float32)

    optimizer = optim.AdamW(ae_proj.parameters(), lr=cfg.lr)
    dataset = AEDataset(cfg.train_path, tokenizer, cfg.max_length)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    history = {"rec_mse": [], "rec_cos": []}
    
    for epoch in range(cfg.epochs):
        ae_proj.train()
        
        e_mse = 0.0
        e_cos = 0.0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            loss_mask = batch['loss_mask'].to(device) 

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                h_real = outputs.hidden_states[-1].detach().to(torch.float32) 
            
            z_latent, h_rec = ae_proj(h_real)
            
            active_mask = loss_mask.unsqueeze(-1).expand_as(h_real).bool()
            
            if active_mask.sum() == 0:
                continue
                
            # loss_rec = nn.functional.mse_loss(h_rec[active_mask], h_real[active_mask])
            h_real_active = h_real[active_mask]
            h_rec_active = h_rec[active_mask]

            # 1. 计算传统的隐空间 MSE 损失
            mse_loss = nn.functional.mse_loss(h_rec_active, h_real_active)
            
            # 2. 计算余弦相似度损失 (1 - cos_sim)，使其范围在 0 到 2 之间，越趋近 0 越好
            cos_sim = nn.functional.cosine_similarity(h_rec_active, h_real_active, dim=-1)
            cos_loss = 1.0 - cos_sim.mean()

            loss_rec = mse_loss + 2.0 * cos_loss




            (loss_rec / cfg.gradient_accumulation_steps).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(ae_proj.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            e_mse += mse_loss.item()
            e_cos += cos_loss.item()
            valid_batches += 1
            # pbar.set_postfix({"MSE": f"{loss_rec.item():.6f}"})
            pbar.set_postfix({"Loss": f"{loss_rec.item():.4f}", "MSE": f"{mse_loss.item():.4f}", "Cos": f"{cos_loss.item():.4f}"})

        if len(train_loader) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(ae_proj.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_mse = e_mse / max(valid_batches, 1)
        avg_cos = e_cos / max(valid_batches, 1)
        history["rec_mse"].append(avg_mse)
        history["rec_cos"].append(avg_cos)
        
        save_dir = os.path.join(cfg.save_path, f"ae_epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ae_proj.state_dict(), os.path.join(save_dir, "ae_proj.pt"))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # 左图：MSE 损失监控
        axes[0].plot(history["rec_mse"], 'b-o', markersize=4, label='Reconstruction MSE')
        axes[0].set_title('Manifold Stability (MSE)', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()
        
        # 右图：Cosine 损失监控（反映方向对齐度）
        axes[1].plot(history["rec_cos"], 'g-s', markersize=4, label='Directional Cosine Loss')
        axes[1].set_title('Trajectory Angle Alignment (1-Cos)', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Cosine Loss')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.save_path, f'ae_monitor_epoch_{epoch+1}.png'), dpi=300) 
        plt.close()

        # avg_rec = e_rec / len(train_loader)
        # history["rec_mse"].append(avg_rec)
        
        # save_dir = os.path.join(cfg.save_path, f"ae_epoch_{epoch+1}")
        # os.makedirs(save_dir, exist_ok=True)
        # torch.save(ae_proj.state_dict(), os.path.join(save_dir, "ae_proj.pt"))

        # plt.figure(figsize=(6, 4))
        # plt.plot(history["rec_mse"], 'b-', label='Reconstruction MSE')
        # plt.title('Manifold Stability (SFT-Frozen)')
        # plt.xlabel('Epoch')
        # plt.ylabel('MSE')
        # plt.grid(True)
        # plt.savefig(os.path.join(cfg.save_path, f'ae_monitor_epoch_{epoch+1}.png'))
        # plt.close()

if __name__ == "__main__":
    train_ae()
