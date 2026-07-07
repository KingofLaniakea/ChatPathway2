import argparse
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

"""FrameworkA PHNN prototype.

This file is copied from `framework_a.py` and changes only the latent dynamics
module from the existing forced/damped HNN-style field to a controlled
port-Hamiltonian field.
"""


PHNN_CONFIG_FIELDS = (
    "base_model_id",
    "sft_lora_path",
    "ae_ckpt_path",
    "data_path",
    "save_path",
    "batch_size",
    "gradient_accumulation_steps",
    "lr",
    "phnn_lr",
    "epochs",
    "max_length",
    "latent_dim",
    "control_dim",
    "control_source",
    "lambda_align",
    "lambda_g",
    "lambda_d",
    "lambda_j",
    "device",
)


def config_to_dict(cfg):
    return {field: getattr(cfg, field) for field in PHNN_CONFIG_FIELDS}


def parse_args():
    parser = argparse.ArgumentParser(description="Train FrameworkA PHNN prompt-control LoRA.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", dest="base_model_id", default=ThirdStagePHNNConfig.base_model_id)
    parser.add_argument("--sft-lora", dest="sft_lora_path", default=ThirdStagePHNNConfig.sft_lora_path)
    parser.add_argument("--ae-ckpt", dest="ae_ckpt_path", default=ThirdStagePHNNConfig.ae_ckpt_path)
    parser.add_argument("--train", dest="data_path", default=ThirdStagePHNNConfig.data_path)
    parser.add_argument("--save-dir", dest="save_path", default=ThirdStagePHNNConfig.save_path)
    parser.add_argument("--batch-size", type=int, default=ThirdStagePHNNConfig.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=ThirdStagePHNNConfig.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=ThirdStagePHNNConfig.lr)
    parser.add_argument("--phnn-lr", type=float, default=ThirdStagePHNNConfig.phnn_lr)
    parser.add_argument("--epochs", type=int, default=ThirdStagePHNNConfig.epochs)
    parser.add_argument("--max-length", type=int, default=ThirdStagePHNNConfig.max_length)
    parser.add_argument("--latent-dim", type=int, default=ThirdStagePHNNConfig.latent_dim)
    parser.add_argument("--control-dim", type=int, default=ThirdStagePHNNConfig.control_dim)
    parser.add_argument("--control-source", default=ThirdStagePHNNConfig.control_source)
    parser.add_argument("--lambda-align", type=float, default=ThirdStagePHNNConfig.lambda_align)
    parser.add_argument("--lambda-g", type=float, default=ThirdStagePHNNConfig.lambda_g)
    parser.add_argument("--lambda-d", type=float, default=ThirdStagePHNNConfig.lambda_d)
    parser.add_argument("--lambda-j", type=float, default=ThirdStagePHNNConfig.lambda_j)
    parser.add_argument("--device", default=ThirdStagePHNNConfig.device)
    args = parser.parse_args()
    cfg = ThirdStagePHNNConfig()
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def time_column(t, x):
    if torch.is_tensor(t):
        return t.to(device=x.device, dtype=x.dtype).reshape(1, 1).expand(x.size(0), 1)
    return torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)

class ThirdStagePHNNConfig:
    base_model_id = "/root/autodl-tmp/models/qwen3_8B"
    sft_lora_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
    # ae_ckpt_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_new_1/ae_epoch_4/ae_proj.pt"
    ae_ckpt_path = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"
    data_path = "/root/autodl-tmp/data/train_11_species_dataset.csv"
    save_path = "/root/autodl-tmp/checkpoints/qwen3_8b_FrameworkA_phnn_ae_cos"

    batch_size = 4
    gradient_accumulation_steps = 12
    lr = 1e-4
    phnn_lr = 2e-4
    epochs = 4
    max_length = 1072
    latent_dim = 128
    control_dim = 128
    control_source = "prompt_latent_mean"

    lambda_align = 0.5
    lambda_g = 1e-4
    lambda_d = 1e-3
    lambda_j = 1e-5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)

class TDPHNNFunc(nn.Module):
    """Time-dependent controlled port-Hamiltonian vector field.

    The control input ``u`` is set once per batch before calling ``odeint``. In
    this first PHNN variant, ``u`` is a prompt-level latent context vector:
    question/species/pathway/phenotype text -> Qwen hidden states -> frozen AE.
    """

    def __init__(self, latent_dim, control_dim=None, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.hamiltonian = nn.Sequential(
            nn.Linear(latent_dim + self.control_dim + 1, hidden_dim),
            Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, hidden_dim),
            Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.raw_J = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.raw_R_diag = nn.Parameter(torch.full((latent_dim,), -3.0))
        self.control_port = nn.Linear(self.control_dim, latent_dim, bias=False)
        self._control = None

    def set_control(self, control):
        self._control = control

    def clear_control(self):
        self._control = None

    def regularization_loss(self, lambda_g, lambda_d, lambda_j):
        damping = nn.functional.softplus(self.raw_R_diag).sum()
        control = self.control_port.weight.abs().sum()
        structure = self.raw_J.abs().sum()
        return lambda_g * control + lambda_d * damping + lambda_j * structure

    def forward(self, t, x):
        if self._control is None:
            raise RuntimeError("TDPHNNFunc control input must be set before odeint.")
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            u = self._control.to(device=x.device, dtype=x.dtype)
            if u.size(0) == 1 and x.size(0) != 1:
                u = u.expand(x.size(0), -1)
            if u.size(0) != x.size(0):
                raise RuntimeError(f"Control batch {u.size(0)} does not match state batch {x.size(0)}.")
            t_tensor = time_column(t, x)
            H = self.hamiltonian(torch.cat([x, u, t_tensor], dim=-1))
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            J = self.raw_J - self.raw_J.t()
            damping = nn.functional.softplus(self.raw_R_diag)
            conservative = dH @ J.t()
            dissipative = damping * dH
            control_drive = self.control_port(u)
            return conservative - dissipative + control_drive

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

def train_third_stage(cfg=None):
    cfg = cfg or parse_args()
    device = cfg.device
    os.makedirs(cfg.save_path, exist_ok=True)
    with open(os.path.join(cfg.save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2, ensure_ascii=False)
        f.write("\n")

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
    proj = CascadeProjection(high_dim=base_model.config.hidden_size, latent_dim=cfg.latent_dim).to(device).float()
    sd = torch.load(cfg.ae_ckpt_path, map_location=device)
    proj.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()})
    for p in proj.parameters(): p.requires_grad = False
    proj.eval()

    # ---- Optional resume point for PHNN weights ----
    # phnn_f_path = os.path.join(epoch_2_dir, "phnn_func.pt")
    # if os.path.exists(phnn_f_path):
    #     print(f"Loading resumed PHNN parameters from {phnn_f_path}...")
    #     phnn_f.load_state_dict(torch.load(phnn_f_path, map_location=device))

    phnn_f = TDPHNNFunc(cfg.latent_dim, cfg.control_dim).to(device).float()

    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': cfg.lr}, {'params': phnn_f.parameters(), 'lr': cfg.phnn_lr}])

    dataset = CSVPathwayDataset(cfg.data_path, tokenizer, cfg.max_length)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)


    # print("Resuming Joint SFT + PHNN Fine-Tuning from Epoch 3...")

    history = {"loss_sft": [], "loss_align": [], "loss_phys_reg": []}

    # print("Starting Joint SFT + PHNN Fine-Tuning on single GPU (Optimized Matrix Alignment)...")
    # ---- 【修改 4】：循环范围改为从第三轮到设定的总轮数 ----
    # for epoch in range(2, cfg.epochs):
    for epoch in range(cfg.epochs):
        model.train()
        phnn_f.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        epoch_sft, epoch_align, epoch_phys = 0.0, 0.0, 0.0

        for step, batch in enumerate(pbar):
            inputs, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels, output_hidden_states=True)
            loss_sft = outputs.loss

            h_real = outputs.hidden_states[-1]
            z_all, _ = proj(h_real.float())

            # 1. Pinpoint the token immediately before the answer span. Padding labels
            # are also -100, so counting -100 labels would choose a padding token.
            answer_mask = labels != -100
            answer_lengths = answer_mask.sum(dim=1)
            valid_samples = answer_lengths > 0

            if valid_samples.any():
                first_answer_idx = answer_mask.to(torch.int64).argmax(dim=1)
                last_p_idx = (first_answer_idx - 1).clamp(min=0)
                z0 = z_all[torch.arange(z_all.size(0), device=device), last_p_idx]
                prompt_mask = ((labels == -100) & (mask == 1)).unsqueeze(-1).to(z_all.dtype)
                prompt_counts = prompt_mask.sum(dim=1).clamp(min=1.0)
                u_context = (z_all * prompt_mask).sum(dim=1) / prompt_counts

                # 2. dynamic access to the current Batch of all internal samples, the true answer Token maximum length
                max_ans_len = answer_lengths.max().item()

                # The time step is only divided into Max + 1 steps
                t_steps = torch.linspace(0.0, 1.0, max_ans_len + 1).to(device)

                # 3. Use Ode to solve the PHNN latent trajectory conditioned on prompt context u.
                phnn_f.set_control(u_context.float())
                try:
                    z_traj = odeint(phnn_f, z0.float(), t_steps, method='rk4').transpose(0, 1)
                finally:
                    phnn_f.clear_control()

                # 4. The change velocity corresponding to the low-dimensional trajectory on the PHNN side is calculated and projected to the high-dimensional hidden space
                # v_phnn_high : [Batch, max_ans_len, High_dim]
                v_phnn_latent = z_traj[:, 1:, :] - z_traj[:, :-1, :]
                # Keep the AE frozen, but preserve this graph so alignment trains the PHNN.
                v_phnn_high = proj.up(v_phnn_latent)

                # 5. The true answer velocity vectors are extracted from each sample to filter Padding and misalignment
                cos_sim_total = 0.0
                valid_count = 0

                for i in range(inputs.size(0)):
                    ans_indices = torch.where(answer_mask[i])[0]
                    if len(ans_indices) == 0:
                        continue

                    actual_positions = torch.cat([last_p_idx[i].unsqueeze(0), ans_indices])

                    h_real_i = h_real[i].float()
                    v_real_i = h_real_i[actual_positions[1:], :] - h_real_i[actual_positions[:-1], :]

                    current_ans_len = len(ans_indices)
                    v_phnn_high_i = v_phnn_high[i, :current_ans_len, :]

                    cos_sim_i = torch.nn.functional.cosine_similarity(v_phnn_high_i, v_real_i, dim=-1)
                    cos_sim_total += cos_sim_i.mean()
                    valid_count += 1

                loss_align = 1.0 - (cos_sim_total / valid_count)
            else:
                loss_align = torch.tensor(0.0, device=device)

            # A priori regularization term of port-Hamiltonian structure.
            loss_phys_reg = phnn_f.regularization_loss(cfg.lambda_g, cfg.lambda_d, cfg.lambda_j)

            total_loss = (loss_sft + cfg.lambda_align * loss_align + loss_phys_reg) / cfg.gradient_accumulation_steps
            total_loss.backward()

            epoch_sft += loss_sft.item()
            epoch_align += loss_align.item()
            epoch_phys += loss_phys_reg.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(phnn_f.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"SFT": f"{loss_sft.item():.3f}", "Align": f"{loss_align.item():.4f}"})

        if len(train_loader) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(phnn_f.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        num_steps = len(train_loader)
        history["loss_sft"].append(epoch_sft / num_steps)
        history["loss_align"].append(epoch_align / num_steps)
        history["loss_phys_reg"].append(epoch_phys / num_steps)
        with open(os.path.join(cfg.save_path, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            f.write("\n")

        save_dir = os.path.join(cfg.save_path, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        torch.save(phnn_f.state_dict(), os.path.join(save_dir, "phnn_func.pt"))
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
        axs[2].set_title('Port-Hamiltonian Regularization')
        axs[2].set_xlabel('Epoch')
        axs[2].grid(True); axs[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.save_path, f'stage3_loss_monitor_epoch_{epoch+1}.png'), dpi=300)
        plt.close()
        print(f"Loss plot updated and saved to {cfg.save_path}/stage3_loss_monitor_epoch_{epoch+1}.png")

if __name__ == "__main__":
    train_third_stage()
