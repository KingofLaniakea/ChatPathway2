import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn.parallel import DistributedDataParallel as DDP

# =========================================================
# 0. ENV
# =========================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Blackwell + NCCL 稳定性
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# =========================================================
# 1. Config
# =========================================================

class Config:

    base_model_id = "/root/autodl-tmp/qwen3_8b_stage3_FULL_MODEL"

    train_dataset_path = \
        "/root/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"

    save_path = \
        "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft"

    batch_size = 2

    gradient_accumulation_steps = 8

    lr = 2e-5

    epochs = 3

    max_length = 1648

    top_k_genes = 200

    perturbation_col = "target_gene"

    control_label = "non-targeting"

    num_workers = 4

    lora_config = {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }


# =========================================================
# 2. Dataset
# =========================================================

class C2SPairingDataset(Dataset):

    def __init__(self, cfg, tokenizer):

        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.top_k_genes = cfg.top_k_genes

        raw_dataset = load_from_disk(cfg.train_dataset_path)

        self.raw_dataset = raw_dataset

        self.prompt_template = """
Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.

Control cell sentence:
{control_cell_sentence}

Perturbed cell sentence:
"""

        self.answer_template = "{perturbed_cell_sentence}"

        self.control_indices = []
        self.pert_pairs = []

        for i, sample in enumerate(raw_dataset):

            if sample[cfg.perturbation_col] == cfg.control_label:
                self.control_indices.append(i)
            else:
                self.pert_pairs.append(
                    (sample[cfg.perturbation_col], i)
                )

        assert len(self.control_indices) > 0

    def _get_cell_sentence_str(self, sample):

        raw_sentence = str(sample.get("cell_sentence", ""))

        words = raw_sentence.split()

        selected_words = words[: self.top_k_genes]

        return " ".join(selected_words), str(len(selected_words))

    def __len__(self):

        return len(self.pert_pairs)

    def __getitem__(self, idx):

        pert_name, perturbed_idx = self.pert_pairs[idx]

        perturbed_sample = self.raw_dataset[perturbed_idx]

        # 不要 random.seed(idx)
        control_idx = random.choice(self.control_indices)

        control_sample = self.raw_dataset[control_idx]

        control_sentence, num_genes_str = \
            self._get_cell_sentence_str(control_sample)

        perturbed_sentence, _ = \
            self._get_cell_sentence_str(perturbed_sample)

        prompt_text = self.prompt_template.format(
            num_genes=num_genes_str,
            perturbation_name=pert_name,
            control_cell_sentence=control_sentence,
        )

        answer_text = self.answer_template.format(
            perturbed_cell_sentence=perturbed_sentence
        )

        full_text = (
            "<|im_start|>user\n"
            + prompt_text
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
            + answer_text
            + "<|im_end|>"
        )

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]

        assistant_start = full_text.find(
            "<|im_start|>assistant\n"
        )

        assistant_prefix = full_text[:assistant_start]

        prompt_ids = self.tokenizer.encode(
            assistant_prefix,
            add_special_tokens=False
        )

        labels = [-100] * len(prompt_ids)

        labels += input_ids[len(prompt_ids):]

        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long
            ),
        }


# =========================================================
# 3. Collate
# =========================================================

def collate_fn(batch, pad_token_id):

    input_ids = [x["input_ids"] for x in batch]

    labels = [x["labels"] for x in batch]

    input_ids = nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )

    labels = nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    attention_mask = (input_ids != pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# =========================================================
# 4. DDP setup
# =========================================================

def setup():

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup():

    dist.destroy_process_group()


# =========================================================
# 5. Train
# =========================================================

def train():

    local_rank = setup()

    device = torch.device(f"cuda:{local_rank}")

    cfg = Config()

    if local_rank == 0:
        os.makedirs(cfg.save_path, exist_ok=True)

    # =====================================================
    # tokenizer
    # =====================================================

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # model
    # =====================================================

    if local_rank == 0:
        print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,

        dtype=torch.bfloat16,

        trust_remote_code=True,

        attn_implementation="eager",
    )

    # VERY IMPORTANT
    model.config.use_cache = False

    # =====================================================
    # LoRA
    # =====================================================

    lora_config = LoraConfig(**cfg.lora_config)

    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()

    if local_rank == 0:
        model.print_trainable_parameters()

    # =====================================================
    # device
    # =====================================================

    model = model.to(device)

    torch.cuda.empty_cache()

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # =====================================================
    # dataset
    # =====================================================

    dataset = C2SPairingDataset(cfg, tokenizer)

    sampler = DistributedSampler(
        dataset,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(
            x,
            tokenizer.pad_token_id
        ),
    )

    # =====================================================
    # optimizer
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr
    )

    loss_history = []

    # =====================================================
    # training
    # =====================================================

    for epoch in range(cfg.epochs):

        sampler.set_epoch(epoch)

        model.train()

        total_loss = 0

        if local_rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        else:
            pbar = loader

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            loss = loss / cfg.gradient_accumulation_steps

            loss.backward()

            if (
                (step + 1)
                % cfg.gradient_accumulation_steps
                == 0
            ) or ((step + 1) == len(loader)):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

                optimizer.zero_grad()

            total_loss += (
                loss.item()
                * cfg.gradient_accumulation_steps
            )

            if local_rank == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        # =================================================
        # save
        # =================================================

        if local_rank == 0:

            avg_loss = total_loss / len(loader)

            loss_history.append(avg_loss)

            print(
                f"Epoch {epoch+1} | "
                f"avg loss = {avg_loss:.4f}"
            )

            save_dir = os.path.join(
                cfg.save_path,
                f"checkpoint_epoch_{epoch+1}"
            )

            model.module.save_pretrained(save_dir)

            tokenizer.save_pretrained(save_dir)

            print(f"Saved to {save_dir}")

            plt.figure(figsize=(8, 5))

            plt.plot(loss_history)

            plt.xlabel("Epoch")

            plt.ylabel("Loss")

            plt.grid(True)

            plt.savefig(
                os.path.join(
                    cfg.save_path,
                    "loss_curve.png"
                )
            )

            plt.close()

    cleanup()


# =========================================================
# main
# =========================================================

if __name__ == "__main__":

    train()