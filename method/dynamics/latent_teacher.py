"""Reusable latent dynamics teachers for ChatPathway2 experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _time_column(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(t):
        return t.to(device=z.device, dtype=z.dtype).reshape(1, 1).expand(z.size(0), 1)
    return torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)


class CascadeProjection(nn.Module):
    def __init__(self, high_dim: int = 4096, mid_dim: int = 1024, latent_dim: int = 128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(high_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.LayerNorm(mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, latent_dim),
        )
        self.up = nn.Sequential(
            nn.Linear(latent_dim, mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, high_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.down(x)
        return latent, self.up(latent)


class ControlledNeuralODEFunc(nn.Module):
    def __init__(self, latent_dim: int, control_dim: int | None = None, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + self.control_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._control: torch.Tensor | None = None

    def set_control(self, control: torch.Tensor) -> None:
        self._control = control

    def clear_control(self) -> None:
        self._control = None

    def regularization_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._control is None:
            raise RuntimeError("Control must be set before ODE rollout.")
        u = self._control.to(device=z.device, dtype=z.dtype)
        if u.size(0) == 1 and z.size(0) != 1:
            u = u.expand(z.size(0), -1)
        t_column = _time_column(t, z)
        return self.net(torch.cat([z, u, t_column], dim=-1))


class EncoderConditionedLatentODEFunc(nn.Module):
    """Latent ODE whose initial state is inferred from prompt-level control.

    The ordinary controlled Neural ODE teacher starts from the gold answer
    trajectory's first latent state. This variant is stricter: it predicts z0
    from the prompt/control vector, then integrates dz/dt=f(z,u,t).
    """

    def __init__(self, latent_dim: int, control_dim: int | None = None, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.init_net = nn.Sequential(
            nn.Linear(self.control_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.field_net = nn.Sequential(
            nn.Linear(latent_dim + self.control_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._control: torch.Tensor | None = None

    def initial_state(self, control: torch.Tensor) -> torch.Tensor:
        return self.init_net(control)

    def set_control(self, control: torch.Tensor) -> None:
        self._control = control

    def clear_control(self) -> None:
        self._control = None

    def regularization_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._control is None:
            raise RuntimeError("Control must be set before ODE rollout.")
        u = self._control.to(device=z.device, dtype=z.dtype)
        if u.size(0) == 1 and z.size(0) != 1:
            u = u.expand(z.size(0), -1)
        t_column = _time_column(t, z)
        return self.field_net(torch.cat([z, u, t_column], dim=-1))


class ControlledGradientFlowFunc(nn.Module):
    def __init__(self, latent_dim: int, control_dim: int | None = None, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.energy = nn.Sequential(
            nn.Linear(latent_dim + self.control_dim + 1, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.raw_damping = nn.Parameter(torch.full((latent_dim,), -2.0))
        self.control_port = nn.Linear(self.control_dim, latent_dim, bias=False)
        self._control: torch.Tensor | None = None

    def set_control(self, control: torch.Tensor) -> None:
        self._control = control

    def clear_control(self) -> None:
        self._control = None

    def regularization_loss(self) -> torch.Tensor:
        damping = nn.functional.softplus(self.raw_damping).sum()
        control = self.control_port.weight.abs().sum()
        return 1e-3 * damping + 1e-4 * control

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._control is None:
            raise RuntimeError("Control must be set before ODE rollout.")
        with torch.set_grad_enabled(True):
            z = z.requires_grad_(True)
            u = self._control.to(device=z.device, dtype=z.dtype)
            if u.size(0) == 1 and z.size(0) != 1:
                u = u.expand(z.size(0), -1)
            t_column = _time_column(t, z)
            energy = self.energy(torch.cat([z, u, t_column], dim=-1))
            grad_e = torch.autograd.grad(energy.sum(), z, create_graph=True)[0]
            damping = nn.functional.softplus(self.raw_damping)
            return -damping * grad_e + self.control_port(u)


class ControlledGENERICFunc(nn.Module):
    """Controlled GENERIC-style reversible + irreversible dynamics.

    This prototype uses a learned skew matrix for the reversible part and a
    positive diagonal mobility for the irreversible entropy-gradient part:

        dz/dt = J grad E(z,u,t) + M grad S(z,u,t) + G u
    """

    def __init__(self, latent_dim: int, control_dim: int | None = None, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.energy = nn.Sequential(
            nn.Linear(latent_dim + self.control_dim + 1, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.entropy = nn.Sequential(
            nn.Linear(latent_dim + self.control_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.raw_J = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.raw_M_diag = nn.Parameter(torch.full((latent_dim,), -2.0))
        self.control_port = nn.Linear(self.control_dim, latent_dim, bias=False)
        self._control: torch.Tensor | None = None

    def set_control(self, control: torch.Tensor) -> None:
        self._control = control

    def clear_control(self) -> None:
        self._control = None

    def regularization_loss(self) -> torch.Tensor:
        mobility = nn.functional.softplus(self.raw_M_diag).sum()
        structure = self.raw_J.abs().sum()
        control = self.control_port.weight.abs().sum()
        return 1e-3 * mobility + 1e-5 * structure + 1e-4 * control

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._control is None:
            raise RuntimeError("Control must be set before ODE rollout.")
        with torch.set_grad_enabled(True):
            z = z.requires_grad_(True)
            u = self._control.to(device=z.device, dtype=z.dtype)
            if u.size(0) == 1 and z.size(0) != 1:
                u = u.expand(z.size(0), -1)
            t_column = _time_column(t, z)
            inputs = torch.cat([z, u, t_column], dim=-1)
            energy = self.energy(inputs)
            entropy = self.entropy(inputs)
            grad_e = torch.autograd.grad(energy.sum(), z, create_graph=True)[0]
            grad_s = torch.autograd.grad(entropy.sum(), z, create_graph=True)[0]
            J = self.raw_J - self.raw_J.t()
            mobility = nn.functional.softplus(self.raw_M_diag)
            reversible = grad_e @ J.t()
            irreversible = mobility * grad_s
            return reversible + irreversible + self.control_port(u)


class ControlledKoopmanDynamics(nn.Module):
    def __init__(self, latent_dim: int, control_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.raw_A = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))
        self.control_port = nn.Linear(self.control_dim, latent_dim, bias=False)

    def stable_A(self) -> torch.Tensor:
        # Keep the first prototype numerically mild without a hard spectral solve.
        return 0.99 * torch.tanh(self.raw_A)

    def regularization_loss(self) -> torch.Tensor:
        identity = torch.eye(self.latent_dim, device=self.raw_A.device, dtype=self.raw_A.dtype)
        return 1e-4 * (self.control_port.weight.abs().sum() + (self.stable_A() - identity).pow(2).mean())

    def rollout(self, z0: torch.Tensor, control: torch.Tensor, steps: int) -> torch.Tensor:
        states = [z0]
        z = z0
        A = self.stable_A()
        u_drive = self.control_port(control.to(device=z0.device, dtype=z0.dtype))
        for _ in range(steps):
            z = z @ A.t() + u_drive
            states.append(z)
        return torch.stack(states, dim=1)


class ControlledSINDyFunc(nn.Module):
    """Sparse-library latent dynamics inspired by SINDy.

    The library is deliberately compact for 128-dimensional latents:
    [1, t, z, z^2, sin(z), u, z*u]. L1 regularization on the linear map encourages
    sparse use of candidate terms.
    """

    def __init__(self, latent_dim: int, control_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim or latent_dim
        self.library_dim = 2 + latent_dim * 4 + self.control_dim
        self.coefficients = nn.Linear(self.library_dim, latent_dim, bias=False)
        self._control: torch.Tensor | None = None

    def set_control(self, control: torch.Tensor) -> None:
        self._control = control

    def clear_control(self) -> None:
        self._control = None

    def library(self, t: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        t_column = _time_column(t, z)
        ones = torch.ones_like(t_column)
        if u.size(1) == z.size(1):
            interaction = z * u
        else:
            interaction = z * u[:, : z.size(1)]
        return torch.cat([ones, t_column, z, z.pow(2), torch.sin(z), u, interaction], dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        return 1e-3 * self.coefficients.weight.abs().mean()

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._control is None:
            raise RuntimeError("Control must be set before ODE rollout.")
        u = self._control.to(device=z.device, dtype=z.dtype)
        if u.size(0) == 1 and z.size(0) != 1:
            u = u.expand(z.size(0), -1)
        return self.coefficients(self.library(t, z, u))


def build_dynamics(variant: str, latent_dim: int, control_dim: int | None = None) -> nn.Module:
    if variant == "neural_ode":
        return ControlledNeuralODEFunc(latent_dim, control_dim)
    if variant == "latent_ode":
        return EncoderConditionedLatentODEFunc(latent_dim, control_dim)
    if variant == "gradient_flow":
        return ControlledGradientFlowFunc(latent_dim, control_dim)
    if variant == "generic":
        return ControlledGENERICFunc(latent_dim, control_dim)
    if variant == "koopman":
        return ControlledKoopmanDynamics(latent_dim, control_dim)
    if variant == "sindy":
        return ControlledSINDyFunc(latent_dim, control_dim)
    raise ValueError(f"Unknown latent dynamics variant: {variant}")


def load_projection(checkpoint: str, high_dim: int, latent_dim: int, device: str) -> CascadeProjection:
    projection = CascadeProjection(high_dim=high_dim, latent_dim=latent_dim).to(device).float()
    raw = torch.load(checkpoint, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in raw.items()}
    projection.load_state_dict(state)
    projection.eval()
    for parameter in projection.parameters():
        parameter.requires_grad = False
    return projection


def load_backbone(base_model: str, adapter: str | None, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    if adapter:
        backbone = PeftModel.from_pretrained(backbone, adapter)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return tokenizer, backbone


@dataclass
class LatentTrajectory:
    sample_id: int
    z: torch.Tensor
    control: torch.Tensor
    question: str
    target_text: str


def read_records(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    if limit is not None:
        df = df.head(limit)
    return df.to_dict(orient="records")


def prompt_for(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def extract_latent_trajectories(
    records: list[dict[str, Any]],
    tokenizer: Any,
    backbone: Any,
    projection: CascadeProjection,
    device: str,
    text_column: str,
    max_length: int,
    max_steps: int,
    start_sample_id: int = 0,
) -> list[LatentTrajectory]:
    full_texts: list[str] = []
    prefix_lengths: list[int] = []
    questions: list[str] = []
    targets: list[str] = []
    for record in records:
        question = str(record.get("question", ""))
        target = str(record.get(text_column, record.get("answer", "")))
        prompt = prompt_for(question)
        questions.append(question)
        targets.append(target)
        prefix_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
        full_texts.append(prompt + target + "<|im_end|>")

    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        hidden = backbone(**encoded, output_hidden_states=True).hidden_states[-1].float()
        z_all, _ = projection(hidden)

    trajectories: list[LatentTrajectory] = []
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    for i, seq_len_raw in enumerate(lengths):
        seq_len = int(seq_len_raw)
        if seq_len < 2:
            continue
        prefix_len = min(prefix_lengths[i], seq_len - 1)
        start = max(prefix_len - 1, 0)
        end = min(seq_len, start + max_steps + 1)
        if end - start < 2:
            continue
        prompt_end = max(prefix_len, 1)
        control = z_all[i, :prompt_end].mean(dim=0)
        trajectories.append(
            LatentTrajectory(
                sample_id=start_sample_id + i,
                z=z_all[i, start:end].float(),
                control=control.float(),
                question=questions[i],
                target_text=targets[i],
            )
        )
    return trajectories


def rollout(model: nn.Module, variant: str, z0: torch.Tensor, control: torch.Tensor, steps: int) -> torch.Tensor:
    if variant == "koopman":
        return model.rollout(z0.unsqueeze(0), control.unsqueeze(0), steps).squeeze(0)
    from torchdiffeq import odeint

    control_batch = control.unsqueeze(0)
    model.set_control(control_batch)
    try:
        if variant == "latent_ode":
            z0 = model.initial_state(control_batch).squeeze(0)
        t_steps = torch.linspace(0.0, 1.0, steps + 1, device=z0.device, dtype=z0.dtype)
        return odeint(model, z0.unsqueeze(0), t_steps, method="rk4").squeeze(1)
    finally:
        model.clear_control()


def trajectory_losses(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    min_len = min(predicted.size(0), target.size(0))
    predicted = predicted[:min_len]
    target = target[:min_len]
    rollout_loss = nn.functional.smooth_l1_loss(predicted, target)
    pred_velocity = predicted[1:] - predicted[:-1]
    target_velocity = target[1:] - target[:-1]
    velocity_loss = nn.functional.smooth_l1_loss(pred_velocity, target_velocity)
    cosine = nn.functional.cosine_similarity(predicted[1:], target[1:], dim=-1).mean()
    return {"rollout": rollout_loss, "velocity": velocity_loss, "cosine": cosine}


def checkpoint_payload(model: nn.Module, variant: str, config: Any) -> dict[str, Any]:
    return {
        "variant": variant,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }
