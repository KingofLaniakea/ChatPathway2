"""Hamiltonian latent dynamics used by the controlled ChatPathway ablations.

The maintained dynamics do not assume that the first half of an arbitrary AE
latent has position semantics and the second half has momentum semantics.
Instead, a constant trainable skew-symmetric Poisson matrix ``J`` defines the
conservative flow in the full latent space.  Because ``J`` is constant and
skew-symmetric, ``grad(H)^T J grad(H)`` is zero by construction.

Two variants are intentionally supported:

``hnn``
    dz/dt = J grad H(z)

``forced_damped_hnn``
    dz/dt = (J - R) grad H(z) + F(t)

``R`` is positive semidefinite by construction and ``F`` is an explicit
time-only force.  This is the full-latent analogue of the forced/damped
port-Hamiltonian form; there is no prompt/control input ``u`` in these two
ablations.  The upstream pathway affects the rollout through its initial state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = ("hnn", "forced_damped_hnn")
STRUCTURE_MODES = ("orthogonal_poisson", "learned_skew", "canonical")
DAMPING_MODES = ("isotropic", "diagonal")


class Sin(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)


def canonical_structure(latent_dim: int, *, device: Any = None, dtype: Any = None) -> torch.Tensor:
    """Return the canonical full-rank skew matrix for an even dimension."""

    if latent_dim % 2:
        raise ValueError("canonical Hamiltonian structure requires an even latent_dim")
    half = latent_dim // 2
    matrix = torch.zeros(latent_dim, latent_dim, device=device, dtype=dtype)
    identity = torch.eye(half, device=device, dtype=dtype)
    matrix[:half, half:] = identity
    matrix[half:, :half] = -identity
    return matrix


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class HamiltonianConfig:
    latent_dim: int
    variant: str = "hnn"
    hidden_dim: int = 256
    structure_mode: str = "orthogonal_poisson"
    initial_damping: float = 1e-2
    structure_reflections: int = 16
    damping_mode: str = "isotropic"


class LatentHamiltonianDynamics(nn.Module):
    """Energy-based latent vector field with exact structural constraints."""

    def __init__(
        self,
        latent_dim: int,
        *,
        variant: str = "hnn",
        hidden_dim: int = 256,
        structure_mode: str = "orthogonal_poisson",
        initial_damping: float = 1e-2,
        structure_reflections: int = 16,
        damping_mode: str = "isotropic",
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown Hamiltonian variant: {variant}")
        if structure_mode not in STRUCTURE_MODES:
            raise ValueError(f"unknown structure mode: {structure_mode}")
        if latent_dim % 2:
            raise ValueError("ChatPathway Hamiltonian variants require an even latent_dim")
        if structure_reflections < 1:
            raise ValueError("structure_reflections must be positive")
        if damping_mode not in DAMPING_MODES:
            raise ValueError(f"unknown damping mode: {damping_mode}")

        self.config = HamiltonianConfig(
            latent_dim=latent_dim,
            variant=variant,
            hidden_dim=hidden_dim,
            structure_mode=structure_mode,
            initial_damping=initial_damping,
            structure_reflections=structure_reflections,
            damping_mode=damping_mode,
        )
        self.latent_dim = latent_dim
        self.variant = variant
        self.structure_mode = structure_mode
        self.damping_mode = damping_mode
        self.energy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        if structure_mode == "orthogonal_poisson":
            self.register_buffer("canonical_poisson", canonical_structure(latent_dim))
            self.raw_reflections = nn.Parameter(
                torch.randn(structure_reflections, latent_dim)
            )
            with torch.no_grad():
                initial_structure = self._orthogonal_structure().detach().clone()
            self.register_buffer("initial_structure", initial_structure)
        elif structure_mode == "learned_skew":
            # Rotate the canonical form into a dense reproducible basis (the
            # caller seeds torch before construction). No coordinate half is
            # assigned position or momentum semantics, while the initial J is
            # still full-rank and has the correct skew structure.
            basis, _ = torch.linalg.qr(torch.randn(latent_dim, latent_dim))
            initial_structure = basis @ canonical_structure(latent_dim) @ basis.transpose(0, 1)
            self.register_buffer("initial_structure", initial_structure)
            self.raw_structure = nn.Parameter(0.5 * initial_structure.clone())
        else:
            initial_structure = canonical_structure(latent_dim)
            self.register_buffer("initial_structure", initial_structure)
            self.register_buffer("fixed_structure", initial_structure.clone())

        if variant == "forced_damped_hnn":
            raw_value = inverse_softplus(initial_damping)
            damping_size = 1 if damping_mode == "isotropic" else latent_dim
            self.raw_damping = nn.Parameter(torch.full((damping_size,), raw_value))
            self.force_net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                Sin(),
                nn.Linear(hidden_dim, latent_dim, bias=False),
            )
            # Start from the conservative system. Force/damping must earn their
            # contribution through the alignment objective.
            nn.init.zeros_(self.force_net[-1].weight)

    def export_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def structure_matrix(self) -> torch.Tensor:
        if self.structure_mode == "canonical":
            return self.fixed_structure
        if self.structure_mode == "orthogonal_poisson":
            return self._orthogonal_structure()
        return self.raw_structure - self.raw_structure.transpose(0, 1)

    def _orthogonal_basis(self) -> torch.Tensor:
        basis = torch.eye(
            self.latent_dim,
            device=self.raw_reflections.device,
            dtype=self.raw_reflections.dtype,
        )
        for vector in self.raw_reflections:
            unit = vector / vector.norm().clamp(min=1e-8)
            basis = basis - 2.0 * unit.unsqueeze(1) @ (unit.unsqueeze(0) @ basis)
        return basis

    def _orthogonal_structure(self) -> torch.Tensor:
        basis = self._orthogonal_basis()
        return basis.transpose(0, 1) @ self.canonical_poisson @ basis

    def damping_diagonal(self) -> torch.Tensor:
        if self.variant == "hnn":
            return torch.zeros(
                self.latent_dim,
                device=self.initial_structure.device,
                dtype=self.initial_structure.dtype,
            )
        damping = F.softplus(self.raw_damping)
        return damping.expand(self.latent_dim) if damping.numel() == 1 else damping

    @staticmethod
    def time_column(t: torch.Tensor | float, z: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(t):
            value = t.to(device=z.device, dtype=z.dtype).reshape(-1)
            scalar = value[0] if value.numel() else torch.zeros((), device=z.device, dtype=z.dtype)
        else:
            scalar = torch.as_tensor(float(t), device=z.device, dtype=z.dtype)
        return scalar.reshape(1, 1).expand(z.size(0), 1)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        return self.energy_net(z)

    def vector_field_components(
        self,
        t: torch.Tensor | float,
        z: torch.Tensor,
        *,
        create_graph: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if create_graph is None:
            create_graph = self.training
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.requires_grad_(True)
            energy = self.hamiltonian(z)
            gradient = torch.autograd.grad(
                energy.sum(),
                z,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            structure = self.structure_matrix().to(device=z.device, dtype=z.dtype)
            conservative = gradient @ structure.transpose(0, 1)
            if self.variant == "hnn":
                dissipative = torch.zeros_like(conservative)
                force = torch.zeros_like(conservative)
            else:
                damping = self.damping_diagonal().to(device=z.device, dtype=z.dtype)
                dissipative = -gradient * damping
                force = self.force_net(self.time_column(t, z))
            return {
                "energy": energy,
                "gradient": gradient,
                "conservative": conservative,
                "dissipative": dissipative,
                "force": force,
            }

    def forward(self, t: torch.Tensor | float, z: torch.Tensor) -> torch.Tensor:
        parts = self.vector_field_components(t, z)
        return parts["conservative"] + parts["dissipative"] + parts["force"]

    def regularization_loss(
        self,
        t: torch.Tensor | float,
        z: torch.Tensor,
        *,
        lambda_structure: float,
        lambda_force: float,
        lambda_damping: float,
    ) -> torch.Tensor:
        structure = self.structure_matrix()
        if self.structure_mode == "orthogonal_poisson":
            identity = torch.eye(
                self.latent_dim,
                device=structure.device,
                dtype=structure.dtype,
            )
            constraint_error = (structure.transpose(0, 1) @ structure - identity).pow(2).mean()
            constraint_error = constraint_error + (structure @ structure + identity).pow(2).mean()
            loss = lambda_structure * constraint_error
        else:
            loss = lambda_structure * (structure - self.initial_structure).pow(2).mean()
        if self.variant == "forced_damped_hnn":
            if torch.is_tensor(t) and t.numel() > 1:
                force_times = t.to(device=z.device, dtype=z.dtype).reshape(-1, 1)
                force = self.force_net(force_times)
            else:
                force = self.force_net(self.time_column(t, z))
            loss = loss + lambda_force * force.abs().mean()
            loss = loss + lambda_damping * self.damping_diagonal().mean()
        return loss

    def energy_rate_terms(self, t: torch.Tensor | float, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dH/dt contributions for diagnostics and structural tests."""

        parts = self.vector_field_components(t, z, create_graph=False)
        gradient = parts["gradient"]
        return {
            "conservative_power": (gradient * parts["conservative"]).sum(dim=-1),
            "dissipative_power": (gradient * parts["dissipative"]).sum(dim=-1),
            "force_power": (gradient * parts["force"]).sum(dim=-1),
        }
