"""Loss-routing and multiscale trajectory helpers for staged dynamics training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch


DYNAMICS_RESOLUTIONS = ("graph_layer", "substep_multiscale")


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value unchanged while scaling gradients to its source.

    When applied to the LoRA-derived hidden state before an HNN rollout, this
    lets the dynamics parameters receive the full dynamics gradient while the
    gradient routed back into LoRA follows a separate warm-up coefficient.
    """

    if not 0.0 <= scale <= 1.0:
        raise ValueError("gradient scale must be in [0, 1]")
    return value.detach() + float(scale) * (value - value.detach())


def linear_warmup_scale(
    optimizer_step: int,
    total_optimizer_steps: int,
    warmup_fraction: float,
) -> float:
    """Return a zero-to-one dynamics-to-LoRA gradient schedule."""

    if optimizer_step < 0 or total_optimizer_steps < 1:
        raise ValueError("invalid optimizer-step schedule")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must be in [0, 1]")
    warmup_steps = math.ceil(total_optimizer_steps * warmup_fraction)
    if warmup_steps == 0:
        return 1.0
    return min(max(optimizer_step / float(warmup_steps), 0.0), 1.0)


def flatten_span_groups(
    groups: Sequence[torch.Tensor],
    *,
    maximum_events: int,
) -> tuple[list[tuple[int, int]], list[int], int]:
    """Flatten event spans while retaining each event's graph-layer index."""

    if maximum_events < 1:
        raise ValueError("maximum_events must be positive")
    spans: list[tuple[int, int]] = []
    layer_indices: list[int] = []
    total = 0
    for layer_index, group in enumerate(groups):
        for start, end in group.reshape(-1, 2).tolist():
            total += 1
            if len(spans) >= maximum_events:
                continue
            spans.append((int(start), int(end)))
            layer_indices.append(layer_index)
    return spans, layer_indices, total


def multiscale_time_points(
    layer_indices: Sequence[int],
    *,
    layer_dt: float,
    substep_dt: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create one slow boundary increment and fast within-layer increments.

    The first target event follows the observed prefix and therefore crosses a
    graph-layer boundary.  Later events in the same layer use ``substep_dt``;
    the first event of each later layer again uses ``layer_dt``.  Canonical
    within-layer order is a reproducible traversal, not a measured biological
    timestamp.
    """

    if layer_dt <= 0 or substep_dt <= 0:
        raise ValueError("time increments must be positive")
    if substep_dt >= layer_dt:
        raise ValueError("substep_dt must be smaller than layer_dt")
    values = [0.0]
    previous: int | None = None
    current = 0.0
    for layer_index in layer_indices:
        if layer_index < 0:
            raise ValueError("layer indices must be non-negative")
        increment = layer_dt if previous is None or layer_index != previous else substep_dt
        current += increment
        values.append(current)
        previous = layer_index
    return torch.tensor(values, dtype=dtype, device=device)


def gradient_conflict_statistics(
    loss_sft: torch.Tensor,
    loss_dynamics: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    dynamics_gradient_scale: float = 1.0,
) -> dict[str, float]:
    """Measure the LoRA gradient angle without mutating ``parameter.grad``.

    ``dynamics_gradient_scale`` removes the known warm-up multiplier introduced
    by :func:`scale_gradient`, so the diagnostic reflects the underlying
    objective rather than the current routing coefficient.
    """

    selected = [parameter for parameter in parameters if parameter.requires_grad]
    if not selected:
        return {"cosine": 0.0, "angle_degrees": 90.0, "sft_norm": 0.0, "dynamics_norm": 0.0}
    if dynamics_gradient_scale <= 0:
        raise ValueError("dynamics_gradient_scale must be positive for diagnostics")
    sft_grads = torch.autograd.grad(
        loss_sft,
        selected,
        retain_graph=True,
        allow_unused=True,
    )
    dynamics_grads = torch.autograd.grad(
        loss_dynamics,
        selected,
        retain_graph=True,
        allow_unused=True,
    )
    device = loss_sft.device
    dot = torch.zeros((), dtype=torch.float64, device=device)
    sft_sq = torch.zeros_like(dot)
    dynamics_sq = torch.zeros_like(dot)
    for sft_grad, dynamics_grad in zip(sft_grads, dynamics_grads):
        if sft_grad is None or dynamics_grad is None:
            continue
        left = sft_grad.detach().double().reshape(-1)
        right = dynamics_grad.detach().double().reshape(-1) / float(dynamics_gradient_scale)
        dot += torch.dot(left, right)
        sft_sq += torch.dot(left, left)
        dynamics_sq += torch.dot(right, right)
    sft_norm = sft_sq.sqrt()
    dynamics_norm = dynamics_sq.sqrt()
    denominator = sft_norm * dynamics_norm
    cosine = torch.where(denominator > 0, dot / denominator, torch.zeros_like(dot)).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return {
        "cosine": float(cosine.item()),
        "angle_degrees": float(angle.item()),
        "sft_norm": float(sft_norm.item()),
        "dynamics_norm": float(dynamics_norm.item()),
    }


def finite_metric_record(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "DYNAMICS_RESOLUTIONS",
    "finite_metric_record",
    "flatten_span_groups",
    "gradient_conflict_statistics",
    "linear_warmup_scale",
    "multiscale_time_points",
    "scale_gradient",
]
