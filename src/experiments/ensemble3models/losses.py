"""Losses used by the stacked-head experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


DEFINED_LOSSES = (
    {"name": "baseline", "label_smoothing": 0.05},
    {"name": "weighted_ce"},
    {"name": "baseline", "label_smoothing": 0.1},
    {"name": "baseline", "label_smoothing": 0.125},
    {"name": "baseline", "label_smoothing": 0.15}, 
    {"name": "logit_adjusted_ce", "tau": 0.5}, 
    {"name": "logit_adjusted_ce", "tau": 0.75},
    {"name": "logit_adjusted_ce", "tau": 1.0},
    {"name": "focal", "gamma": 0.5},
    {"name": "focal", "gamma": 1.0},
    {"name": "focal", "gamma": 1.5},
    {"name": "focal", "gamma": 2.0},
)


def _class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    weights = class_counts.float().clamp_min(1).rsqrt()
    return weights / weights.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: torch.Tensor | None = None):
        super().__init__()
        if gamma < 0:
            raise ValueError("Focal-loss gamma must be non-negative.")
        self.gamma = gamma
        # Register alpha as a buffer so it moves to the correct device automatically
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Compute standard cross-entropy without reduction to get -log(p_t)
        ce_loss = F.cross_entropy(logits, target, reduction="none")
        
        # Keep fractional powers and their gradients finite for confident predictions.
        eps = torch.finfo(logits.dtype).eps
        p_t = torch.exp(-ce_loss).clamp(min=eps, max=1.0 - eps)
        
        # 3. Compute the modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # 4. Apply focal weight
        focal_loss = modulating_factor * ce_loss
        
        # 5. Apply alpha weighting if provided (balances class frequencies)
        if self.alpha is not None:
            # Gather alpha weights corresponding to each target class in the batch
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


def build_loss(
    loss_cfg: DictConfig | None = None,
    class_counts: torch.Tensor | None = None,
    log_prior: torch.Tensor | None = None,
) -> nn.Module:
    """Build a loss while keeping class statistics derived from the train split."""
    name = str((loss_cfg or {}).get("name", "baseline"))
    if name == "baseline":
        return nn.CrossEntropyLoss(
            label_smoothing=float((loss_cfg or {}).get("label_smoothing", 0.05))
        )
    if name == "weighted_ce":
        if class_counts is None:
            raise ValueError("weighted_ce requires training class counts.")
        return nn.CrossEntropyLoss(weight=_class_weights(class_counts))
    if name == "logit_adjusted_ce":
        if log_prior is None:
            raise ValueError("logit_adjusted_ce requires training class priors.")
        tau = float((loss_cfg or {}).get("tau", 1.0))

        class LogitAdjustedCrossEntropy(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("log_prior", log_prior.detach().clone())

            def forward(self, logits, target):
                return F.cross_entropy(logits - tau * self.log_prior, target)

        return LogitAdjustedCrossEntropy()
    if name == "focal":
        if class_counts is None:
            raise ValueError("focal loss requires training class counts for alpha weighting.")
        gamma = float((loss_cfg or {}).get("gamma", 2.0))
        
        # Inverse class frequency for alpha, normalized to average to 1.0
        alpha = _class_weights(class_counts)
        return FocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(
        f"Unknown loss {name!r}; expected baseline, weighted_ce, "
        "logit_adjusted_ce, or focal."
    )
