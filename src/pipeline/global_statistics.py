"""Global scalar feature computation for spectral training."""

import math
from pathlib import Path

import numpy as np
import torch

from src.pipeline.dataset import BlobDataset


def point_statistics(x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Compute 12 scalar features from normalized, padded point tensors."""
    features = []
    for batch in x.split(batch_size):
        points, density = batch[..., :3].float(), batch[..., 3].float().clamp_min(0)
        valid = density > 0
        count = valid.sum(1).float()
        total = density.sum(1)
        weights = density / total.clamp_min(1e-8)[:, None]
        center = (points * weights[..., None]).sum(1, keepdim=True)
        centered = points - center
        radius = centered.norm(dim=-1).masked_fill(~valid, 0)
        covariance = torch.einsum("bn,bni,bnj->bij", weights, centered, centered)
        covariance += 1e-8 * torch.eye(3, device=batch.device, dtype=batch.dtype)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).sqrt()
        valid_points = valid.unsqueeze(-1)
        mins = points.masked_fill(~valid_points, 0).amin(1)
        maxs = points.masked_fill(~valid_points, 0).amax(1)
        bounds = (maxs - mins).sort(dim=1).values
        counts = torch.stack(
            (torch.log1p(count), torch.log1p(total), torch.log1p(density.amax(1))),
            dim=1,
        )
        radii = torch.stack(
            (
                radius.amax(1),
                (weights * radius.square()).sum(1).clamp_min(0).sqrt(),
                (weights * radius).sum(1),
            ),
            dim=1,
        )
        features.append(torch.cat((counts, radii, eigenvalues, bounds), dim=1).cpu())
    return torch.cat(features)


def complete_input_statistics(npz) -> torch.Tensor:
    """Compute 12 scalar features from all entries in one sparse input blob."""
    coordinates = npz["indices"].astype(np.float64)
    density = np.maximum(npz["values"].astype(np.float64), 0)
    total = density.sum()
    weights = density / max(total, 1e-12)
    centered = coordinates - (coordinates * weights[:, None]).sum(0)
    radius = np.linalg.norm(centered, axis=1)
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues = np.sqrt(np.clip(np.linalg.eigvalsh(covariance), 0, None))
    bounds = np.sort(coordinates.max(0) - coordinates.min(0))
    return torch.tensor(
        [
            math.log1p(len(density)),
            math.log1p(total),
            math.log1p(density.max(initial=0)),
            radius.max(initial=0),
            math.sqrt(max((weights * radius * radius).sum(), 0)),
            (weights * radius).sum(),
            *eigenvalues,
            *bounds,
        ],
        dtype=torch.float32,
    )


def complete_input_statistics_for_splits(
    data_dir: Path, label_to_idx: dict, workers: int,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load complete-input statistics in the same order as point datasets."""
    result = {}
    for split in splits:
        dataset = BlobDataset(
            path=str(data_dir / split),
            transform=complete_input_statistics,
            normalize=False,
            cache=True,
            num_workers=workers,
            label_to_idx=label_to_idx,
        )
        result[split] = (
            torch.stack([item[0] for item in dataset.cached_data]),
            torch.tensor([item[1] for item in dataset.cached_data], dtype=torch.long),
        )
    return result
