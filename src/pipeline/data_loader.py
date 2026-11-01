"""Dataset transforms and loaders used by the training entry points."""

import math
from pathlib import Path
from functools import partial
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.pipeline.dataset import BlobDataset
from src.pipeline.global_statistics import complete_input_statistics


MAX_POINTS = 2000


def transform_clifford(npz, max_points: int = MAX_POINTS):
    """Prepare coordinate-only input with shape ``(3, max_points)``."""
    indices = npz["indices"].astype(np.float32)
    if indices.shape[0] > max_points:
        raise ValueError(f"Clifford input contains {indices.shape[0]} points; maximum is {max_points}.")
    points = np.zeros((max_points, 3), dtype=np.float32)
    points[: indices.shape[0], :3] = indices
    return torch.from_numpy(points.transpose(1, 0))


def transform_e3nn(npz, max_points: int = MAX_POINTS):
    """Prepare coordinate and density input with shape ``(max_points, 4)``."""
    coords = npz["indices"].astype(np.float32)
    if coords.shape[0] > max_points:
        raise ValueError(f"e3nn input contains {coords.shape[0]} points; maximum is {max_points}.")
    density = np.maximum(npz["values"].astype(np.float32), 0)
    points = np.zeros((max_points, 4), dtype=np.float32)
    points[: coords.shape[0], :3] = coords
    points[: coords.shape[0], 3] = density
    return torch.from_numpy(points)

class GPULoader:
    """Keep a split on the selected device when memory permits."""

    def __init__(self, dataset, batch_size: int, shuffle: bool, device, num_workers: int = 0):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=min(num_workers, 8))
        samples = [(x, torch.as_tensor(y).reshape(-1), *rest) for x, y, *rest in loader]
        x = torch.cat([sample[0] for sample in samples])
        y = torch.cat([sample[1] for sample in samples]).long()
        
        g = None
        if samples[0][2] is not None:
            g = torch.cat([sample[2] for sample in samples])

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.on_gpu = False
        if device.type == "cuda":
            try:
                self.x, self.y = x.to(device), y.to(device)
                self.g = g.to(device) if g is not None else None
                self.on_gpu = True
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(
                    f"[GPULoader] GPU memory is insufficient for {x.numel() * 4 / 1e9:.1f} GB; "
                    "using pinned CPU storage."
                )
        if not self.on_gpu:
            self.x = x.pin_memory() if device.type == "cuda" else x
            self.y = y
            self.g = (g.pin_memory() if device.type == "cuda" else g) if g is not None else None

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        count = len(self.y)
        index_device = self.device if self.on_gpu else "cpu"
        indices = (
            torch.randperm(count, device=index_device)
            if self.shuffle
            else torch.arange(count, device=index_device)
        )
        for start in range(0, count, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            x, y = self.x[batch_indices], self.y[batch_indices]
            g = self.g[batch_indices] if self.g is not None else None
            if not self.on_gpu:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                g = g.to(self.device, non_blocking=True) if g is not None else None
            yield (x, y, g) if g is not None else (x, y)


def get_dataset(path: Path, cfg: DictConfig, transform, label_to_idx: dict, global_transform=None):
    return BlobDataset(
        path=str(path),
        transform=transform,
        global_transform=global_transform,
        normalize=cfg.train.normalize_data,
        cache=cfg.machine.cache_dataset,
        num_workers=cfg.machine.num_workers,
        label_to_idx=label_to_idx,
    )


def validate_data_dir(data_dir: Path):
    missing = [
        str(data_dir / split)
        for split in ("train", "val", "test")
        if not (data_dir / split).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"Configured paths.data_dir={data_dir} must contain train/, val/, and test/ directories. "
            f"Missing: {', '.join(missing)}."
        )


def build_data(cfg: DictConfig, model_type: str):
    """Build train, validation, and test loaders for a configured model."""
    max_points = cfg.data.max_points
    match model_type:
        case "clifford":
            transform = partial(transform_clifford, max_points=max_points)
            global_transform = None
        case "spectral":
            transform = partial(transform_e3nn, max_points=max_points)
            global_transform = complete_input_statistics
        case _:
            raise ValueError(f"Unknown model type: {model_type}")   
    data_dir = Path(cfg.paths.data_dir)
    splits = ("train", "val", "test")
    classes = sorted(
        {directory.name for split in splits for directory in (data_dir / split).iterdir() if directory.is_dir()}
    )
    label_to_idx = {name: index for index, name in enumerate(classes)}
    datasets = {
        split: get_dataset(data_dir / split, cfg, transform, label_to_idx, global_transform=global_transform)
        for split in splits
    }

    batch_size, workers = cfg.machine.batch_size, cfg.machine.num_workers
    if cfg.machine.use_gpu_loader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return tuple(
            GPULoader(datasets[split], batch_size, split == "train", device, workers)
            for split in splits
        )
    return tuple(
        DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers if not cfg.machine.cache_dataset else 1,
            pin_memory=cfg.machine.pin_memory,
            persistent_workers=workers > 0,
        )
        for split in splits
    )
