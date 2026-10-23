"""Dataset transforms and loaders used by the training entry points."""

import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

from src.pipeline.dataset import BlobDataset


MAX_POINTS = 2000
POINT_CLOUD_TYPES = ("spectral",)


def transform_clifford(npz, max_points: int = MAX_POINTS):
    """Prepare coordinate-only input with shape ``(3, max_points)``."""
    indices = npz["indices"]
    if indices.shape[0] > max_points:
        raise ValueError(f"Clifford input contains {indices.shape[0]} points; maximum is {max_points}")
    points = np.pad(
        indices,
        ((0, max_points - indices.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return torch.from_numpy(points.astype(np.float32).transpose(1, 0))


def transform_e3nn(npz, max_points: int = MAX_POINTS):
    """Keep the highest-density points and return a zero-padded spectral tensor."""
    coords = npz["indices"].astype(np.float32)
    density = np.maximum(npz["values"].astype(np.float32), 0)
    if coords.shape[0] > max_points:
        keep = np.argsort(-density, kind="stable")[:max_points]
        coords = coords[keep]
        density = density[keep]
    points = np.zeros((max_points, 4), dtype=np.float32)
    points[: coords.shape[0], :3] = coords
    points[: coords.shape[0], 3] = density
    return torch.from_numpy(points)


class GPULoader:
    """Keep a split on the selected device when memory permits."""

    def __init__(self, dataset, batch_size: int, shuffle: bool, device, num_workers: int = 0):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=min(num_workers, 8))
        samples = [(x, torch.as_tensor(y).reshape(-1)) for x, y in loader]
        x = torch.cat([sample[0] for sample in samples])
        y = torch.cat([sample[1] for sample in samples]).long()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.on_gpu = False
        if device.type == "cuda":
            try:
                self.x, self.y = x.to(device), y.to(device)
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
            if not self.on_gpu:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            yield x, y


def get_dataset(path: Path, cfg: DictConfig, transform, label_to_idx: dict):
    return BlobDataset(
        path=str(path),
        transform=transform,
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
    data_dir = Path(cfg.paths.data_dir)
    validate_data_dir(data_dir)
    splits = ("train", "val", "test")
    classes = sorted(
        {directory.name for split in splits for directory in (data_dir / split).iterdir() if directory.is_dir()}
    )
    label_to_idx = {name: index for index, name in enumerate(classes)}
    transform = transform_clifford if model_type == "clifford" else transform_e3nn
    datasets = {
        split: get_dataset(data_dir / split, cfg, transform, label_to_idx)
        for split in splits
    }
    if cfg.train.get("train_on_val", False):
        datasets["train"] = ConcatDataset([datasets["train"], datasets["val"]])

    batch_size, workers = cfg.machine.batch_size, cfg.machine.num_workers
    if model_type in POINT_CLOUD_TYPES:
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
