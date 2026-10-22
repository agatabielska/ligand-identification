"""Training entry point: builds data, model and trainer from a Hydra config."""

from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

from src.models.clifford.model import CliffordSteerableNetwork
from src.pipeline.dataset import BlobDataset
from src.utils.set_seed import set_seed

torch.set_float32_matmul_precision("high")  # TF32 matmuls

# Model classes are imported lazily in build_model, so the script runs with whichever model files exist.
# Models that take class counts (for the log-prior used by logit adjustment).
POINT_CLOUD_TYPES = (
    "points-2000-r3", "raw-npz-scalars",
)
# Models that use the top-density transform + GPU-resident loader.
GPU_TYPES = POINT_CLOUD_TYPES

# Fixed on purpose: every blob is reduced to its 2000 densest points, for every model and every cache.
MAX_POINTS = 2000


def check_config(cfg: DictConfig):
    """Refuse configs that would silently invalidate results (called before any logger / data is created)."""
    mp = cfg.model.get("max_points", MAX_POINTS)
    if mp != MAX_POINTS:
        raise ValueError(f"model.max_points={mp}, but max_points is fixed at {MAX_POINTS}. Remove the override.")


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def transform_clifford(npz, max_points: int = MAX_POINTS):
    """Prepare a coordinate-only Clifford input with shape (3, max_points)."""
    points = np.pad(
        npz["indices"],
        ((0, max_points - npz["indices"].shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return torch.from_numpy(points.astype(np.float32).transpose(1, 0))


def make_top_density_transform(max_points: int):
    """Keep the max_points densest points, zero-padded (density 0 => masked in the model).

    The e3nn models are permutation invariant, so no ordering is needed.
    Returns a (max_points, 4) tensor: [x, y, z, density].
    """

    def transform(npz):
        c = npz["indices"].astype(np.float32)
        v = npz["values"].astype(np.float32)
        keep = np.argsort(-v)[:max_points]
        out = np.zeros((max_points, 4), np.float32)
        out[: len(keep), :3] = c[keep]
        out[: len(keep), 3] = v[keep]
        return torch.from_numpy(out)

    return transform


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class GPULoader:
    """Whole split lives on the GPU; a batch is just an index slice (no workers, no H2D copies).

    If the GPU lacks free memory, falls back to pinned CPU memory and copies each batch over.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool, device, num_workers: int = 0):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=min(num_workers, 8))
        xs, ys = zip(*[(x, torch.as_tensor(y).reshape(-1)) for x, y in loader])
        x, y = torch.cat(xs), torch.cat(ys).long()

        self.bs, self.shuffle, self.device = batch_size, shuffle, device
        self.on_gpu = False
        if device.type == "cuda":
            try:
                self.x, self.y = x.to(device), y.to(device)
                self.on_gpu = True
            except Exception as e:  # CUDA OOM surfaces as AcceleratorError / OutOfMemoryError
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(
                    f"[GPULoader] not enough free GPU memory for {x.numel() * 4 / 1e9:.1f} GB, "
                    "falling back to pinned CPU memory (check nvidia-smi for other processes)."
                )
        if not self.on_gpu:
            self.x = x.pin_memory() if device.type == "cuda" else x
            self.y = y

    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.y)
        where = self.device if self.on_gpu else "cpu"
        idx = torch.randperm(n, device=where) if self.shuffle else torch.arange(n, device=where)
        for i in range(0, n, self.bs):
            b = idx[i : i + self.bs]
            xb, yb = self.x[b], self.y[b]
            if not self.on_gpu:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            yield xb, yb


def get_dataset(path: Path, cfg: DictConfig, transform, label_to_idx: dict):
    return BlobDataset(
        path=str(path),
        transform=transform,
        normalize=cfg.train.normalize_data,
        cache=cfg.machine.cache_dataset,
        num_workers=cfg.machine.num_workers,
        label_to_idx=label_to_idx,
    )


def build_data(cfg: DictConfig, model_type: str):
    """Returns (train, val, test) loaders for the given model type."""
    data_dir = Path(cfg.paths.data_dir)
    splits = ("train", "val", "test")
    validate_data_dir(data_dir)
    classes = sorted({d.name for s in splits for d in (data_dir / s).iterdir() if d.is_dir()})
    label_to_idx = {name: i for i, name in enumerate(classes)}

    if model_type == "clifford":
        transform = transform_clifford
    else:
        transform = make_top_density_transform(MAX_POINTS)

    datasets = {s: get_dataset(data_dir / s, cfg, transform, label_to_idx) for s in splits}
    # Preserve the original optional final-refit behavior for Clifford runs.
    if cfg.train.get("train_on_val", False):
        datasets["train"] = ConcatDataset([datasets["train"], datasets["val"]])

    bs, nw = cfg.machine.batch_size, cfg.machine.num_workers
    if model_type in GPU_TYPES:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return tuple(GPULoader(datasets[s], bs, s == "train", device, nw) for s in splits)

    return tuple(
        DataLoader(
            datasets[s],
            batch_size=bs,
            shuffle=(s == "train"),
            num_workers=nw if not cfg.machine.cache_dataset else 1,
            pin_memory=cfg.machine.pin_memory,
            persistent_workers=nw > 0,
        )
        for s in splits
    )


def validate_data_dir(data_dir: Path):
    """Fail before W&B setup when the configured dataset is not a split dataset."""
    missing = [str(data_dir / split) for split in ("train", "val", "test")
               if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Configured paths.data_dir={data_dir} must contain train/, val/, and test/ directories. "
            f"Missing: {', '.join(missing)}. "
            "Override it, for example: paths.data_dir=data/unthresholded_transform_probabilistic"
        )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_model(cfg: DictConfig, model_type: str, train_loader):
    m, t = cfg.model, cfg.train

    if model_type == "clifford":
        return CliffordSteerableNetwork(
            p=m.p,
            q=m.q,
            in_channels=m.in_channels,
            hidden_channels=m.hidden_channels,
            out_channels=t.out_channels,
            n_shells=m.n_shells,
            kernel_size=m.kernel_size,
            learning_rate=t.learning_rate,
            weight_decay=t.weight_decay,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    check_config(cfg)
    set_seed(cfg.random_seed)

    model_type = cfg.model.type
    if model_type != "clifford" and model_type not in GPU_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    validate_data_dir(Path(cfg.paths.data_dir))

    if model_type in {"points-2000-r3", "raw-npz-scalars"}:
        from src.experiments.spectral_experiments import run_default_from_config

        mode = "raw_npz" if model_type == "raw-npz-scalars" else "points_2000"
        run_default_from_config(cfg, globals_mode=mode)
        return

    logger_kwargs = {
        "project": "ligand-identification",
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if cfg.wandb.run_name is not None:
        logger_kwargs["name"] = cfg.wandb.run_name
    logger = WandbLogger(**logger_kwargs)

    ckpt_dir = Path(cfg.paths.model_checkpoint) / logger.experiment.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_data(cfg, model_type)
    model = build_model(cfg, model_type, train_loader)

    es = cfg.train.early_stopping
    on_gpu = model_type in GPU_TYPES  # GPU-resident loader is single-device
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=[
            ModelCheckpoint(
                monitor=es.monitor,
                mode=es.mode,
                dirpath=str(ckpt_dir),
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(monitor=es.monitor, mode=es.mode, patience=es.patience, verbose=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        accelerator="auto",
        devices=1 if on_gpu else cfg.machine.devices,
        precision=cfg.train.get("precision", "bf16-mixed" if on_gpu else "32-true"),
        log_every_n_steps=10,
        logger=logger,
        gradient_clip_val=cfg.train.gradient_clip_val,
        use_distributed_sampler=False,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":
    main()