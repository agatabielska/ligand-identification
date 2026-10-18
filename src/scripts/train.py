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
from src.models.e3nn.model import E3NNPointCloudModel
from src.pipeline.dataset import BlobDataset
from src.utils.set_seed import set_seed

torch.set_float32_matmul_precision("high")  # TF32 matmuls

# Model classes are imported lazily in build_model, so the script runs with whichever model files exist.
# Models that take class counts (for the log-prior used by logit adjustment).
POINT_CLOUD_TYPES = ("e3nn", "spectral_baseline", "spectral")
# Models that use the top-density transform + GPU-resident loader.
GPU_TYPES = POINT_CLOUD_TYPES + ("e3nn_graph",)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def transform_clifford(npz, max_points: int = 2000):
    """Order points by (distance from centroid, x, y, z); pad to (4, max_points) = [x, y, z, density]."""
    c = npz["indices"].astype(np.float32)
    v = npz["values"].astype(np.float32)
    radius = np.linalg.norm(c - c.mean(axis=0), axis=1)
    order = np.lexsort((c[:, 2], c[:, 1], c[:, 0], radius))[:max_points]

    out = np.zeros((4, max_points), np.float32)
    out[:3, : len(order)] = c[order].T
    out[3, : len(order)] = v[order]
    return torch.from_numpy(out)


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
    classes = sorted({d.name for s in splits for d in (data_dir / s).iterdir() if d.is_dir()})
    label_to_idx = {name: i for i, name in enumerate(classes)}

    if model_type == "clifford":
        transform = transform_clifford
    else:
        transform = make_top_density_transform(cfg.model.get("max_points", 2000))

    datasets = {s: get_dataset(data_dir / s, cfg, transform, label_to_idx) for s in splits}

    if cfg.train.get("train_on_val", False):
        # Final refit on train+val. Test stays untouched; val metrics become optimistic.
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

    if model_type == "e3nn_graph":
        from src.models.e3nn.graph_model import E3NNGraphModel

        return E3NNGraphModel(
            num_classes=t.out_channels,
            learning_rate=t.learning_rate,
            weight_decay=t.weight_decay,
            n_points=m.get("n_points", 256),
            k=m.get("k", 12),
            mul=m.get("mul", 32),
            lmax=m.get("lmax", 2),
            n_layers=m.get("n_layers", 3),
            num_basis=m.get("num_basis", 8),
            r_max=m.get("r_max", 1.0),
            hidden=m.get("hidden", 512),
            dropout=m.get("dropout", 0.2),
            label_smoothing=m.get("label_smoothing", 0.0),
            point_dropout=m.get("point_dropout", 0.1),
        )

    # e3nn point-cloud family: "e3nn", "spectral_baseline", "spectral"
    common = dict(
        num_classes=t.out_channels,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        batch_norm_momentum=m.get("batch_norm_momentum", 0.1),
        n_shells=m.get("n_shells", 16),
        lmax=m.get("lmax", 6),
        r_max=m.get("r_max", 1.0),
        hidden=m.get("hidden", 1024),
        dropout=m.get("dropout", 0.2),
        label_smoothing=m.get("label_smoothing", 0.0),
        warmup_frac=m.get("warmup_frac", 0.03),
        full_train_metrics=m.get("full_train_metrics", False),
        point_dropout=m.get("point_dropout", 0.1),
        jitter=m.get("jitter", 0.0),
        chunk_size=m.get("chunk_size", 128),
        logit_adjust_tau=m.get("logit_adjust_tau", 0.0),
    )
    spectral = dict(n_mix=m.get("n_mix", 16), bispec_lmax=m.get("bispec_lmax", 6))

    if model_type == "spectral":
        from src.models.e3nn.spectral_model import SpectralModel

        model = SpectralModel(
            **spectral,
            use_globals=m.get("use_globals", False),
            gram_all=m.get("gram_all", False),
            feat_dropout=m.get("feat_dropout", 0.0),
            density_noise=m.get("density_noise", 0.0),
            **common,
        )
    elif model_type == "spectral_baseline":
        from src.models.e3nn.baseline_spectral_model import BaselineSpectralModel

        model = BaselineSpectralModel(**spectral, **common)  # forces use_equivariant=False
    else:
        model = E3NNPointCloudModel(
            use_equivariant=m.get("use_equivariant", True),
            lh=m.get("lh", 3),
            n_scalars=m.get("n_scalars", 128),
            mul_gated=m.get("mul_gated", 24),
            n_blocks=m.get("n_blocks", 2),
            **common,
        )
    model.set_class_counts(torch.bincount(train_loader.y.cpu(), minlength=t.out_channels))
    return model


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.random_seed)

    model_type = cfg.model.type
    if model_type != "clifford" and model_type not in GPU_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")

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